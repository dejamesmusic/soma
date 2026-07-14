"""isolated training worker for the soma gui.

the gui launches this as a plain child process so torch/mps training state
dies with the process instead of accumulating inside the tkinter app.
"""

import gc
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

ORIG_STDOUT = sys.stdout

def emit(kind, *items):
    print(json.dumps([kind, *items], ensure_ascii=True),
          file=ORIG_STDOUT, flush=True)


def release_torch_memory(device_type=None):
    try:
        import torch
        if device_type == "mps" and hasattr(torch, "mps"):
            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
    except Exception:
        pass
    gc.collect()


class Worker:
    def __init__(self, config_path, stop_path):
        self.config_path = Path(config_path)
        self.stop_path = Path(stop_path)
        self.stop = False
        self.save_on_stop = True
        self.model = None
        self.last_memory_hygiene = 0.0
        self.autosave_interval_s = 0.0
        self.last_save_t = [time.time()]
        self.dream_every_batches = 0
        self.dream_length = "200"
        self.dream_temperature = 0.8
        self.start_byte = 0

    def read_stop(self):
        if not self.stop_path.exists():
            return False
        self.stop = True
        try:
            data = json.loads(self.stop_path.read_text(encoding="utf-8"))
            self.save_on_stop = bool(data.get("save", True))
        except Exception:
            self.save_on_stop = True
        return True

    def memory_hygiene(self, model=None, force=False):
        now = time.time()
        if not force and now - self.last_memory_hygiene < 60.0:
            return
        self.last_memory_hygiene = now
        device = getattr(getattr(model, "device", None), "type", None)
        release_torch_memory(device)

    def quiet(self, fn, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kwargs)

    def release_model(self):
        model = self.model
        device = getattr(getattr(model, "device", None), "type", None)
        self.model = None
        if model is not None:
            del model
        release_torch_memory(device)

    def run(self):
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        try:
            import soma_v12_2 as soma_runtime
        except Exception as e:
            emit("error", f"import failed: {e}")
            return

        try:
            emit("start")
            ckpt = config.get("checkpoint")
            lr_val, lr_auto, lr_base = soma_runtime._parse_auto_or_float(
                config.get("lr", "auto 0.001"), 0.001)
            try:
                grad_clip = float(config.get("grad_clip", "1.0"))
            except (TypeError, ValueError):
                grad_clip = 1.0
            auto_mode = str(config.get(
                "auto_mode", soma_runtime.DEFAULT_AUTO_MODE)).strip().lower()
            if auto_mode not in ("wallclock", "model", "io2", "off"):
                auto_mode = soma_runtime.DEFAULT_AUTO_MODE
            try:
                decimation_range = float(config.get(
                    "decimation",
                    soma_runtime.default_decimation_range(auto_mode)))
            except (TypeError, ValueError):
                decimation_range = soma_runtime.default_decimation_range(auto_mode)

            if ckpt and Path(ckpt).exists():
                model = soma_runtime.SOMA(device="auto")
                model.load(ckpt)
                loaded_auto_mode = model.auto_mode
                model.description = str(
                    config.get("description", model.description))
                model.lr = lr_val
                model.lr_base = lr_base
                model.lr_auto = lr_auto
                model.auto_mode = auto_mode
                model.decimation_range = decimation_range
                model.grad_clip = grad_clip
                model.batch_size = int(config["batch"])
                for group in model.opt.param_groups:
                    group["lr"] = model.lr
                model._update_decimation()
                # A wallclock checkpoint carries a learned centre. Only
                # install the fixed initial centre when the user explicitly
                # switches an older/non-wallclock model into this mode.
                if (model.auto_mode == "wallclock"
                        and loaded_auto_mode != "wallclock"):
                    model._configure_wallclock_stride()
                emit("info", f"resumed {Path(ckpt).name}")
            else:
                model = soma_runtime.SOMA(
                    n_bands=int(config["bands"]),
                    base=float(config["base"]),
                    hidden_dim=int(config["hidden"]),
                    depth=int(config.get("layers", 3)),
                    lr=lr_val,
                    lr_base=lr_base,
                    lr_auto=lr_auto,
                    auto_mode=auto_mode,
                    decimation_range=decimation_range,
                    grad_clip=grad_clip,
                    batch_size=int(config["batch"]),
                    description=str(config.get("description", "")),
                    device="auto",
                )
                emit("info", "fresh model")

            self.model = model
            n_params = self.param_count(model)
            hidden_str = (f"hidden={model.hidden_dim:,}"
                          if model.hidden_dim > 0 else "linear")
            if hasattr(model, "depth"):
                hidden_str += f" · depth={model.depth}"

            corpus_name = config.get("corpus_label") or Path(config["corpus"]).name
            emit("summary", {
                "corpus": corpus_name,
                "bands": model.n_bands,
                "base": model.base,
                "hidden": hidden_str,
                "params": n_params,
                "device": str(model.device),
                "dtype": str(getattr(
                    model.bank, "dtype",
                    getattr(model.bank.traces, "dtype", "unknown")
                )).replace("torch.", ""),
                "batch": model.batch_size,
                "lr": getattr(model, "lr", 0.001),
                "lr_base": getattr(model, "lr_base", 0.001),
                "lr_auto": getattr(model, "lr_auto", True),
                "grad_clip": getattr(model, "grad_clip", 1.0),
                "description": getattr(model, "description", ""),
                "auto_mode": getattr(
                    model, "auto_mode", soma_runtime.DEFAULT_AUTO_MODE),
                "decimation_range": getattr(
                    model, "decimation_range",
                    soma_runtime.default_decimation_range(
                        soma_runtime.DEFAULT_AUTO_MODE)),
                "decimation_stride_cap": getattr(
                    model, "decimation_stride_cap", 1024),
                "bytes_seen": model.bytes_seen,
                "save_path": Path(config["save_path"]).name,
                "loop": bool(config.get("loop", False)),
            })

            autosave_min = float(config.get("autosave_min", 0.5))
            self.autosave_interval_s = max(0.0, autosave_min * 60.0)
            self.last_save_t = [time.time()]
            self.dream_every_batches = int(config.get("dream_every_batches", 0))
            self.dream_length = str(config.get("dream_length", "200"))
            self.dream_temperature = float(config.get("dream_temperature", 0.8))
            self.install_callbacks(model, config)

            if bool(config.get("loop", False)):
                self.run_loop(
                    model,
                    config["corpus"],
                    config["save_path"],
                    int(config.get("head_bytes", 50_000_000)),
                    float(config.get("cycle_pause", 60.0)),
                )
            elif config.get("cycle_files"):
                self.run_cycle(
                    model,
                    config["cycle_files"],
                    int(config.get("cycle_count", 1)),
                    config["save_path"],
                )
            else:
                self.run_single(
                    model,
                    config["corpus"],
                    config["save_path"],
                    int(config.get("start_byte", 0)),
                )
        except Exception as e:
            emit("error", f"{type(e).__name__}: {e}")
        finally:
            self.release_model()
            emit("done")
            try:
                self.config_path.unlink()
            except OSError:
                pass
            try:
                self.stop_path.unlink()
            except OSError:
                pass

    def param_count(self, model):
        if hasattr(model, "net") and hasattr(model, "params"):
            return model.params()
        if hasattr(model, "W_list"):
            n_params = sum(W.numel() for W in model.W_list)
            n_params += sum(U.numel() for U in getattr(model, "U_list", []))
            n_params += sum(G.numel() for G in getattr(model, "G_list", []))
            n_params += sum(gb.numel() for gb in getattr(model, "gb_list", []))
            if model.Wd is not None:
                n_params += model.Wd.numel()
            return n_params
        if model.hidden_dim > 0:
            n_params = model.U.numel() + model.W.numel()
            if model.Wd is not None:
                n_params += model.Wd.numel()
            return n_params
        return model.W.numel()

    def install_callbacks(self, model, config):
        ln2 = math.log(2)
        session_t0 = time.time()
        session_bytes_start = model.bytes_seen
        last_emit_t = [0.0]
        min_emit_s = 0.15
        save_path = config["save_path"]
        self.start_byte = int(config.get("start_byte", 0))

        def patched_report(epoch, epochs, pos, total, loss,
                           correct, samples, t0):
            should_stop = self.read_stop()
            now = time.time()
            if should_stop or now - last_emit_t[0] >= min_emit_s:
                cum_avg = loss / samples if samples > 0 else 0
                life_dt = max(1e-9, now - session_t0)
                life_bytes = model.bytes_seen - session_bytes_start
                emit("report", {
                    "pos": int(pos),
                    "total": int(total),
                    "absolute_pos": int(self.start_byte + pos),
                    "absolute_total": int(self.start_byte + total),
                    "loss": float(cum_avg),
                    "bpb": float(cum_avg / ln2),
                    "acc": float(100 * correct / samples if samples > 0 else 0),
                    "bps": float(pos / max(1e-9, now - t0)),
                    "life_bytes": int(life_bytes),
                    "life_bps": float(life_bytes / life_dt),
                    "life_seconds": float(life_dt),
                    "epoch": int(epoch + 1),
                    "epochs": int(epochs),
                    "lr": float(getattr(model, "lr", 0.001)),
                    "auto_mode": getattr(
                        model, "auto_mode", soma_runtime.DEFAULT_AUTO_MODE),
                    "grad_clip": float(getattr(model, "grad_clip", 1.0)),
                    "decimation_band": float(getattr(model, "decimation_band", 0.0)),
                    "stride": int(getattr(model, "_stride", 1)),
                    "sampled_stride": int(getattr(
                        model, "_sampled_stride",
                        getattr(model, "_stride", 1))),
                    "io2": float(getattr(model, "_io2_plasticity", 0.0)),
                    "motor": float(getattr(model, "_motor_value", 0.0)),
                    "motor_delta": float(getattr(
                        model, "_motor_delta", 0.0)),
                    "motor_energy": float(getattr(
                        model, "_motor_energy_push", 0.0)),
                    "row_clip": float(getattr(model, "_row_clip_fraction", 0.0)),
                })
                last_emit_t[0] = now
            if should_stop:
                raise KeyboardInterrupt("gui requested stop")

            self.memory_hygiene(model)
            if (self.autosave_interval_s > 0
                    and now - self.last_save_t[0] >= self.autosave_interval_s):
                try:
                    self.quiet(model.save, save_path)
                    self.memory_hygiene(model, force=True)
                    emit("autosaved", save_path)
                except Exception as e:
                    emit("error", f"autosave failed: {e}")
                self.last_save_t[0] = now

        model._report = patched_report
        model._dream_callback = lambda text, batch, seen: emit("dream", {
            "text": text,
            "batch": batch,
            "bytes_seen": seen,
        })

    def run_single(self, model, corpus_path, save_path, start_byte=0):
        try:
            self.quiet(model.train,
                corpus_path,
                epochs=1,
                save_every=0,
                save_path=save_path,
                start_byte=start_byte,
                report_every=1,
                dream_every_batches=self.dream_every_batches,
                dream_length=self.dream_length,
                dream_temperature=self.dream_temperature,
                dream_callback=model._dream_callback,
            )
        except KeyboardInterrupt:
            self.read_stop()
        self.final_save(model, save_path)

    def run_cycle(self, model, corpus_paths, cycle_count, save_path):
        files = [str(p) for p in corpus_paths]
        for cycle_idx in range(max(1, int(cycle_count))):
            if self.read_stop():
                break
            emit("info", f"cycle {cycle_idx + 1}/{cycle_count} · {len(files)} files")
            for idx, corpus_path in enumerate(files, 1):
                if self.read_stop():
                    break
                emit("info", f"file {idx}/{len(files)} · {Path(corpus_path).name}")
                try:
                    self.quiet(model.train,
                        corpus_path,
                        epochs=1,
                        save_every=0,
                        save_path=save_path,
                        start_byte=0,
                        report_every=1,
                        dream_every_batches=self.dream_every_batches,
                        dream_length=self.dream_length,
                        dream_temperature=self.dream_temperature,
                        dream_callback=model._dream_callback,
                    )
                except KeyboardInterrupt:
                    self.read_stop()
                    break
                except Exception as e:
                    emit("error", f"cycle file failed: {Path(corpus_path).name}: {e}")
        self.final_save(model, save_path)

    def run_loop(self, model, corpus_path, save_path, head_bytes, cycle_pause):
        tmp_head = Path(save_path).parent / f".{Path(save_path).stem}_head.tmp"

        def snapshot_head():
            corpus = Path(corpus_path)
            if not corpus.exists():
                return None
            n = min(head_bytes, corpus.stat().st_size)
            if n < 1_000_000:
                return None
            written = 0
            with open(corpus, "rb") as src, open(tmp_head, "wb") as dst:
                rem = n
                while rem > 0:
                    chunk = src.read(min(1024 * 1024, rem))
                    if not chunk:
                        break
                    dst.write(chunk)
                    rem -= len(chunk)
                    written += len(chunk)
            return written

        cycle = 0
        while not self.read_stop():
            cycle += 1
            n_written = snapshot_head()
            if n_written is None:
                emit("info", "corpus too small or missing — waiting")
                self.sleep_responsive(cycle_pause)
                continue
            emit("info", f"cycle {cycle} · head={n_written // 1_000_000}M")
            try:
                self.quiet(model.train,
                    str(tmp_head),
                    epochs=1,
                    save_every=0,
                    save_path=save_path,
                    start_byte=0,
                    report_every=1,
                    dream_every_batches=self.dream_every_batches,
                    dream_length=self.dream_length,
                    dream_temperature=self.dream_temperature,
                    dream_callback=model._dream_callback,
                )
            except KeyboardInterrupt:
                self.read_stop()
                break
            except Exception as e:
                emit("error", f"cycle error: {e}")
                self.sleep_responsive(cycle_pause)
                continue
            finally:
                try:
                    tmp_head.unlink()
                except FileNotFoundError:
                    pass

            if (self.autosave_interval_s > 0
                    and time.time() - self.last_save_t[0] >= self.autosave_interval_s):
                try:
                    self.quiet(model.save, save_path)
                    self.memory_hygiene(model, force=True)
                    self.last_save_t[0] = time.time()
                    emit("checkpoint_saved", save_path)
                except Exception as e:
                    emit("error", f"save failed: {e}")
            if not self.read_stop():
                emit("info", f"pausing {int(cycle_pause)}s before next cycle")
                self.sleep_responsive(cycle_pause)
        self.final_save(model, save_path)

    def sleep_responsive(self, seconds):
        slept = 0.0
        while slept < seconds and not self.read_stop():
            time.sleep(min(1.0, seconds - slept))
            slept += 1.0

    def final_save(self, model, save_path):
        self.read_stop()
        if not self.save_on_stop:
            emit("info", "stopped without saving")
            return
        try:
            self.quiet(model.save, save_path)
            self.memory_hygiene(model, force=True)
            emit("checkpoint_saved", save_path)
        except Exception as e:
            emit("error", f"save failed: {e}")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: soma_train_worker.py config.json stop.json")
    Worker(sys.argv[1], sys.argv[2]).run()


if __name__ == "__main__":
    main()

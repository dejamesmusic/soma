"""
soma_gui.py — minimal tkinter interface for living with soma.

three views:
  chat      talk to a checkpoint, optionally with online learning
  train     load or start fresh, configure, run; with optional 
            "loop" mode that keeps training on a rolling corpus
  stream    pick a stream from streams/ and run it; see its output

aesthetic: black background, sf mono, lowercase. rounded pill
widgets for inputs and buttons.

state persistence: form values are saved to gui_state.json beside the
launcher so a user can return and pick up where they left off.

requires only what already ships with python on macOS — tkinter is part
of the standard library. no extra pip packages beyond torch+numpy that
soma already requires.

launches via:  ./soma gui
"""

import gc
import math
import os
import sys
import json
import time
import queue
import signal
import threading
import subprocess
import shutil
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

from soma_voice import LocalSpeaker, OnDeviceRecognizer
from soma_profiles import (
    CAPACITY_NAMES,
    DEFAULT_CAPACITY,
    capacity_values,
    default_batch_size,
    estimated_mps_training_bytes,
    infer_capacity,
    mps_fit_status,
    parameter_count,
    recommended_mps_batch,
)

try:
    import certifi
    URL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    URL_CONTEXT = ssl.create_default_context()


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



# ──────────────────────────────────────────────────────────────────────
# theme
# ──────────────────────────────────────────────────────────────────────

BG = "#000302"           # main page background
SURFACE = "#07100a"      # deep phosphor-black instrument surface
SURFACE_HOVER = "#0e1c12"
SURFACE_ACTIVE = "#15301d"
FG = "#d9ffe1"           # warm white-green primary text
DIM = "#6f9477"          # quiet phosphor labels
ACCENT = "#63ff82"
AMBER = "#e6b45f"        # progress, caution, and secondary emphasis
SELECT_BG = "#143b20"
TAB_ACTIVE = "#0a1b0f"
SIDEBAR_BG = "#010603"
SPARK_LINE = "#70ff8b"
SPARK_GLOW = "#174f27"
SPARK_FILL = "#06160b"
GRID_LINE = "#0d2815"

MONO_PREFS = ("SF Mono", "SFMono-Regular", "Menlo", "Monaco",
              "Courier New", "monospace")

DEFAULT_CHECKPOINT_ART = (
    "███  ███  █▄█  ███\n"
    "█▄▄  █ █  █ █  █▄█\n"
    "▄▄█  ███  █ █  █ █"
)


def pick_mono(root):
    available = set(tkfont.families(root))
    for name in MONO_PREFS:
        if name in available:
            return name
    return "monospace"


# ──────────────────────────────────────────────────────────────────────
# bundle paths
# ──────────────────────────────────────────────────────────────────────

BUNDLE_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = Path(os.environ.get(
    "SOMA_HOME", "~/Library/Application Support/soma")).expanduser()
DATA_DIR = RUNTIME_DIR / "data"
STREAMS_DATA_DIR = RUNTIME_DIR / "data" / "streams"
STREAMS_DIR = RUNTIME_DIR / "streams"
CHECKPOINTS_DIR = RUNTIME_DIR / "checkpoints"
LOGS_DIR = RUNTIME_DIR / "logs"
STATE_DIR = RUNTIME_DIR / "state"
STATE_FILE = STATE_DIR / "gui_state.json"
RUNS_DIR = STATE_DIR / "runs"
ACTIVE_TRAIN_RUN_FILE = STATE_DIR / "active_train_run.json"
CHAT_LOG_FILE = LOGS_DIR / "chat_history.txt"
TRAIN_LOG_FILE = LOGS_DIR / "train_history.txt"
MAX_CHAT_LOG_BYTES = 1_500_000
MAX_TRAIN_LOG_BYTES = 600_000
PUBLIC_DIR = Path.home() / "Documents" / "soma"
PUBLIC_DATA_LINK = PUBLIC_DIR / "data"
LOGOS_API = "https://logossoma.com"
DEFAULT_AUTO_MODE = "io2"
GUI_RUNTIME_VERSION = "14.0.0-dev.2"
ALL_DATA_OPTION = "(all data files)"
NO_VALIDATION_OPTION = "(no validation)"
VOCAB_SIZE = 1024
SCALE_KERNEL = 5


def default_decimation_for_mode(mode):
    return "1"


def _mps_recommended_memory():
    try:
        import torch
        if (
            torch.backends.mps.is_available()
            and hasattr(torch.mps, "recommended_max_memory")
        ):
            return int(torch.mps.recommended_max_memory())
    except Exception:
        pass
    return 0


def _aperture_shape(n_bands, capacity=DEFAULT_CAPACITY):
    n_bands = max(1, int(n_bands))
    values = capacity_values(capacity, n_bands)
    params = parameter_count(
        capacity, n_bands, vocab_size=VOCAB_SIZE, scale_kernel=SCALE_KERNEL
    )
    return [n_bands], values["scale_rank"], params


def ensure_dirs():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    STREAMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    STREAMS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if PUBLIC_DATA_LINK.is_symlink():
            if PUBLIC_DATA_LINK.resolve() != DATA_DIR.resolve():
                PUBLIC_DATA_LINK.unlink()
        if not PUBLIC_DATA_LINK.exists():
            PUBLIC_DATA_LINK.symlink_to(DATA_DIR, target_is_directory=True)
    except OSError:
        pass
    _seed_stream_scripts()


def _seed_stream_scripts():
    """Copy bundled stream scripts into the user-manageable streams folder.

    The app can always fall back to bundled streams, but first-run should
    also make those scripts visible in Application Support so users can
    inspect, replace, or add stream accessories without editing the app
    bundle. Existing user copies are left untouched.
    """
    bundled = BUNDLE_DIR / "streams"
    if not bundled.is_dir():
        return
    try:
        readme = STREAMS_DIR / "README.txt"
        if not readme.exists():
            readme.write_text(
                "soma stream scripts live here.\n\n"
                "drop compatible .py stream accessories into this folder, "
                "then restart soma. stream output corpora are written to "
                "data/streams/.\n",
                encoding="utf-8",
            )
        for src in bundled.glob("*.py"):
            if src.name.startswith("_"):
                continue
            dest = STREAMS_DIR / src.name
            if not dest.exists():
                dest.write_text(src.read_text(encoding="utf-8"),
                                encoding="utf-8")
    except OSError:
        pass


def _fmt_bytes(n):
    """compact binary source-byte size with an explicit byte unit."""
    n = float(n)
    for unit in ("b", "kb", "mb", "gb"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}tb"


def _fmt_tokens(n):
    """compact decimal token count; b means billion, never bytes."""
    n = float(n)
    for unit in ("", "k", "m", "b"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}t"


def _fmt_duration(seconds):
    """Compact duration formatter. 12 → '12s', 90 → '1.5m', 7200 → '2.0h'."""
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _parse_cycle_spec(text):
    parts = str(text).strip().split()
    if not parts or parts[0].lower() != "cycle":
        return None
    if len(parts) == 1:
        return 1
    try:
        return max(1, int(float(parts[1])))
    except ValueError:
        return 1


def _is_trainable_corpus(path):
    """Keep metadata and compressed/source artifacts out of corpus cycles."""
    if not path.is_file() or path.name.startswith("."):
        return False
    lowered = path.name.lower()
    if "_manifest" in lowered or "_readme" in lowered:
        return False
    return path.suffix.lower() in ("", ".txt")


def _checkpoint_meta_path(checkpoint_path):
    return Path(f"{Path(checkpoint_path)}.meta.json")


def _unused_checkpoint_path(path):
    """Return path, or a numbered sibling, without clobbering a lineage."""
    path = Path(path)
    if not path.exists():
        return path
    suffix = path.suffix or ".pt"
    stem = path.stem if path.suffix else path.name
    for number in range(2, 1_000_000):
        candidate = path.with_name(f"{stem}-{number}{suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"no unused checkpoint name beside {path}")


def _read_json_dict(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _pid_is_alive(pid):
    try:
        pid = int(pid)
        if pid < 1:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _pid_is_training_worker(pid, config_path=None):
    if not _pid_is_alive(pid):
        return False
    try:
        result = subprocess.run(
            ["/bin/ps", "-p", str(int(pid)), "-o", "command="],
            capture_output=True, text=True, timeout=2, check=False,
        )
        command = result.stdout.strip()
    except (OSError, subprocess.SubprocessError, TypeError, ValueError):
        return False
    if "soma_train_worker.py" not in command:
        return False
    return not config_path or str(config_path) in command


def _read_checkpoint_metadata(checkpoint_path):
    try:
        data = json.loads(
            _checkpoint_meta_path(checkpoint_path).read_text(
                encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _checkpoint_lattice_rate(checkpoint):
    training = checkpoint.get("training")
    training = training if isinstance(training, dict) else {}
    try:
        value = float(checkpoint.get(
            "lattice_rate", training.get("lattice_rate", 0.0)))
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def _checkpoint_profile(checkpoint_path, metadata=None):
    metadata = metadata if isinstance(metadata, dict) else {}
    raw = metadata.get("profile")
    raw = raw if isinstance(raw, dict) else {}
    path = Path(checkpoint_path)
    return {
        "name": str(raw.get("name") or path.stem),
        "tagline": str(
            raw.get("tagline") or "local continual language model"),
        "art": str(raw.get("art") or DEFAULT_CHECKPOINT_ART),
        "about": str(raw.get("about") or ""),
    }


def _write_checkpoint_profile(checkpoint_path, profile):
    """Atomically update only the human-owned sidecar profile."""
    checkpoint_path = Path(checkpoint_path)
    metadata = _read_checkpoint_metadata(checkpoint_path)
    metadata["sidecar_version"] = 2
    metadata["profile"] = {
        key: str(profile.get(key, ""))
        for key in ("name", "tagline", "art", "about")
    }
    meta_path = _checkpoint_meta_path(checkpoint_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{meta_path}.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    os.replace(temporary, meta_path)
    return meta_path


def _data_cycle_files():
    if not DATA_DIR.exists():
        return []
    files = []
    for path in sorted(DATA_DIR.iterdir(), key=lambda p: p.name.lower()):
        if _is_trainable_corpus(path):
            files.append(path)
    return files


# ──────────────────────────────────────────────────────────────────────
# state persistence
# ──────────────────────────────────────────────────────────────────────

DEFAULT_STATE = {
    "runtime_version": GUI_RUNTIME_VERSION,
    "train": {
        "checkpoint": "",
        "mode": "corpus",        # "corpus" or "stream"
        "corpus": "",            # used when mode == "corpus"
        "stream_name": "",       # used when mode == "stream"
        "save_path": "model.pt",
        "start_byte": "0",
        "head_mb": "50",
        "autosave_min": "30",
        "dream_every_batches": "64",
        "dream_length": "128",
        "dream_temperature": "1.0",
        "epochs": "1",
        "bands": "48",
        "capacity": DEFAULT_CAPACITY,
        "base": "1.618033988749895",
        "lr": "auto 0.0003",
        "decimation": "1",
        "grad_clip": "1.0",
        "batch": "512",
        "update_interval": "1",
        "advanced": False,
        "checkpoint_action": "continue",
        "validation_corpus": "",
        "validation_every_mb": "100",
        "validation_max_mb": "0.25",
    },
    "chat": {
        "checkpoint": "",
        "online": False,
        "voice": False,
        "speech_locale": "auto",
        "temperature": "0.8",
        "max_length": "200",
        "logos_mode": False,
        "pass_prompt": "",
    },
    "stream": {
        "stream_name": "",
    },
    "logOS": {
        "sort": "newest",
    },
    "resume_positions": {},
    "last_view": "models",
}


def load_state():
    if not STATE_FILE.exists():
        return {k: dict(v) if isinstance(v, dict) else v
                for k, v in DEFAULT_STATE.items()}
    try:
        with open(STATE_FILE) as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {k: dict(v) if isinstance(v, dict) else v
                for k, v in DEFAULT_STATE.items()}
    state = {k: dict(v) if isinstance(v, dict) else v
             for k, v in DEFAULT_STATE.items()}
    for k, v in saved.items():
        if k in state and isinstance(state[k], dict) and isinstance(v, dict):
            state[k].update(v)
        else:
            state[k] = v
    saved_version = saved.get("runtime_version")
    if saved_version != GUI_RUNTIME_VERSION:
        state["runtime_version"] = GUI_RUNTIME_VERSION
        state["train"].update({
            "checkpoint": "(new model)",
            "save_path": "model.pt",
            "start_byte": "0",
            "bands": "48",
            "capacity": DEFAULT_CAPACITY,
            "base": "1.618033988749895",
            "lr": "auto 0.0003",
            "decimation": "1",
            "grad_clip": "1.0",
            "batch": "512",
            "update_interval": "1",
            "advanced": False,
            "checkpoint_action": "continue",
            "validation_corpus": "",
            "validation_every_mb": "100",
            "validation_max_mb": "0.25",
            "autosave_min": "30",
            "epochs": "1",
            "dream_every_batches": "64",
            "dream_length": "128",
            "dream_temperature": "1.0",
        })
        state["chat"]["checkpoint"] = ""
    return state


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except OSError:
        pass


# ──────────────────────────────────────────────────────────────────────
# rounded widgets — drawn on a canvas backdrop
# ──────────────────────────────────────────────────────────────────────
#
# tk has no built-in border-radius. the trick: a Canvas draws a filled
# rounded rectangle in a chosen surface color, and the actual Entry or
# Label is placed inside it with that *same* surface color as its bg.
# the Entry/Label is rectangular but its rectangle stays inside the
# pill silhouette, so the corners that would otherwise be visible are
# already painted in the surface color and look perfectly round.
#
# this gives us full visual control with zero external dependencies.
# ──────────────────────────────────────────────────────────────────────


def _round_rect_points(x1, y1, x2, y2, r):
    """Polygon vertices for a rounded rectangle, suitable for
    Canvas.create_polygon(..., smooth=True). Smooth=True turns 
    the corner triplets into Bezier curves."""
    return [
        x1 + r, y1,
        x2 - r, y1,
        x2, y1,            # corner control
        x2, y1 + r,
        x2, y2 - r,
        x2, y2,            # corner control
        x2 - r, y2,
        x1 + r, y2,
        x1, y2,            # corner control
        x1, y2 - r,
        x1, y1 + r,
        x1, y1,            # corner control
    ]


class RoundedEntry(tk.Canvas):
    """A pill-shaped text entry. The actual Entry sits inside a
    rounded-rect surface drawn on this canvas."""

    def __init__(self, parent, width=200, height=32, value="",
                 font=None, fg=FG, surface=SURFACE):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.surface = surface
        self.radius = 4
        self._draw_surface()

        self.entry = tk.Entry(
            self, bg=surface, fg=fg, font=font,
            insertbackground=fg, bd=0, highlightthickness=0,
            relief="flat", selectbackground=SELECT_BG,
            selectforeground=fg)
        if value:
            self.entry.insert(0, value)
        # place the Entry inside the pill, leaving room for the curves
        pad = 10
        self.create_window(
            pad, height // 2,
            anchor="w",
            width=width - 2 * pad,
            window=self.entry,
        )

    def _draw_surface(self):
        self.delete("surface")
        pts = _round_rect_points(
            1, 1, self.width - 1, self.height - 1, self.radius - 1)
        self.create_polygon(
            *pts, fill=self.surface, outline="",
            smooth=True, tags="surface")

    def get(self):
        return self.entry.get()

    def set(self, value):
        was_ro = (self.entry.cget("state") == "readonly")
        if was_ro:
            self.entry.config(state="normal")
        self.entry.delete(0, "end")
        self.entry.insert(0, value)
        self.entry.xview_moveto(0)
        if was_ro:
            self.entry.config(state="readonly")

    def set_readonly(self, readonly):
        """Lock/unlock the entry. Locked entries display dimmer text 
        on a darker surface to signal they're informational only."""
        if readonly:
            self.entry.config(
                state="readonly",
                fg=DIM, readonlybackground=BG,
                disabledbackground=BG)
            self.surface = BG
        else:
            self.entry.config(state="normal", fg=FG)
            self.surface = SURFACE
        self._draw_surface()

    def bind_change(self, callback):
        """Call `callback(value)` when the entry loses focus or Enter pressed."""
        def commit(_event=None):
            callback(self.entry.get())
            self.entry.xview_moveto(0)
        self.entry.bind(
            "<FocusOut>", commit)
        self.entry.bind(
            "<Return>", commit)


def _bind_reliable_pointer(widget, command, enabled=lambda: True,
                           on_press=None, on_release=None):
    """Invoke on press because macOS can drop stationary canvas releases."""
    widget._soma_click_armed = False

    def settle_visual():
        if not widget.winfo_exists() or not widget._soma_click_armed:
            return
        widget._soma_click_armed = False
        pointer_inside = (
            widget.winfo_rootx() <= widget.winfo_pointerx()
            < widget.winfo_rootx() + widget.winfo_width()
            and widget.winfo_rooty() <= widget.winfo_pointery()
            < widget.winfo_rooty() + widget.winfo_height()
        )
        if on_release:
            on_release(None, pointer_inside)

    def press(event):
        if not enabled():
            return "break"
        widget._soma_click_armed = True
        if on_press:
            on_press(event)
        # Do not defer activation to ButtonRelease. Aqua/Tk can omit that
        # event when the pointer has not moved between press and release.
        command(event)
        if widget.winfo_exists():
            widget.after(120, settle_visual)
        return "break"

    def release(event):
        widget._soma_click_armed = False
        inside = (
            widget.winfo_rootx() <= event.x_root
            < widget.winfo_rootx() + widget.winfo_width()
            and widget.winfo_rooty() <= event.y_root
            < widget.winfo_rooty() + widget.winfo_height()
        )
        if on_release:
            on_release(event, inside)
        return "break"

    widget.bind("<ButtonPress-1>", press)
    widget.bind("<ButtonRelease-1>", release)


def _system_control_font(parent, requested=None, default_size=12):
    """Use the platform UI family while preserving the control size."""
    size = default_size
    if isinstance(requested, (tuple, list)) and len(requested) >= 2:
        try:
            size = int(requested[1])
        except (TypeError, ValueError):
            pass
    else:
        try:
            size = int(tkfont.Font(root=parent, font=requested).cget("size"))
        except (tk.TclError, TypeError, ValueError):
            pass
    try:
        family = tkfont.nametofont(
            "TkDefaultFont", root=parent).actual("family")
    except tk.TclError:
        family = "TkDefaultFont"
    return (family, size)


class RoundedButton(tk.Canvas):
    """A dark canvas button with reliable macOS press/release handling."""

    def __init__(self, parent, text, command,
                 width=120, height=32, font=None,
                 fg=FG, surface=SURFACE,
                 hover=SURFACE_HOVER, active=SURFACE_ACTIVE):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.command = command
        self._enabled = True
        self.fg = fg
        self.surface = surface
        self.hover = hover
        self.active = active
        self.radius = 4
        self._current_surface = surface
        self.text = text
        self.font = _system_control_font(parent, font)
        self._surface_id = None
        self._text_id = None
        self._redraw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Return>", self._on_key)
        self.bind("<space>", self._on_key)
        self.config(cursor="hand2", takefocus=True)
        _bind_reliable_pointer(
            self, lambda _event: self._fire(),
            enabled=lambda: self._enabled,
            on_press=self._on_pointer_press,
            on_release=self._on_pointer_release)

    def _redraw(self):
        if self._surface_id is None:
            pts = _round_rect_points(
                1, 1, self.width - 1, self.height - 1, self.radius - 1)
            self._surface_id = self.create_polygon(
                *pts, fill=self._current_surface,
                outline=GRID_LINE, smooth=True)
            self._text_id = self.create_text(
                self.width // 2, self.height // 2,
                text=self.text, fill=self.fg, font=self.font)
        else:
            self.itemconfigure(
                self._surface_id, fill=self._current_surface)
            self.itemconfigure(
                self._text_id, text=self.text,
                fill=self.fg if self._enabled else DIM)

    def _fire(self):
        if not self._enabled:
            return
        try:
            self.command()
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            self._root().report_callback_exception(
                exc_type, exc_value, exc_traceback)

    def set_text(self, text):
        self.text = text
        self._redraw()

    def set_enabled(self, enabled):
        self._enabled = enabled
        self._current_surface = self.surface
        self.config(cursor="hand2" if enabled else "")
        self._redraw()

    def _on_enter(self, event):
        if not self._enabled:
            return
        self._current_surface = self.hover
        self._redraw()

    def _on_leave(self, event):
        if not self._soma_click_armed:
            self._current_surface = self.surface
            self._redraw()

    def _on_pointer_press(self, _event):
        self.focus_set()
        self._current_surface = self.active
        self._redraw()

    def _on_pointer_release(self, _event, inside):
        self._current_surface = (
            self.hover if inside and self._enabled else self.surface)
        self._redraw()

    def _on_key(self, _event):
        self._fire()
        return "break"


class RoundedDropdown(tk.Canvas):
    """A dark canvas dropdown whose menu opens on a complete click."""

    def __init__(self, parent, options, value=None, on_select=None,
                 width=240, height=32, font=None):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.radius = 4
        self.font = _system_control_font(parent, font)
        self.options = list(options)
        self.value = value if value in self.options else (
            self.options[0] if self.options else "")
        self.on_select = on_select
        self._enabled = True
        self._current_surface = SURFACE
        self._redraw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.config(cursor="hand2")
        _bind_reliable_pointer(
            self, lambda _event: self._show_menu(),
            enabled=lambda: self._enabled,
            on_press=lambda _event: self._set_surface(SURFACE_ACTIVE),
            on_release=lambda _event, inside: self._set_surface(
                SURFACE_HOVER if inside and self._enabled else SURFACE))

    def _set_surface(self, surface):
        self._current_surface = surface
        self._redraw()

    def _redraw(self):
        self.delete("all")
        pts = _round_rect_points(
            1, 1, self.width - 1, self.height - 1, self.radius - 1)
        self.create_polygon(
            *pts, fill=self._current_surface, outline="", smooth=True)
        self.create_text(
            10, self.height // 2, anchor="w",
            text=self._ellipsize(self.value),
            fill=FG if self._enabled else DIM, font=self.font)
        cx = self.width - 14
        cy = self.height // 2
        self.create_polygon(
            cx - 4, cy - 2, cx + 4, cy - 2, cx, cy + 3,
            fill=DIM, outline="")

    def _ellipsize(self, text):
        text = str(text)
        try:
            measure_font = tkfont.Font(font=self.font)
            pad = 10
            available = max(20, self.width - pad - 30)
            if measure_font.measure(text) <= available:
                return text
            suffix = "..."
            while text and measure_font.measure(text + suffix) > available:
                text = text[:-1]
            return text + suffix if text else suffix
        except Exception:
            return text

    def _update_display(self):
        self._redraw()

    def _on_enter(self, event):
        if not self._enabled:
            return
        if not self._soma_click_armed:
            self._set_surface(SURFACE_HOVER)

    def _on_leave(self, event):
        if not self._soma_click_armed:
            self._set_surface(SURFACE)

    def _show_menu(self):
        if not self._enabled or not self.options:
            return
        menu = tk.Menu(
            self, tearoff=0,
            bg=SURFACE, fg=FG, font=self.font,
            activebackground=SURFACE_HOVER, activeforeground=FG,
            bd=0, relief="flat")
        for opt in self.options:
            menu.add_command(
                label=opt, command=lambda v=opt: self._pick(v))
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.height + 2
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _pick(self, value):
        self.value = value
        self._update_display()
        if self.on_select:
            self.on_select(value)

    def set_options(self, options, value=None):
        self.options = list(options)
        if value is not None and value in self.options:
            self.value = value
        elif self.value not in self.options:
            self.value = self.options[0] if self.options else ""
        self._update_display()

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self._current_surface = SURFACE
        self.config(cursor="hand2" if self._enabled else "")
        self._update_display()

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)
        if self.value not in self.options:
            self.options.append(self.value)
        self._update_display()


class RoundedCheckbox(tk.Canvas):
    """Dark canvas checkbox with release-inside behavior."""

    def __init__(self, parent, text, value=False,
                 on_change=None, width=200, height=24, font=None):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.value = bool(value)
        self.text = text
        self.font = _system_control_font(parent, font, default_size=11)
        self.on_change = on_change
        self._redraw()
        self.bind("<Return>", self._toggle)
        self.bind("<space>", self._toggle)
        self.config(cursor="hand2", takefocus=True)
        _bind_reliable_pointer(
            self, lambda _event: self._toggle(),
            on_press=lambda _event: self.focus_set())

    def _redraw(self):
        self.delete("all")
        size = 14
        x1 = 4
        y1 = (self.height - size) // 2
        x2 = x1 + size
        y2 = y1 + size
        pts = _round_rect_points(x1, y1, x2, y2, 4)
        self.create_polygon(
            *pts, fill=FG if self.value else SURFACE,
            outline=DIM, width=1, smooth=True)
        if self.value:
            self.create_line(
                x1 + 3, (y1 + y2) // 2,
                x1 + 6, y2 - 3, x2 - 3, y1 + 3,
                fill=BG, width=2, capstyle="round", joinstyle="round")
        self.create_text(
            x2 + 8, self.height // 2, anchor="w",
            text=self.text, fill=FG, font=self.font)

    def _update_display(self):
        self._redraw()

    def _toggle(self, _event=None):
        self.focus_set()
        self.value = not self.value
        self._update_display()
        if self.on_change:
            self.on_change(self.value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = bool(value)
        self._update_display()


class RoundedSegmented(tk.Canvas):
    """A dark canvas segmented control."""

    def __init__(self, parent, options, value=None, on_change=None,
                 width=240, height=30, font=None):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.radius = 4
        self.options = list(options)
        self.value = value if value in self.options else self.options[0]
        self.on_change = on_change
        self.font = _system_control_font(parent, font, default_size=11)
        self._segment_bounds = []
        self._redraw()
        self.config(cursor="hand2")
        _bind_reliable_pointer(self, self._on_click)

    def _redraw(self):
        self.delete("all")
        pts = _round_rect_points(
            1, 1, self.width - 1, self.height - 1, self.radius - 1)
        self.create_polygon(*pts, fill=SURFACE, outline="", smooth=True)
        if not self.options:
            return
        segment_width = self.width / len(self.options)
        self._segment_bounds = []
        for index, option in enumerate(self.options):
            x1 = index * segment_width
            x2 = (index + 1) * segment_width
            self._segment_bounds.append((x1, x2, option))
            if option == self.value:
                inset = 2
                active = _round_rect_points(
                    x1 + inset, inset, x2 - inset, self.height - inset,
                    self.radius - inset - 1)
                self.create_polygon(
                    *active, fill=SURFACE_ACTIVE, outline="", smooth=True)
            self.create_text(
                (x1 + x2) / 2, self.height / 2, text=option,
                fill=FG if option == self.value else DIM, font=self.font)

    def _update_display(self):
        self._redraw()

    def _on_click(self, event):
        local_x = event.x_root - self.winfo_rootx()
        for x1, x2, option in self._segment_bounds:
            if x1 <= local_x <= x2:
                if option != self.value:
                    self.value = option
                    self._redraw()
                    if self.on_change:
                        self.on_change(option)
                return

    def get(self):
        return self.value

    def set(self, value):
        if value in self.options:
            self.value = value
            self._update_display()


# ──────────────────────────────────────────────────────────────────────
# subprocess wrapper
# ──────────────────────────────────────────────────────────────────────

class ManagedProcess:
    def __init__(self, name):
        self.name = name
        self.proc = None
        self.queue = queue.Queue(maxsize=10000)
        self._reader = None

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, args, cwd=None):
        if self.is_running():
            return False
        self._reap_finished()
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "--run-script"] + list(args)
        else:
            cmd = [sys.executable] + list(args)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["SOMA_HOME"] = str(RUNTIME_DIR)
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd, bufsize=1, text=True, errors='replace', env=env)
        self._reader = threading.Thread(
            target=self._read_loop, daemon=True)
        self._reader.start()
        return True

    def stop(self, wait=False):
        if not self.is_running():
            return
        try:
            self.proc.send_signal(signal.SIGINT)
        except (OSError, ProcessLookupError):
            return
        if wait:
            self._finish_stop()
        else:
            threading.Thread(
                target=self._finish_stop, daemon=True).start()

    def _finish_stop(self):
        """escalate a stop without occupying tkinter's event loop."""
        proc = self.proc
        if proc is None:
            return
        try:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except (OSError, ProcessLookupError):
            pass
        finally:
            self._reap_finished(proc)

    def _reap_finished(self, proc=None):
        """close pipes and retire the reader for a completed child."""
        proc = proc or self.proc
        if proc is None or proc.poll() is None:
            return
        for stream in (proc.stdout, proc.stdin, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=0.2)
        if self.proc is proc:
            self.proc = None
            self._reader = None

    def _read_loop(self):
        try:
            for line in self.proc.stdout:
                try:
                    self.queue.put_nowait(line.rstrip('\n'))
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(line.rstrip('\n'))
                    except queue.Empty:
                        pass
        except Exception:
            pass

    def drain(self, max_lines=200):
        out = []
        while len(out) < max_lines:
            try:
                out.append(self.queue.get_nowait())
            except queue.Empty:
                break
        self._reap_finished()
        return out


# ──────────────────────────────────────────────────────────────────────
# chat worker
# ──────────────────────────────────────────────────────────────────────

class ChatWorker:
    def __init__(self):
        self.model = None
        self.online = False
        self.temperature = 0.8
        self.max_length = 200
        self.tasks = queue.Queue()
        self.outbox = queue.Queue(maxsize=10000)
        self._thread = None
        self._running = False
        self._stop_generation = threading.Event()
        self.dirty = False

    def load(self, ckpt_path, online=False,
             temperature=0.8, max_length=200):
        import soma_v14 as soma_runtime
        self.release_model()
        model = soma_runtime.SOMA.from_checkpoint(
            ckpt_path, device="auto")
        self.model = model
        self.dirty = False
        self.online = online
        self.temperature = temperature
        self.max_length = max_length
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self.model

    def submit(self, text):
        self._stop_generation.clear()
        self.tasks.put(text)

    def stop_generation(self):
        """Request a cooperative stop after the current decoded fragment."""
        self._stop_generation.set()

    def _run(self):
        while self._running:
            try:
                text = self.tasks.get(timeout=0.5)
            except queue.Empty:
                continue
            if text is None:
                break
            try:
                self.outbox.put(
                    ('status', 'learning prompt...'
                     if self.online else 'reading prompt...'))
                self.model.ingest_prompt(text + ' ', online=self.online)
                self.outbox.put(('status', 'generating...'))
                stream = self.model.generate(
                    length=self.max_length,
                    temperature=self.temperature,
                    prelude_source="chat")
                try:
                    for ch in stream:
                        if self._stop_generation.is_set():
                            self.outbox.put(('status', 'generation stopped'))
                            break
                        self.outbox.put(('char', ch))
                finally:
                    stream.close()
                # Prompt ingestion and generation both advance the trace bank;
                # online mode may also update weights. Treat a completed turn
                # as dirty so unload can offer save/discard/cancel.
                self.dirty = True
                self._stop_generation.clear()
                self.outbox.put(('done',))
            except Exception as e:
                self._stop_generation.clear()
                self.outbox.put(('error', f"{type(e).__name__}: {e}"))

    def drain(self, max_items=500):
        out = []
        while len(out) < max_items:
            try:
                out.append(self.outbox.get_nowait())
            except queue.Empty:
                break
        return out

    def shutdown(self):
        self._running = False
        self.release_model()

    def release_model(self):
        model = self.model
        if model is None:
            return
        device = getattr(getattr(model, "device", None), "type", None)
        self.model = None
        del model
        release_torch_memory(device)


# ──────────────────────────────────────────────────────────────────────
# train worker — supports both single-pass training and loop mode
# ──────────────────────────────────────────────────────────────────────

class TrainWorker:
    """subprocess-backed train worker.

    training is deliberately isolated from the gui process. when a run stops,
    the child process exits and releases torch/mps state at the os boundary.
    """

    def __init__(self):
        self.proc = ManagedProcess("train")
        self.config_path = None
        self.stop_path = None
        self.manifest_path = None
        self.event_path = None
        self.metrics_path = None
        self.run_id = ""
        self.pid = None
        self.attached = False
        self.recovery_notice = ""
        self._event_offset = 0
        self._manifest_heartbeat_seen = 0.0
        self._recovery_events = deque()
        self._done_seen = False
        self._attach_active_run()

    def is_running(self):
        if self.proc.is_running():
            return True
        return bool(
            self.attached
            and _pid_is_training_worker(self.pid, self.config_path)
        )

    def _attach_active_run(self):
        pointer = _read_json_dict(ACTIVE_TRAIN_RUN_FILE)
        manifest_path = pointer.get("manifest_path")
        if not manifest_path:
            return
        manifest = _read_json_dict(manifest_path)
        if not manifest:
            self.recovery_notice = "the previous run record is unreadable"
            return
        self.run_id = str(manifest.get("run_id") or "")
        self.manifest_path = Path(manifest_path)
        self.event_path = Path(pointer.get("event_path", ""))
        metrics_value = pointer.get("metrics_path")
        self.metrics_path = Path(metrics_value) if metrics_value else None
        self.config_path = Path(pointer.get("config_path", ""))
        self.stop_path = Path(pointer.get("stop_path", ""))
        self.pid = manifest.get("pid") or pointer.get("pid")
        status = str(manifest.get("status") or "unknown")
        if status in (
                "starting", "running", "saving", "validating", "stopping",
                "restarting"):
            if _pid_is_training_worker(self.pid, self.config_path):
                self.attached = True
                self._event_offset = (
                    self.event_path.stat().st_size
                    if self.event_path.exists() else 0
                )
                self._manifest_heartbeat_seen = float(
                    manifest.get("last_report_at", 0.0) or 0.0)
                self._recovery_events.append(("start",))
                summary = manifest.get("summary")
                report = manifest.get("last_report")
                if isinstance(summary, dict):
                    self._recovery_events.append(("summary", summary))
                if isinstance(report, dict):
                    self._recovery_events.append(("report", report))
                if manifest.get("last_warning"):
                    self._recovery_events.append((
                        "warning", str(manifest["last_warning"])
                    ))
                self._recovery_events.append((
                    "info", f"reattached to run {self.run_id[:12]}"
                ))
                return
            manifest["status"] = "interrupted"
            manifest["ended_at"] = time.time()
            try:
                _write_json_atomic(self.manifest_path, manifest)
            except OSError:
                pass
            status = "interrupted"
        if status in ("failed", "interrupted", "stopped_without_save"):
            self.recovery_notice = (
                f"previous training run {status}; the last published "
                "checkpoint remains safe"
            )

    def start(self, config):
        if self.is_running():
            return False
        token = f"{int(time.time() * 1000)}-{os.urandom(4).hex()}"
        self.run_id = token
        self.config_path = RUNS_DIR / f"{token}.config.json"
        self.stop_path = RUNS_DIR / f"{token}.stop.json"
        self.manifest_path = RUNS_DIR / f"{token}.manifest.json"
        self.event_path = RUNS_DIR / f"{token}.events.jsonl"
        self.metrics_path = RUNS_DIR / f"{token}.metrics.jsonl"
        config = dict(config)
        config.update({
            "_run_id": token,
            "_run_manifest_path": str(self.manifest_path),
            "_run_event_path": str(self.event_path),
            "_run_metrics_path": str(self.metrics_path),
        })
        started_at = time.time()
        manifest = {
            "version": 1,
            "run_id": token,
            "status": "starting",
            "pid": None,
            "started_at": started_at,
            "heartbeat_at": started_at,
            "checkpoint_path": config.get("save_path", ""),
            "source_path": config.get("corpus", ""),
            "mode": "continuous" if config.get("loop") else "corpus",
            "metrics_path": str(self.metrics_path),
        }
        try:
            if self.stop_path.exists():
                self.stop_path.unlink()
            self.event_path.write_text("", encoding="utf-8")
            self.metrics_path.write_text("", encoding="utf-8")
            self.config_path.write_text(
                json.dumps(config), encoding="utf-8")
            _write_json_atomic(self.manifest_path, manifest)
            _write_json_atomic(ACTIVE_TRAIN_RUN_FILE, {
                "version": 1,
                "run_id": token,
                "manifest_path": str(self.manifest_path),
                "event_path": str(self.event_path),
                "metrics_path": str(self.metrics_path),
                "config_path": str(self.config_path),
                "stop_path": str(self.stop_path),
                "pid": None,
            })
        except OSError as e:
            self.proc.queue.put(f'["error", "worker config failed: {e}"]')
            return False
        self._done_seen = False
        self.attached = False
        self._event_offset = 0
        self._manifest_heartbeat_seen = 0.0
        started = self.proc.start([
            str(BUNDLE_DIR / "soma_train_worker.py"),
            str(self.config_path),
            str(self.stop_path),
        ], cwd=str(BUNDLE_DIR))
        if not started:
            return False
        self.pid = self.proc.proc.pid
        manifest["pid"] = self.pid
        pointer = _read_json_dict(ACTIVE_TRAIN_RUN_FILE)
        pointer["pid"] = self.pid
        try:
            _write_json_atomic(self.manifest_path, manifest)
            _write_json_atomic(ACTIVE_TRAIN_RUN_FILE, pointer)
        except OSError:
            pass
        return True

    def stop(self, save=True):
        try:
            if self.stop_path:
                self.stop_path.write_text(
                    json.dumps({"save": bool(save)}), encoding="utf-8")
        except OSError:
            pass

    def drain(self, max_items=500):
        out = []
        while self._recovery_events and len(out) < max_items:
            out.append(self._recovery_events.popleft())
        lines = self.proc.drain(max_items - len(out))
        if self.attached and len(out) < max_items:
            lines.extend(self._drain_event_journal(max_items - len(out)))
        for line in lines:
            try:
                event = json.loads(line)
                if isinstance(event, list) and event:
                    if event[0] == "done":
                        self._done_seen = True
                    out.append(tuple(event))
                else:
                    out.append(("info", str(line)))
            except (json.JSONDecodeError, TypeError):
                if line:
                    out.append(("info", line))
        if self.attached and len(out) < max_items:
            manifest = self.manifest()
            heartbeat = float(manifest.get("last_report_at", 0.0) or 0.0)
            report = manifest.get("last_report")
            if (heartbeat > self._manifest_heartbeat_seen
                    and isinstance(report, dict)):
                self._manifest_heartbeat_seen = heartbeat
                out.append(("report", report))
        if (not self.proc.is_running()
                and not (
                    self.attached
                    and _pid_is_training_worker(self.pid, self.config_path)
                )
                and not self._done_seen
                and self.config_path is not None):
            manifest = self.manifest()
            if manifest.get("status") in (
                    "starting", "running", "saving", "validating",
                    "stopping", "restarting"):
                manifest["status"] = "interrupted"
                manifest["ended_at"] = time.time()
                try:
                    _write_json_atomic(self.manifest_path, manifest)
                except OSError:
                    pass
                out.append((
                    "error",
                    "training worker stopped unexpectedly; resume from the "
                    "last safe checkpoint",
                ))
            self._done_seen = True
            out.append(("done",))
        return out

    def _drain_event_journal(self, max_lines):
        if not self.event_path or not self.event_path.exists():
            return []
        lines = []
        try:
            with self.event_path.open("r", encoding="utf-8") as handle:
                handle.seek(self._event_offset)
                while len(lines) < max_lines:
                    line = handle.readline()
                    if not line:
                        break
                    lines.append(line.rstrip("\n"))
                self._event_offset = handle.tell()
        except OSError:
            return []
        return lines

    def manifest(self):
        return _read_json_dict(self.manifest_path) if self.manifest_path else {}


# ──────────────────────────────────────────────────────────────────────
# sparkline
# ──────────────────────────────────────────────────────────────────────

class Sparkline(tk.Canvas):
    def __init__(self, parent, width=400, height=80, capacity=300):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.values = deque(maxlen=capacity)
        self.recent_values = deque(maxlen=capacity)
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event):
        self.width = max(1, event.width)
        self.height = max(1, event.height)
        self._redraw()

    def push(self, v, recent=None):
        self.values.append(float(v))
        self.recent_values.append(
            float(v if recent is None else recent))
        self._redraw()

    def reset(self):
        self.values.clear()
        self.recent_values.clear()
        self.delete("all")

    def _redraw(self):
        self.delete("all")
        for fraction in (0.25, 0.5, 0.75):
            y = int(self.height * fraction)
            self.create_line(0, y, self.width, y, fill=GRID_LINE, width=1)
        if len(self.values) < 2:
            return
        all_values = list(self.values) + list(self.recent_values)
        vmin = min(all_values)
        vmax = max(all_values)
        span = max(1e-9, vmax - vmin)
        n = len(self.values)
        pad = 2
        w = self.width - 2 * pad
        h = self.height - 2 * pad

        pts = []
        for i, v in enumerate(self.values):
            x = pad + (i / (n - 1)) * w
            y = pad + (1 - (v - vmin) / span) * h
            pts.extend([x, y])

        fill_pts = list(pts) + [
            pad + w, self.height - pad,
            pad, self.height - pad,
        ]
        self.create_polygon(*fill_pts, fill=SPARK_FILL, outline="")
        self.create_line(*pts, fill=SPARK_GLOW, width=5, smooth=False)
        self.create_line(*pts, fill=SPARK_LINE, width=1.5, smooth=False)

        recent_pts = []
        for i, value in enumerate(self.recent_values):
            x = pad + (i / max(1, len(self.recent_values) - 1)) * w
            y = pad + (1 - (value - vmin) / span) * h
            recent_pts.extend([x, y])
        if len(recent_pts) >= 4:
            self.create_line(
                *recent_pts, fill=AMBER, width=1.2, smooth=False)


class ResponsiveTelemetry(tk.Frame):
    """A compact instrument grid that reflows rather than truncating."""

    def __init__(self, parent, fields, mono, min_cell=132,
                 value_font=None, value_fg=FG):
        super().__init__(parent, bg=BG)
        self.fields = list(fields)
        self.min_cell = int(min_cell)
        self._columns = 0
        self.labels = {}
        self.values = {}
        self._cell_by_key = {}
        for key, label_text in self.fields:
            cell = tk.Frame(self, bg=BG)
            label = tk.Label(
                cell, text=label_text, font=(mono, 9), bg=BG, fg=DIM,
                anchor="w")
            value = tk.Label(
                cell, text="—", font=value_font or (mono, 12),
                bg=BG, fg=value_fg, anchor="w")
            label.pack(anchor="w")
            value.pack(anchor="w")
            self.labels[key] = label
            self.values[key] = value
            cell._telemetry_key = key
            self._cell_by_key[key] = cell
        self._cells = [self.values[key].master for key, _ in self.fields]
        self._visible_cells = list(self._cells)
        self.bind("<Configure>", self._on_resize)
        self.after_idle(self._initial_layout)

    def _initial_layout(self):
        self._reflow(max(self.winfo_width(), self.min_cell))

    def _on_resize(self, event):
        self._reflow(event.width)

    def _reflow(self, width):
        columns = max(
            1, min(len(self._visible_cells), width // self.min_cell))
        if columns == self._columns:
            return
        self._columns = columns
        for cell in self._cells:
            cell.grid_forget()
        for column in range(len(self._cells)):
            self.grid_columnconfigure(column, weight=0, minsize=0)
        for i, cell in enumerate(self._visible_cells):
            row, column = divmod(i, columns)
            cell.grid(row=row, column=column, sticky="ew",
                      padx=(0, 12), pady=(0, 5))
        for column in range(columns):
            self.grid_columnconfigure(
                column, weight=1, minsize=self.min_cell,
                uniform="telemetry")

    def set_visible(self, keys=None):
        visible = set(keys) if keys is not None else set(self._cell_by_key)
        self._visible_cells = [
            self._cell_by_key[key]
            for key, _label in self.fields
            if key in visible
        ]
        for cell in self._cells:
            cell.grid_forget()
        self._columns = 0
        self._reflow(max(self.winfo_width(), self.min_cell))

    def set_value(self, key, value):
        if key in self.values:
            self.values[key].config(text=value)

    def set_label(self, key, value):
        if key in self.labels:
            self.labels[key].config(text=value)

    def reset(self):
        for value in self.values.values():
            value.config(text="—")


class LayerPlasticity(tk.Canvas):
    """Live view of how plasticity is distributed through depth."""

    def __init__(self, parent, mono, height=50):
        super().__init__(parent, height=height, bg=BG,
                         highlightthickness=0, bd=0)
        self.mono = mono
        self.height = height
        self.maturity = 0.0
        self.multipliers = [1.0]
        self.bind("<Configure>", lambda _event: self._redraw())

    def set(self, maturity, multipliers):
        self.maturity = max(0.0, min(1.0, float(maturity)))
        values = [max(0.0, min(1.0, float(v))) for v in multipliers]
        self.multipliers = values or [1.0]
        self._redraw()

    def reset(self):
        self.set(0.0, [1.0])

    def _redraw(self):
        self.delete("all")
        width = max(1, self.winfo_width())
        title = f"depth saturation  {100 * self.maturity:.0f}%"
        self.create_text(0, 7, text=title, fill=DIM,
                         font=(self.mono, 9), anchor="nw")
        n = len(self.multipliers)
        top, bottom = 23, self.height - 3
        gap = 5
        cell_width = max(8, (width - gap * (n - 1)) / n)
        for i, value in enumerate(self.multipliers):
            x0 = i * (cell_width + gap)
            x1 = min(width, x0 + cell_width)
            self.create_rectangle(x0, top, x1, bottom,
                                  fill=SURFACE, outline=GRID_LINE)
            fill_width = (x1 - x0) * value
            if fill_width > 0:
                self.create_rectangle(x0, top, x0 + fill_width, bottom,
                                      fill=SPARK_GLOW, outline="")
                self.create_line(x0 + fill_width, top,
                                 x0 + fill_width, bottom,
                                 fill=ACCENT, width=1)


class SpectrumPlot(tk.Canvas):
    """Input and residual spectra over the trace-bank bands."""

    def __init__(self, parent, mono, height=96):
        super().__init__(parent, height=height, bg=BG,
                         highlightthickness=0, bd=0)
        self.mono = mono
        self.height = height
        self.input_values = []
        self.weight_values = []
        self.error_values = []
        self.target_band = 0.0
        self.input_coherence = 0.0
        self.output_coherence = 0.0
        self.error_energy = 0.0
        self.bind("<Configure>", lambda _event: self._redraw())

    def set(self, input_values, weight_values, error_values, target_band=0.0,
            input_coherence=0.0, output_coherence=0.0,
            error_energy=0.0):
        self.input_values = [max(0.0, float(v)) for v in input_values]
        self.weight_values = [max(0.0, float(v)) for v in weight_values]
        self.error_values = [max(0.0, float(v)) for v in error_values]
        self.target_band = float(target_band)
        self.input_coherence = float(input_coherence)
        self.output_coherence = float(output_coherence)
        self.error_energy = float(error_energy)
        self._redraw()

    def reset(self):
        self.input_values = []
        self.weight_values = []
        self.error_values = []
        self.target_band = 0.0
        self.input_coherence = 0.0
        self.output_coherence = 0.0
        self.error_energy = 0.0
        self._redraw()

    def _redraw(self):
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        title = (
            "trace activity  readout allocation  "
            f"controller residual {self.error_energy:.2f}")
        self.create_text(
            0, 4, text=title, fill=DIM,
            font=(self.mono, 9), anchor="nw")
        values = self.input_values or self.weight_values or self.error_values
        if not values:
            self.create_line(0, height - 2, width, height - 2,
                             fill=GRID_LINE)
            return
        n = max(
            len(self.input_values), len(self.weight_values),
            len(self.error_values))
        top, bottom = 22, height - 10
        plot_height = max(1, bottom - top)
        gap = 2
        bar_width = max(1, (width - gap * max(0, n - 1)) / max(1, n))
        maximum = max(
            [0.1] + self.input_values + self.weight_values
            + self.error_values)
        for i in range(n):
            x0 = i * (bar_width + gap)
            x1 = min(width, x0 + bar_width)
            input_value = self.input_values[i] if i < len(
                self.input_values) else 0.0
            error_value = self.error_values[i] if i < len(
                self.error_values) else 0.0
            weight_value = self.weight_values[i] if i < len(
                self.weight_values) else 0.0
            input_y = bottom - plot_height * input_value / maximum
            error_y = bottom - plot_height * error_value / maximum
            weight_y = bottom - plot_height * weight_value / maximum
            self.create_rectangle(
                x0, input_y, x1, bottom, fill=SPARK_GLOW, outline="")
            center = (x0 + x1) / 2
            self.create_line(
                x0, weight_y, x1, weight_y,
                fill=AMBER, width=2)
            self.create_line(
                center, error_y, center, bottom,
                fill=ACCENT, width=max(1, int(bar_width * 0.28)))
        if n > 1:
            marker_x = max(0, min(width, self.target_band / (n - 1) * width))
            self.create_line(
                marker_x, top, marker_x, bottom,
                fill=AMBER, width=1, dash=(2, 3))
        self.create_text(0, height - 1, text="0", fill=DIM,
                         font=(self.mono, 8), anchor="sw")
        self.create_text(width, height - 1, text=str(n - 1), fill=DIM,
                         font=(self.mono, 8), anchor="se")


# ──────────────────────────────────────────────────────────────────────
# main app
# ──────────────────────────────────────────────────────────────────────

class App:
    def __init__(self):
        ensure_dirs()
        self.state = load_state()

        self.root = tk.Tk()
        self.root.title("soma")
        self.root.geometry("1140x780")
        self.root.minsize(900, 48)
        self.root.configure(bg=BG)
        self.root.report_callback_exception = self._report_callback_exception
        # macos may briefly report a one-pixel window before applying the
        # requested geometry. That must not activate compact monitor mode.
        self._shell_layout_ready = False

        self.mono = pick_mono(self.root)
        self.font_main = (self.mono, 12)
        self.font_dim = (self.mono, 11)
        self.font_title = (self.mono, 13)
        self.font_huge = (self.mono, 28)
        self.font_med = (self.mono, 14)
        self.font_micro = (self.mono, 9)
        self.font_button = ("TkDefaultFont", 12)
        self.font_button_sm = ("TkDefaultFont", 11)
        self.font_page_title = ("TkDefaultFont", 17)

        self.stream_proc = ManagedProcess("stream")
        self.logos_bridge = ManagedProcess("logos_bridge")
        self.chat = ChatWorker()
        self.voice = LocalSpeaker()
        self.speech = OnDeviceRecognizer()
        self.train_worker = TrainWorker()
        self._chat_pending = False
        self._chat_readonly_snapshot = False
        self._logos_mode = False
        self._logos_connected = False
        self.chat_response_buf = []
        self._speech_prefix = ""
        self._speech_received_final = False
        self.logos_response_buf = []
        self.logos_checkpoints = []
        self.logos_selected = None
        self.logos_downloading = False
        self._closing = False
        self._train_resume_cursor_enabled = False
        self._train_start_byte_override = False
        self._active_train_resume_key = ""
        self._train_last_absolute_pos = 0
        self._train_last_absolute_total = 0
        self._train_started_new_model = False
        self._train_last_saved_checkpoint = ""
        self._train_last_report_at = 0.0
        self._train_last_save_at = 0.0
        self._train_save_path = ""
        self._train_report_pos = 0
        self._train_report_total = 0
        self._train_report_bps = 0.0
        self._train_continuous = False
        self._train_saving = False
        self._train_validating = False
        self._train_last_validation_at = 0.0
        self._train_validation_text = "off"
        self._train_safety_refresh_at = 0.0
        # train ui state: 'idle' | 'configured' | 'training' | 'stopped'
        self._train_state = 'idle'

        # discover streams once at startup
        try:
            sys.path.insert(0, str(BUNDLE_DIR))
            import streams_registry
            self.streams = streams_registry.list_streams(
                extra_dirs=[STREAMS_DIR])
        except Exception:
            self.streams = []

        self._build_layout()
        self._build_models()
        self._build_chat()
        self._build_train()
        self._build_stream()
        self._build_logos()

        # if a checkpoint is pre-selected from saved state, populate the
        # train form fields from it (the dropdown's on_select callback
        # only fires on click, not on initial value assignment)
        initial_ckpt = self.train_ckpt_dd.value
        if initial_ckpt:
            self._on_train_ckpt_change(initial_ckpt)

        if self.train_worker.is_running():
            manifest = self.train_worker.manifest()
            self._train_last_report_at = float(
                manifest.get("last_report_at", 0.0) or 0.0)
            self._train_last_save_at = float(
                manifest.get("last_save_at", 0.0) or 0.0)
            self._train_save_path = str(
                manifest.get("checkpoint_path", "") or "")
            report = manifest.get("last_report") or {}
            self._train_report_pos = int(report.get("pos", 0) or 0)
            self._train_report_total = int(report.get("total", 0) or 0)
            self._train_report_bps = float(report.get("bps", 0.0) or 0.0)
            self._train_continuous = manifest.get("mode") == "continuous"
            self._train_saving = manifest.get("status") == "saving"
            self._train_validating = manifest.get("status") == "validating"
            validation = manifest.get("last_validation") or {}
            self._train_last_validation_at = float(
                manifest.get("last_validation_at", 0.0) or 0.0)
            if validation:
                self._train_validation_text = (
                    f"{float(validation.get('bpb', 0.0)):.2f} est. bpb"
                )
            elif manifest.get("validation_path"):
                self._train_validation_text = "waiting"
            self._set_train_state("training")
            self._set_global_status("training · reattached", active=True)
            self.train_live_status.config(text="reattaching to active run...")
            self._refresh_train_run_safety(force=True)
        elif self.train_worker.recovery_notice:
            self.train_setup_status.config(
                text=self.train_worker.recovery_notice)

        # validate saved view — earlier versions had different tabs
        last = self.state.get("last_view", "models")
        if last not in self.frames:
            last = "chat"
        self._show(last)
        self.root.after(100, self._poll)
        self.root.bind("<Configure>", self._reflow_shell)
        self.root.after_idle(self._finish_initial_layout)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── layout ──

    def _build_layout(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=136)
        self.sidebar.grid(row=0, column=0, rowspan=3, sticky="ns")
        self.sidebar.grid_propagate(False)

        tk.Label(self.sidebar, text="▓▒ soma ▒▓",
                 font=self.font_title, bg=SIDEBAR_BG, fg=ACCENT,
                 anchor="w", padx=18, pady=20).pack(fill="x")

        self.tabs = {}
        nav = (("models", "models"), ("chat", "chat"),
               ("train", "train"), ("stream", "stream"),
               ("logOS", "logOS"))
        for name, label in nav:
            btn = tk.Label(
                self.sidebar, text=f"  {label}",
                font=self.font_button, bg=SIDEBAR_BG, fg=DIM,
                anchor="w", padx=18, pady=6, cursor="hand2",
                bd=0, highlightthickness=0)
            btn.pack(fill="x")
            _bind_reliable_pointer(
                btn, lambda _event, n=name: self._show(n))
            self.tabs[name] = btn

        # quit button anchored at the bottom of the sidebar
        quit_wrap = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        quit_wrap.pack(side="bottom", fill="x", padx=18, pady=18)
        self.quit_btn = RoundedButton(
            quit_wrap, "quit", self._quit_app,
            width=104, height=30, font=self.font_dim)
        self.quit_btn.pack(anchor="w")

        self.header = tk.Frame(self.root, bg=BG, height=64)
        self.header.grid(row=0, column=1, sticky="ew", padx=20)
        self.header.grid_propagate(False)
        self.header.grid_columnconfigure(0, weight=1)
        self.header_title = tk.Label(
            self.header, text="models", font=self.font_page_title,
            bg=BG, fg=FG, anchor="w")
        self.header_title.grid(row=0, column=0, sticky="sw", pady=(10, 0))
        self.header_subtitle = tk.Label(
            self.header, text="local checkpoint library",
            font=self.font_dim, bg=BG, fg=DIM, anchor="w")
        self.header_subtitle.grid(row=1, column=0, sticky="nw")
        self.header_context = tk.Label(
            self.header, text="no checkpoint active", font=self.font_dim,
            bg=BG, fg=DIM, anchor="e")
        self.header_context.grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(20, 0))

        self.main = tk.Frame(self.root, bg=BG)
        self.main.grid(row=1, column=1, sticky="nsew")
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for name in ("models", "chat", "train", "stream", "logOS"):
            f = tk.Frame(self.main, bg=BG)
            f.grid(row=0, column=0, sticky="nsew")
            self.frames[name] = f

        self.statusbar = tk.Frame(self.root, bg=SIDEBAR_BG, height=28)
        self.statusbar.grid(row=2, column=1, sticky="ew")
        self.statusbar.grid_propagate(False)
        self.statusbar.grid_rowconfigure(0, weight=1)
        self.statusbar.grid_columnconfigure(1, weight=1)
        self.status_light = tk.Label(
            self.statusbar, text="●", font=self.font_micro,
            bg=SIDEBAR_BG, fg=DIM)
        self.status_light.grid(row=0, column=0, padx=(18, 7), sticky="w")
        self.global_status = tk.Label(
            self.statusbar, text="idle", font=self.font_micro,
            bg=SIDEBAR_BG, fg=DIM, anchor="w")
        self.global_status.grid(row=0, column=1, sticky="ew")
        self.system_status = tk.Label(
            self.statusbar, text="local · continuous", font=self.font_micro,
            bg=SIDEBAR_BG, fg=DIM, anchor="e")
        self.system_status.grid(row=0, column=2, padx=(8, 18), sticky="e")

    def _reflow_shell(self, event):
        """Collapse to a single persistent monitor strip when very short."""
        if event.widget is not self.root:
            return
        if not self._shell_layout_ready:
            return
        self._apply_shell_layout(event.height)

    def _finish_initial_layout(self):
        """Render once native macos geometry has settled."""
        self.root.update_idletasks()
        self._shell_layout_ready = True
        self._apply_shell_layout(max(150, self.root.winfo_height()), force=True)
        self.root.after(
            80, lambda: self._apply_shell_layout(
                max(150, self.root.winfo_height()), force=True))

    def _apply_shell_layout(self, height, force=False):
        monitor = int(height) < 150
        if (not force and
                monitor == getattr(self, "_monitor_strip", False)):
            return
        self._monitor_strip = monitor
        if monitor:
            self.sidebar.grid_remove()
            self.header.grid_remove()
            self.main.grid_remove()
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_rowconfigure(1, weight=0)
            self.statusbar.grid(
                row=0, column=0, columnspan=2, sticky="nsew")
            self.system_status.grid_remove()
        else:
            self.root.grid_rowconfigure(0, weight=0)
            self.root.grid_rowconfigure(1, weight=1)
            self.sidebar.grid(
                row=0, column=0, rowspan=3, sticky="ns")
            self.header.grid(row=0, column=1, sticky="ew", padx=20)
            self.main.grid(row=1, column=1, sticky="nsew")
            self.statusbar.grid(
                row=2, column=1, columnspan=1, sticky="ew")
            self.system_status.grid(
                row=0, column=2, padx=(8, 18), sticky="e")

    def _request_native_repaint(self):
        """remap the visible hierarchy after a worker-driven view change."""

        def repaint():
            if self._closing or not self.root.winfo_exists():
                return
            current = self.state.get("last_view", "models")
            frame = self.frames.get(current)
            if frame is not None and not self._monitor_strip:
                frame.grid()
                frame.tkraise()
            self._apply_shell_layout(
                max(150, self.root.winfo_height()), force=True
            )
            self.root.update_idletasks()
            self.root.event_generate("<Expose>", when="tail")

        self.root.after_idle(repaint)
        self.root.after(60, repaint)

    def _quit_app(self):
        """Triggered by the sidebar quit button. Same flow as window close."""
        self._on_close()

    def _show(self, which):
        for name, frame in self.frames.items():
            if name == which:
                frame.grid()
            else:
                frame.grid_remove()
        for name, btn in self.tabs.items():
            if name == which:
                btn.config(fg=FG, bg=TAB_ACTIVE)
            else:
                btn.config(fg=DIM, bg=SIDEBAR_BG)
        titles = {
            "models": ("models", "checkpoints and charge"),
            "chat": ("chat", "inference and interaction"),
            "train": ("train", "config and cockpit"),
            "stream": ("stream", "attune and attend"),
            "logOS": ("logOS", "network and navigate"),
        }
        title, subtitle = titles.get(which, (which, ""))
        self.header_title.config(text=title)
        self.header_subtitle.config(text=subtitle)
        if which == "models" and hasattr(self, "model_list"):
            self._refresh_model_library()
        self.state["last_view"] = which
        save_state(self.state)

    def _set_global_status(self, text, active=False, warning=False):
        self.global_status.config(text=text)
        color = AMBER if warning else (ACCENT if active else DIM)
        self.status_light.config(fg=color)

    def _report_callback_exception(self, exc_type, exc_value, exc_traceback):
        """make tkinter callback failures visible in the console-free app."""
        import traceback
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        message = f"{exc_type.__name__}: {exc_value}"
        if hasattr(self, "global_status"):
            self._set_global_status(
                f"action failed · {message}", warning=True)
        try:
            messagebox.showerror(
                "soma action failed", message, parent=self.root)
        except tk.TclError:
            pass

    def _set_active_checkpoint(self, name=None, detail=""):
        text = name or "no checkpoint active"
        if detail:
            text = f"{text}  ·  {detail}"
        self.header_context.config(text=text)

    # ── widget helpers ──

    def _label(self, parent, text, fg=DIM, font=None):
        return tk.Label(parent, text=text,
                        font=font or self.font_dim, bg=BG, fg=fg)

    def _load_text_file(self, path, max_chars=120_000):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return ""
        return text[-max_chars:]

    def _cap_text_file(self, path, max_bytes):
        try:
            if path.stat().st_size <= max_bytes:
                return
            with open(path, "rb") as f:
                f.seek(-max_bytes, os.SEEK_END)
                data = f.read()
            nl = data.find(b"\n")
            if nl > 0:
                data = data[nl + 1:]
            with open(path, "wb") as f:
                f.write(data)
        except OSError:
            pass

    def _append_text_file(self, path, text, max_bytes=None):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(text)
            if max_bytes:
                self._cap_text_file(path, max_bytes)
        except OSError:
            pass

    def _cap_text_widget(self, widget, max_lines, keep_lines, max_chars):
        idx = int(float(widget.index("end").split(".")[0]))
        if idx > max_lines:
            widget.delete("1.0", f"{idx - keep_lines}.0")
        chars = int(widget.count("1.0", "end", "chars")[0])
        if chars > max_chars:
            drop = chars - max_chars
            widget.delete("1.0", f"1.0+{drop}c")

    def _bind_text_scroll(self, widget):
        def on_wheel(event):
            delta = -1 if event.delta > 0 else 1
            widget.yview_scroll(delta, "units")
            return "break"
        widget.bind("<MouseWheel>", on_wheel)
        widget.bind("<Button-4>", lambda e: widget.yview_scroll(-1, "units"))
        widget.bind("<Button-5>", lambda e: widget.yview_scroll(1, "units"))

    def _save_form_field(self, section, key, value):
        try:
            if section in self.state and isinstance(
                    self.state[section], dict):
                self.state[section][key] = value
                save_state(self.state)
        except Exception:
            pass

    # ── model library ──

    def _build_models(self):
        f = self.frames["models"]
        f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure(1, weight=1)

        left = tk.Frame(f, bg=BG, width=300)
        left.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=14)
        left.grid_propagate(False)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        library_head = tk.Frame(left, bg=BG)
        library_head.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        library_head.grid_columnconfigure(0, weight=1)
        tk.Label(
            library_head, text="local checkpoints", font=self.font_main,
            bg=BG, fg=FG, anchor="w").grid(row=0, column=0, sticky="w")
        self.model_count = tk.Label(
            library_head, text="0", font=self.font_micro,
            bg=BG, fg=DIM, anchor="e")
        self.model_count.grid(row=0, column=1, sticky="e")

        self.model_list = tk.Listbox(
            left, bg=SURFACE, fg=FG, font=self.font_main,
            selectbackground=SELECT_BG, selectforeground=ACCENT,
            activestyle="none", bd=0, highlightthickness=0,
            relief="flat", exportselection=False)
        self.model_list.grid(row=1, column=0, sticky="nsew")
        self.model_list.bind("<<ListboxSelect>>", self._model_library_pick)

        left_actions = tk.Frame(left, bg=BG)
        left_actions.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.model_new_btn = RoundedButton(
            left_actions, "new model", self._library_new_model,
            width=126, height=32, font=self.font_dim)
        self.model_new_btn.pack(side="left")
        self.model_folder_btn = RoundedButton(
            left_actions, "folder", self._open_checkpoints_folder,
            width=92, height=32, font=self.font_dim)
        self.model_folder_btn.pack(side="left", padx=(8, 0))

        right = tk.Frame(f, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(10, 20), pady=14)
        right.grid_rowconfigure(2, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.model_detail_title = tk.Label(
            right, text="select a checkpoint", font=self.font_med,
            bg=BG, fg=FG, anchor="w")
        self.model_detail_title.grid(row=0, column=0, sticky="ew")
        self.model_detail_meta = tk.Label(
            right, text="", font=self.font_micro,
            bg=BG, fg=DIM, anchor="w")
        self.model_detail_meta.grid(row=1, column=0, sticky="ew", pady=(2, 10))

        detail_frame = tk.Frame(right, bg=SURFACE)
        detail_frame.grid(row=2, column=0, sticky="nsew")
        detail_frame.grid_rowconfigure(0, weight=1)
        self.model_detail_panel = detail_frame
        self.model_detail_columns = []
        self._model_detail_values = ("", "", "")
        self._model_detail_width = 900
        for _ in range(3):
            pane = tk.Text(
                detail_frame, bg=SURFACE, fg=DIM, font=self.font_main,
                width=1, height=1, wrap="word", bd=0,
                highlightthickness=1,
                highlightbackground=GRID_LINE,
                highlightcolor=GRID_LINE,
                padx=16, pady=14, selectbackground=SELECT_BG,
                state="disabled")
            self._bind_text_scroll(pane)
            self.model_detail_columns.append(pane)
        self.model_detail_compact = tk.Text(
            detail_frame, bg=SURFACE, fg=DIM, font=self.font_main,
            width=1, height=1, wrap="word", bd=0,
            highlightthickness=1, highlightbackground=GRID_LINE,
            highlightcolor=GRID_LINE, padx=16, pady=14,
            selectbackground=SELECT_BG, state="disabled")
        self._bind_text_scroll(self.model_detail_compact)
        self.model_detail = self.model_detail_columns[0]
        detail_frame.bind(
            "<Configure>",
            lambda event: self._layout_model_detail_panel(event.width))
        self._layout_model_detail_panel(900)

        actions = tk.Frame(right, bg=BG)
        actions.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        self.model_train_btn = RoundedButton(
            actions, "train", self._library_train_selected,
            width=72, height=32, font=self.font_dim)
        self.model_train_btn.pack(side="left")
        self.model_chat_btn = RoundedButton(
            actions, "chat", self._library_chat_selected,
            width=72, height=32, font=self.font_dim)
        self.model_chat_btn.pack(side="left", padx=(6, 0))
        self.model_profile_btn = RoundedButton(
            actions, "edit profile", self._edit_selected_model_profile,
            width=104, height=32, font=self.font_dim)
        self.model_profile_btn.pack(side="left", padx=(6, 0))
        self.model_refresh_btn = RoundedButton(
            actions, "refresh", self._refresh_selected_model_profile,
            width=76, height=32, font=self.font_dim)
        self.model_refresh_btn.pack(side="left", padx=(6, 0))
        self.model_train_btn.set_enabled(False)
        self.model_chat_btn.set_enabled(False)
        self.model_profile_btn.set_enabled(False)
        self.model_refresh_btn.set_enabled(False)
        self.model_paths = []
        self.model_selected = None
        self._refresh_model_library()

    def _set_model_detail(self, text):
        if hasattr(self, "model_detail_columns"):
            self._set_model_detail_columns(text, "", "")
            return
        self._set_readonly_text(self.model_detail, text)

    def _set_model_detail_columns(self, identity, branding, diagnostics):
        if not hasattr(self, "model_detail_columns"):
            return
        self._model_detail_values = (
            str(identity), str(branding), str(diagnostics))
        self._render_model_detail_columns()

    def _render_model_detail_columns(self):
        if not hasattr(self, "model_detail_columns"):
            return
        identity, branding, diagnostics = self._model_detail_values
        if self._model_detail_width < 700:
            combined = "\n\n".join(
                value for value in (identity, branding, diagnostics)
                if value.strip())
            self._set_readonly_text(self.model_detail_compact, combined)
            return
        self._set_readonly_text(self.model_detail_columns[0], identity)
        self._set_readonly_text(
            self.model_detail_columns[1], branding, ACCENT)
        self._set_readonly_text(self.model_detail_columns[2], diagnostics)

    def _layout_model_detail_panel(self, width):
        if not hasattr(self, "model_detail_columns"):
            return
        self._model_detail_width = int(width)
        for pane in self.model_detail_columns:
            pane.grid_forget()
        self.model_detail_compact.grid_forget()
        panel = self.model_detail_panel
        panel.grid_rowconfigure(0, weight=1)
        for column in range(3):
            panel.grid_columnconfigure(
                column, weight=1 if width >= 700 else 0,
                minsize=0, uniform=(
                    "model_detail" if width >= 700 else ""))
        if width >= 700:
            for column, pane in enumerate(self.model_detail_columns):
                pane.grid(row=0, column=column, sticky="nsew")
        else:
            panel.grid_columnconfigure(0, weight=1)
            self.model_detail_compact.grid(
                row=0, column=0, columnspan=3, sticky="nsew")
        self._render_model_detail_columns()

    def _refresh_model_library(self):
        selected_name = (
            self.model_selected.name if self.model_selected else None)
        paths = sorted(
            CHECKPOINTS_DIR.glob("*.pt"),
            key=lambda path: path.stat().st_mtime, reverse=True)
        self.model_paths = paths
        self.model_list.delete(0, "end")
        selected_index = None
        for i, path in enumerate(paths):
            size = _fmt_bytes(path.stat().st_size)
            self.model_list.insert("end", f"{path.stem}  ·  {size}")
            if path.name == selected_name:
                selected_index = i
        self.model_count.config(text=f"{len(paths)} local")
        if not paths:
            self.model_list.insert("end", "no checkpoints yet")
            self.model_selected = None
            self.model_train_btn.set_enabled(False)
            self.model_chat_btn.set_enabled(False)
            self.model_profile_btn.set_enabled(False)
            self.model_refresh_btn.set_enabled(False)
            self.model_detail_title.config(text="no checkpoints yet")
            self.model_detail_meta.config(text="")
            self._set_model_detail(
                "begin with a new model, then give it a corpus.\n\n"
                "saved models will live here with their training history.")
        elif selected_index is not None:
            self.model_list.selection_set(selected_index)
            self.model_list.see(selected_index)

    def _model_library_pick(self, _event=None):
        selection = self.model_list.curselection()
        if not selection:
            return
        index = int(selection[0])
        if index >= len(self.model_paths):
            return
        path = self.model_paths[index]
        self.model_selected = path
        stat = path.stat()
        modified = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(stat.st_mtime))
        self.model_detail_title.config(text=path.stem)
        self.model_detail_meta.config(
            text=f"{_fmt_bytes(stat.st_size)}  ·  modified {modified}")
        metadata = _read_checkpoint_metadata(path)
        if metadata:
            description = metadata.get("description") or "no description"
            profile = _checkpoint_profile(path, metadata)
            parameter_count = int(metadata.get("params", 0) or 0)
            parameter_text = (
                f"{parameter_count / 1e9:.2f}b"
                if parameter_count >= 1_000_000_000 else
                f"{parameter_count / 1e6:.1f}m"
                if parameter_count >= 1_000_000 else
                f"{parameter_count / 1e3:.1f}k")
            identity = "\n".join((
                "description",
                "",
                description,
                "",
                f"architecture  {metadata.get('architecture', 'soma')}",
                f"shape         {metadata.get('n_bands', '?')} bands  ·  "
                f"symbol d{metadata.get('token_dim', '?')}  ·  "
                f"model d{metadata.get('model_dim', '?')}",
                f"scale         rank {metadata.get('scale_rank', '?')}  ·  "
                f"depth {metadata.get('scale_depth', '?')}",
                f"composition   depth "
                f"{metadata.get('language_depth', '?')}",
                f"parameters    {parameter_text}",
            ))
            branding = "\n".join((
                profile["name"],
                profile["tagline"],
                "",
                profile["art"],
                "",
                profile["about"],
            )).rstrip()
            diagnostics = "\n".join((
                "experience",
                "",
                f"experience    {_fmt_bytes(metadata.get('bytes_seen', 0))}",
                f"tokens        {_fmt_tokens(metadata.get('tokens_seen', 0))}",
                "",
                "learning",
                "",
                f"optimizer     adamw  ·  lr "
                f"{metadata.get('lr', 0):.5g}",
                f"update clock  every "
                "io2 pulse scheduling",
                "",
                f"checkpoint id {metadata.get('checkpoint_id', 'not saved')}",
            ))
            self._set_model_detail_columns(
                identity, branding, diagnostics)
        else:
            self._set_model_detail(
                f"{path.name}\n\n"
                "legacy checkpoint metadata is embedded with the weights. "
                "it will become instantly inspectable here after its next "
                "save.\n\n"
                "train opens its saved architecture and diagnostic history. "
                "chat loads it for generation and optional online "
                "learning.")
        self.model_train_btn.set_enabled(True)
        self.model_chat_btn.set_enabled(True)
        self.model_profile_btn.set_enabled(True)
        self.model_refresh_btn.set_enabled(True)

    def _edit_selected_model_profile(self):
        if not self.model_selected:
            return
        metadata = _read_checkpoint_metadata(self.model_selected)
        profile = _checkpoint_profile(self.model_selected, metadata)
        try:
            meta_path = _write_checkpoint_profile(
                self.model_selected, profile)
            subprocess.Popen(["open", "-t", str(meta_path)])
        except OSError as error:
            self.model_detail_meta.config(
                text=f"profile open failed: {error}")

    def _refresh_selected_model_profile(self):
        if not self.model_selected:
            return
        selected = self.model_selected
        try:
            index = self.model_paths.index(selected)
        except ValueError:
            self._refresh_model_library()
            return
        self.model_list.selection_clear(0, "end")
        self.model_list.selection_set(index)
        self._model_library_pick()

    def _library_new_model(self):
        self._show("train")
        self.train_ckpt_dd._pick("(new model)")

    def _library_train_selected(self):
        if not self.model_selected:
            return
        name = self.model_selected.name
        self.train_ckpt_dd.set_options(
            self._list_checkpoint_names_with_none(), value=name)
        self._on_train_ckpt_change(name)
        self._show("train")

    def _library_chat_selected(self):
        if not self.model_selected:
            return
        name = self.model_selected.name
        self.chat_ckpt_dd.set_options(
            self._list_checkpoint_names(), value=name)
        self._save_form_field("chat", "checkpoint", name)
        self._show("chat")

    # ── chat view ──

    def _build_chat(self):
        f = self.frames["chat"]
        f.grid_rowconfigure(4, weight=1)
        f.grid_columnconfigure(0, weight=1)

        # row 1 — checkpoint picker + load/unload
        top = tk.Frame(f, bg=BG)
        top.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 6))

        self._label(top, "checkpoint").pack(side="left")

        names = self._list_checkpoint_names()
        last = self.state["chat"].get("checkpoint", "")
        if last not in names:
            last = names[0] if names else ""
        self.chat_ckpt_dd = RoundedDropdown(
            top, names, value=last,
            on_select=lambda v: self._save_form_field(
                "chat", "checkpoint", v),
            width=260, height=30, font=self.font_main)
        self.chat_ckpt_dd.pack(side="left", padx=(8, 12))

        self.chat_load_btn = RoundedButton(
            top, "load", self._chat_load,
            width=80, height=30, font=self.font_dim)
        self.chat_load_btn.pack(side="left", padx=(0, 8))

        self.chat_unload_btn = RoundedButton(
            top, "unload", self._chat_unload,
            width=90, height=30, font=self.font_dim)
        self.chat_unload_btn.pack(side="left")
        self.chat_unload_btn.set_enabled(False)

        # row 2 — settings + status
        settings = tk.Frame(f, bg=BG)
        settings.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 10))

        self.chat_online_cb = RoundedCheckbox(
            settings, "online", value=self.state["chat"]["online"],
            on_change=lambda v: self._save_form_field(
                "chat", "online", v),
            width=110, height=24, font=self.font_dim)
        self.chat_online_cb.pack(side="left")

        self._label(settings, "response max tokens").pack(
            side="left", padx=(20, 6))
        self.chat_maxlen_entry = RoundedEntry(
            settings, width=80, height=26,
            value=self.state["chat"].get("max_length", "200"),
            font=self.font_dim)
        self.chat_maxlen_entry.bind_change(
            lambda v: self._save_form_field("chat", "max_length", v))
        self.chat_maxlen_entry.pack(side="left")

        self._label(settings, "temperature").pack(
            side="left", padx=(18, 6))
        self.chat_temp_entry = RoundedEntry(
            settings, width=70, height=26,
            value=self.state["chat"].get("temperature", "0.8"),
            font=self.font_dim)
        self.chat_temp_entry.bind_change(
            lambda v: self._save_form_field("chat", "temperature", v))
        self.chat_temp_entry.pack(side="left")

        self.chat_status = self._label(settings, "no model loaded")
        self.chat_status.pack(side="left", padx=(20, 0))

        # row 2b — entirely local speech input and output
        speech_row = tk.Frame(f, bg=BG)
        speech_row.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 10))

        self.chat_voice_cb = RoundedCheckbox(
            speech_row, "speak replies",
            value=self.state["chat"].get("voice", False),
            on_change=self._toggle_voice,
            width=150, height=24, font=self.font_dim)
        self.chat_voice_cb.pack(side="left")

        self.chat_voice_status = self._label(
            speech_row, "local voice · microphone idle")
        self.chat_voice_status.pack(side="left", padx=(18, 0))
        self.voice.enabled = self.chat_voice_cb.get()

        # row 3 — logOS remote chat
        logos_row = tk.Frame(f, bg=BG)
        logos_row.grid(row=3, column=0, sticky="w", padx=20, pady=(0, 10))

        self.logos_cb = RoundedCheckbox(
            logos_row, "logOS remote",
            value=False,
            on_change=self._toggle_logos_mode,
            width=160, height=24, font=self.font_dim)
        self.logos_cb.pack(side="left")

        self._label(logos_row, "pass prompt").pack(
            side="left", padx=(20, 6))
        self.logos_pass_entry = RoundedEntry(
            logos_row, width=200, height=26,
            value=self.state["chat"].get("pass_prompt", ""),
            font=self.font_dim)
        self.logos_pass_entry.bind_change(
            lambda v: self._save_form_field("chat", "pass_prompt", v))
        self.logos_pass_entry.pack(side="left")

        self.logos_status = self._label(logos_row, "")
        self.logos_status.pack(side="left", padx=(16, 0))

        # transcript
        transcript = tk.Frame(f, bg=BG)
        transcript.grid(row=4, column=0, sticky="nsew",
                        padx=20, pady=(0, 10))
        transcript.grid_rowconfigure(0, weight=1)
        transcript.grid_columnconfigure(0, weight=1)

        self.chat_text = tk.Text(
            transcript, bg=SURFACE, fg=FG, font=self.font_main,
            wrap="word", bd=0, highlightthickness=0,
            insertbackground=FG, padx=20, pady=14,
            selectbackground=SELECT_BG, state="disabled")
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        self.chat_text.tag_config("you", foreground=ACCENT)
        self.chat_text.tag_config("soma", foreground=FG)
        self.chat_text.tag_config("dim", foreground=DIM)
        history = self._load_text_file(CHAT_LOG_FILE)
        if history:
            self.chat_text.config(state="normal")
            self.chat_text.insert("1.0", history, "dim")
            self.chat_text.see("end")
            self.chat_text.config(state="disabled")

        # input
        bottom = tk.Frame(f, bg=BG)
        bottom.grid(row=5, column=0, sticky="ew", padx=20, pady=(0, 20))
        bottom.grid_columnconfigure(0, weight=1)

        self.chat_input_frame = tk.Frame(bottom, bg=SURFACE)
        self.chat_input_frame.grid(row=0, column=0, sticky="ew",
                                   padx=(0, 8))
        self.chat_input_frame.grid_columnconfigure(0, weight=1)

        self.chat_input = tk.Text(
            self.chat_input_frame, bg=SURFACE, fg=FG,
            font=self.font_main, wrap="word",
            height=1, bd=0, highlightthickness=0,
            insertbackground=FG, padx=14, pady=8,
            selectbackground=SELECT_BG)
        self.chat_input.grid(row=0, column=0, sticky="ew")
        self.chat_input.bind("<Return>", self._chat_input_return)
        self.chat_input.bind("<Shift-Return>", self._chat_input_newline)
        self.chat_input.bind("<KeyRelease>", self._resize_chat_input)

        self.chat_send_btn = RoundedButton(
            bottom, "send", self._chat_send,
            width=80, height=36, font=self.font_dim)
        self.chat_send_btn.grid(row=0, column=2, sticky="s")

        self.chat_mic_btn = RoundedButton(
            bottom, "speak", self._chat_listen,
            width=88, height=36, font=self.font_dim)
        self.chat_mic_btn.grid(row=0, column=1, sticky="s", padx=(0, 8))

    def _list_checkpoint_names(self):
        files = sorted(p.name for p in CHECKPOINTS_DIR.glob("*.pt"))
        return files if files else ["(no checkpoints yet)"]

    def _refresh_chat_ckpts(self):
        names = self._list_checkpoint_names()
        last = self.state["chat"].get("checkpoint", "")
        self.chat_ckpt_dd.set_options(names, value=last)

    def _chat_load(self):
        name = self.chat_ckpt_dd.value
        if not name or name.startswith("("):
            self.chat_status.config(text="no checkpoint selected")
            return
        path = CHECKPOINTS_DIR / name
        if not path.exists():
            self.chat_status.config(text=f"not found: {name}")
            return
        self.chat_status.config(text="loading...")
        self.chat_load_btn.set_text("loading…")
        self.chat_load_btn.set_enabled(False)
        self._set_global_status(f"loading {name}", active=True)

        # speaker name = checkpoint stem (e.g. priant.pt → priant)
        speaker = Path(name).stem

        try:
            max_length = int(self.chat_maxlen_entry.get())
        except (ValueError, TypeError):
            max_length = 200
        try:
            temperature = max(0.05, float(self.chat_temp_entry.get()))
        except (ValueError, TypeError):
            temperature = 0.8
        active_training_path = Path(self._train_save_path).expanduser()
        training_snapshot = bool(
            self.train_worker.is_running()
            and self._train_save_path
            and path.resolve() == active_training_path.resolve()
        )
        online_mode = self.chat_online_cb.get() and not training_snapshot

        def thread():
            try:
                self.chat.load(
                    str(path),
                    online=online_mode,
                    temperature=temperature,
                    max_length=max_length)
                self._chat_speaker = speaker
                self._chat_loaded_path = str(path)
                self._chat_readonly_snapshot = training_snapshot
                mode = (
                    "training snapshot"
                    if training_snapshot
                    else ("online" if online_mode else "context")
                )
                self.root.after(0, lambda: self.chat_status.config(
                    text=f"loaded {name} · {mode}"))
                self.root.after(0, lambda: self._set_active_checkpoint(
                    Path(name).stem, mode))
                self.root.after(0, lambda: self._set_global_status(
                    f"{Path(name).stem} ready", active=True))
                self.root.after(0, lambda: self._chat_append(
                    f"loaded {name}\n", "dim"))
                self.root.after(0,
                    lambda: self.chat_unload_btn.set_enabled(True))
                self.root.after(0, self._finish_chat_load_action)
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: self.chat_status.config(
                    text="load failed"))
                self.root.after(0, lambda: self._set_global_status(
                    "checkpoint load failed", warning=True))
                self.root.after(0, lambda: self._chat_append(
                    f"load failed: {msg}\n", "dim"))
                self.root.after(0, self._finish_chat_load_action)

        threading.Thread(target=thread, daemon=True).start()

    def _finish_chat_load_action(self):
        self.chat_load_btn.set_text("load")
        self.chat_load_btn.set_enabled(True)

    def _restore_chat_training_config(self):
        """Restore checkpoint hyperparameters before saving chat state."""
        model = self.chat.model
        if model is None:
            return
        if hasattr(model, 'checkpoint_batch_size'):
            model.batch_size = model.checkpoint_batch_size

    def _chat_unload(self):
        """Unload the chat model, optionally saving any changed state.

        Yes = save then unload. No = discard runtime changes and unload.
        Cancel = keep the model loaded.
        """
        if self.chat.model is None:
            return

        path = getattr(self, '_chat_loaded_path', None)
        name = Path(path).name if path else "model"
        should_save = False

        if getattr(self.chat, 'dirty', False) and not self._chat_readonly_snapshot:
            choice = messagebox.askyesnocancel(
                "unload model",
                "save changes before unloading?\n\n"
                "yes: save & unload\n"
                "no: unload without saving\n"
                "cancel: keep model loaded",
                parent=self.root,
            )
            if choice is None:
                self._closing = False
                return
            should_save = bool(choice)

        if should_save:
            if not path:
                self._chat_append("nothing to save (no source path)\n", "dim")
                return
            self.chat_status.config(text="saving...")
            self.root.update_idletasks()

            def thread():
                try:
                    self._restore_chat_training_config()
                    self.chat.model.save(path)
                    self.chat.dirty = False
                    self.root.after(0, lambda: self._finish_chat_unload(
                        f"saved & unloaded {name}\n"))
                except Exception as e:
                    msg = f"{type(e).__name__}: {e}"
                    self.root.after(0, lambda: self._chat_append(
                        f"save failed: {msg}\n", "dim"))
                    self.root.after(0, lambda: self.chat_status.config(
                        text="save failed"))

            threading.Thread(target=thread, daemon=True).start()
        else:
            note = (
                f"unloaded training snapshot {name}; live lineage unchanged\n"
                if self._chat_readonly_snapshot
                else f"unloaded {name} without saving\n"
            )
            self._finish_chat_unload(note)

    def _finish_chat_unload(self, message=None):
        """Clear chat model/UI state after save or discard."""
        if message:
            self._chat_append(message, "dim")
        self.chat.release_model()
        self.voice.stop()
        self.speech.cancel()
        self.chat.dirty = False
        self._chat_speaker = None
        self._chat_loaded_path = None
        self._chat_readonly_snapshot = False
        self.chat_status.config(text="no model loaded")
        self._set_active_checkpoint()
        self._set_global_status("model unloaded")
        self.chat_unload_btn.set_enabled(False)

    def _chat_input_return(self, event):
        self._chat_send()
        return "break"

    def _chat_input_newline(self, event):
        self.chat_input.insert("insert", "\n")
        self._resize_chat_input()
        return "break"

    def _toggle_voice(self, enabled):
        enabled = bool(enabled)
        self.voice.enabled = enabled
        self._save_form_field("chat", "voice", enabled)
        if not enabled:
            self.voice.stop()
        status = "local voice on" if enabled else "local voice off"
        if not self.speech.listening:
            status += " · microphone idle"
        self.chat_voice_status.config(text=status)

    def _chat_listen(self):
        if self.speech.listening:
            self.speech.stop()
            self.chat_mic_btn.set_text("finishing…")
            self.chat_mic_btn.set_enabled(False)
            return
        if self._chat_pending:
            return
        existing = self.chat_input.get("1.0", "end-1c").strip()
        self._speech_prefix = (existing + " ") if existing else ""
        self._speech_received_final = False
        locale = self.state["chat"].get("speech_locale", "auto")
        if self.speech.start(locale):
            self.chat_mic_btn.set_text("stop mic")
            self.chat_voice_status.config(text="local speech starting…")

    def _set_speech_input(self, transcript):
        text = (self._speech_prefix + str(transcript)).strip()
        self.chat_input.delete("1.0", "end")
        self.chat_input.insert("1.0", text)
        self._resize_chat_input()
        self.chat_input.see("end")

    def _resize_chat_input(self, event=None):
        text = self.chat_input.get("1.0", "end-1c")
        logical_lines = text.count("\n") + 1
        try:
            counted = self.chat_input.count("1.0", "end", "displaylines")
            display_lines = max(1, int(counted[0] if counted else 1))
        except tk.TclError:
            display_lines = logical_lines
        lines = min(5, max(1, logical_lines, display_lines))
        if str(self.chat_input.cget("height")) != str(lines):
            self.chat_input.config(height=lines)

    def _chat_send(self):
        if self._chat_pending:
            self._chat_stop()
            return
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not text or self.chat.model is None:
            return
        self.chat_input.delete("1.0", "end")
        self._resize_chat_input()
        # pick up live settings each send — user may have changed them
        self.chat.online = (
            self.chat_online_cb.get() and not self._chat_readonly_snapshot
        )
        self.chat_status.config(
            text="online queued" if self.chat.online else "context queued")
        try:
            self.chat.max_length = int(self.chat_maxlen_entry.get())
        except (ValueError, TypeError):
            self.chat.max_length = 200
        try:
            self.chat.temperature = max(0.05, float(self.chat_temp_entry.get()))
            self._save_form_field("chat", "temperature",
                                  self.chat_temp_entry.get())
        except (ValueError, TypeError):
            self.chat.temperature = 0.8
        speaker = getattr(self, '_chat_speaker', None) or 'soma'
        self.voice.enabled = self.chat_voice_cb.get()
        if self.voice.enabled:
            self.voice.begin()
        # right-align the two speaker labels by padding to a common width
        w = max(4, len(speaker))
        you_label = "you".ljust(w)
        soma_label = speaker.ljust(w)
        self._chat_append(f"{you_label} › {text}\n", "you")
        self._chat_append(f"{soma_label} › ", "soma")
        self._set_chat_pending(True)
        self.chat.submit(text)

    def _chat_stop(self):
        self.chat.stop_generation()
        self.voice.stop()
        self.chat_status.config(text="stopping generation...")
        self.chat_send_btn.set_text("stopping…")
        self.chat_send_btn.set_enabled(False)

    def _set_chat_pending(self, pending):
        self._chat_pending = bool(pending)
        self.chat_send_btn.set_text("stop" if pending else "send")
        self.chat_send_btn.set_enabled(True)
        self.chat_mic_btn.set_enabled(not pending)

    def _chat_append(self, text, tag, persist=True):
        self.chat_text.config(state="normal")
        self.chat_text.insert("end", text, tag)
        self._cap_text_widget(
            self.chat_text, max_lines=8200, keep_lines=8000,
            max_chars=1_000_000)
        self.chat_text.see("end")
        self.chat_text.config(state="disabled")
        if persist:
            self._append_text_file(
                CHAT_LOG_FILE, text, max_bytes=MAX_CHAT_LOG_BYTES)

    def _toggle_logos_mode(self, enabled):
        """Start or stop the logOS bridge subprocess."""
        self._logos_mode = enabled
        self._save_form_field("chat", "logos_mode", enabled)

        if enabled:
            if self.chat.model is None:
                self.logos_status.config(text="load a model first")
                self.logos_cb.value = False
                self.logos_cb._update_display()
                self._logos_mode = False
                return

            pass_prompt = self.logos_pass_entry.get().strip()
            if not pass_prompt:
                self.logos_status.config(text="enter a pass prompt")
                self.logos_cb.value = False
                self.logos_cb._update_display()
                self._logos_mode = False
                return

            self.logos_status.config(text="connecting...")
            self._logos_connected = False
            bridge_script = str(BUNDLE_DIR / "soma_logos_bridge.py")
            self.logos_bridge.start([
                bridge_script,
                "--pass-prompt", pass_prompt,
                "--api-base", "https://logossoma.com",
            ], cwd=str(RUNTIME_DIR))
            self._chat_append("logOS remote enabled — waiting for "
                              "prompts from the web\n", "dim")
        else:
            self.logos_bridge.stop()
            self._logos_connected = False
            self.logos_status.config(text="")
            self.logos_response_buf.clear()
            self._chat_append("logOS remote disabled\n", "dim")

    # ── train view ──
    #
    # staged flow:
    #   idle        → "load existing" or "new"
    #   configured  → params editable, big start button
    #   training    → live readout, stop button
    #   stopped     → params editable, start button (continues)
    #
    # the loop checkbox lives in the params group. when checked, train
    # never finishes naturally — it cycles on the head of the corpus
    # forever until stopped.

    def _build_train(self):
        f = self.frames["train"]
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(0, weight=1)

        # we stack two frames in the same cell and raise one at a time
        self.train_card = tk.Frame(f, bg=BG)
        self.train_card.grid(row=0, column=0, sticky="nsew",
                             padx=16, pady=16)
        self.train_card.grid_columnconfigure(0, weight=1)
        self.train_card.grid_rowconfigure(0, weight=1)

        self._build_train_setup_v2()
        self._build_train_live()

        self._set_train_state('idle')

    def _build_train_setup(self):
        """The setup view. Three sections: source (load or new),
        params, action. All shown together so the user can fluidly
        switch between resuming and starting fresh."""
        self.train_setup = tk.Frame(self.train_card, bg=BG)
        self.train_setup.grid(row=0, column=0, sticky="nsew")
        self.train_setup.grid_columnconfigure(0, weight=1)

        # ── source picker ──
        src = tk.Frame(self.train_setup, bg=BG)
        src.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self._label(src, "load checkpoint", font=self.font_main, fg=FG
                    ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        names = self._list_checkpoint_names_with_none()
        last = self.state["train"].get("checkpoint", "(new model)")
        if last not in names:
            last = names[0] if names else "(new model)"
        self.train_ckpt_dd = RoundedDropdown(
            src, names, value=last,
            on_select=self._on_train_ckpt_change,
            width=260, height=32, font=self.font_main)
        self.train_ckpt_dd.grid(row=1, column=0, sticky="w")
        self.train_open_ckpts_btn = RoundedButton(
            src, "open checkpoints", self._open_checkpoints_folder,
            width=160, height=32, font=self.font_dim)
        self.train_open_ckpts_btn.grid(
            row=1, column=1, sticky="w", padx=(12, 0))

        # ── params grid ──
        params = tk.Frame(self.train_setup, bg=BG)
        params.grid(row=1, column=0, sticky="ew", pady=(6, 10))

        self._label(params, "configuration", font=self.font_main, fg=FG
                    ).grid(row=0, column=0, columnspan=4,
                           sticky="w", pady=(0, 8))

        self.train_fields = {}
        s = self.state["train"]

        # defaults for a new model — these match the CLI defaults
        _DEFAULTS = {
            "bands": "40", "base": "1.618033988749895", "hidden": "4096",
            "layers": "4",
            "lr": "auto 0.0003", "grad_clip": "1.0",
            "batch": "512", "auto_mode": DEFAULT_AUTO_MODE,
            "decimation": default_decimation_for_mode(DEFAULT_AUTO_MODE),
            "start_byte": "0", "autosave_min": "30",
            "epochs": "1", "dream_every_batches": "64", "dream_length": "128",
            "dream_temperature": "1.0",
            "description": "",
        }

        def field(label, key, row, col, width=108):
            self._label(params, label).grid(
                row=row, column=col, sticky="w", padx=(0, 8), pady=3)
            v = s.get(key, _DEFAULTS.get(key, ""))
            ent = RoundedEntry(
                params, width=width, height=30, value=str(v),
                font=self.font_main)
            if key == "auto_mode":
                ent.bind_change(self._on_auto_mode_change)
            else:
                ent.bind_change(
                    lambda val, field_key=key: self._save_form_field(
                        "train", field_key, val))
            ent.grid(row=row, column=col + 1, sticky="w",
                     padx=(0, 14), pady=3)
            self.train_fields[key] = ent

        # corpus / stream toggle row
        self._label(params, "data source").grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=4)

        ds_row = tk.Frame(params, bg=BG)
        ds_row.grid(row=1, column=1, columnspan=3, sticky="w", pady=4)

        current_mode = s.get("mode", "corpus")
        if current_mode not in ("corpus", "stream"):
            current_mode = "corpus"
        self.train_mode_seg = RoundedSegmented(
            ds_row, ["corpus", "stream"], value=current_mode,
            on_change=self._on_train_mode_change,
            width=150, height=30, font=self.font_dim)
        self.train_mode_seg.pack(side="left", padx=(0, 10))

        # corpus filename entry — visible when mode == "corpus"
        self.train_corpus_entry = RoundedEntry(
            ds_row, width=170, height=30, value=s.get("corpus", ""),
            font=self.font_main)
        self.train_corpus_entry.bind_change(
            self._on_train_corpus_text_change)

        self.train_choose_data_btn = RoundedButton(
            ds_row, "data", self._show_train_data_menu,
            width=58, height=30, font=self.font_dim)

        self.train_open_data_btn = RoundedButton(
            ds_row, "open data", self._open_data_folder,
            width=100, height=30, font=self.font_dim)

        # stream picker — visible when mode == "stream"
        stream_names = [st["name"] for st in self.streams] or \
                       ["(no streams found)"]
        last_stream = s.get("stream_name") or stream_names[0]
        if last_stream not in stream_names:
            last_stream = stream_names[0]
        self.train_stream_dd = RoundedDropdown(
            ds_row, stream_names, value=last_stream,
            on_select=lambda v: self._save_form_field(
                "train", "stream_name", v),
            width=220, height=30, font=self.font_main)

        self.train_head_row = tk.Frame(params, bg=BG)
        self._label(self.train_head_row, "head MB").pack(
            side="left", padx=(0, 8))
        self.train_head_entry = RoundedEntry(
            self.train_head_row, width=108, height=30,
            value=s.get("head_mb", "50"), font=self.font_main)
        self.train_head_entry.bind_change(
            lambda v: self._save_form_field("train", "head_mb", v))
        self.train_head_entry.pack(side="left")

        # show one based on current mode
        if current_mode == "corpus":
            self.train_corpus_entry.pack(side="left")
            self.train_choose_data_btn.pack(side="left", padx=(8, 0))
            self.train_open_data_btn.pack(side="left", padx=(8, 0))
            self.train_head_row.grid_remove()
        else:
            self.train_stream_dd.pack(side="left")
            self.train_head_row.grid(
                row=11, column=1, sticky="w", pady=3)

        # save_path — persisted so user doesn't have to retype
        self._label(params, "save as").grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        self.train_fields["save_path"] = RoundedEntry(
            params, width=430, height=30, value=s.get("save_path", "model.pt"),
            font=self.font_main)
        self.train_fields["save_path"].bind_change(
            lambda val: self._save_form_field("train", "save_path", val))
        self.train_fields["save_path"].grid(
            row=2, column=1, columnspan=3, sticky="w", pady=4)

        self._label(params, "description").grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=4)
        self.train_fields["description"] = RoundedEntry(
            params, width=430, height=30,
            value=s.get("description", _DEFAULTS["description"]),
            font=self.font_main)
        self.train_fields["description"].bind_change(
            lambda val: self._save_form_field("train", "description", val))
        self.train_fields["description"].grid(
            row=3, column=1, columnspan=3, sticky="w", pady=4)

        # model params — defaults shown, overwritten when a checkpoint is loaded
        field("bands", "bands", 4, 0)
        field("depth", "layers", 4, 2)
        field("hidden", "hidden", 5, 0)
        field("base", "base", 5, 2)
        field("batch tokens", "batch", 6, 0)
        field("lr", "lr", 6, 2)
        field("gradclip", "grad_clip", 7, 0)
        field("auto mode", "auto_mode", 7, 2)
        field("decimation gain", "decimation", 8, 0)
        field("start byte", "start_byte", 8, 2)
        field("autosave min", "autosave_min", 9, 0)
        field("dream every updates", "dream_every_batches", 9, 2)
        field("dream max tokens", "dream_length", 10, 0)
        field("temperature", "dream_temperature", 10, 2)

        # ── action row ──
        self.train_actions = tk.Frame(params, bg=BG)

        self.train_start_btn = RoundedButton(
            self.train_actions, "start", self._train_start_safe,
            width=120, height=36, font=self.font_main)
        self.train_start_btn.pack(side="left")

        self.train_setup_status = self._label(self.train_actions, "")
        self.train_setup_status.pack(side="left", padx=(16, 0))
        self._place_train_actions(current_mode)

        info_label = self._label(
            self.train_setup, "checkpoint info",
            font=self.font_main, fg=FG)
        info_label.grid(row=2, column=0, sticky="w", pady=(10, 6))

        info_frame = tk.Frame(self.train_setup, bg=SURFACE)
        info_frame.grid(row=3, column=0, sticky="nsew")
        info_frame.grid_rowconfigure(0, weight=1)
        info_frame.grid_columnconfigure(0, weight=1)
        self.train_setup.grid_rowconfigure(3, weight=1)

        self.train_ckpt_info = tk.Text(
            info_frame, bg=SURFACE, fg=DIM, font=self.font_dim,
            height=8, wrap="word", bd=0, highlightthickness=0,
            padx=14, pady=10, selectbackground=SELECT_BG)
        self.train_ckpt_info.grid(row=0, column=0, sticky="nsew")
        self.train_ckpt_info.config(state="disabled")
        self._bind_text_scroll(self.train_ckpt_info)
        self._set_train_ckpt_info(
            "select a checkpoint to inspect id history, saved settings, "
            "and weight energy.")

    def _build_train_setup_v2(self):
        """Instrument-style setup grouped by conceptual responsibility."""
        self._train_setup_v2 = True
        self.train_setup = tk.Frame(self.train_card, bg=BG)
        self.train_setup.grid(row=0, column=0, sticky="nsew")
        self.train_setup.grid_columnconfigure(0, weight=1)
        self.train_setup.grid_rowconfigure(1, weight=0)

        source = tk.Frame(self.train_setup, bg=BG)
        source.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        source.grid_columnconfigure(1, weight=1)
        self._label(source, "checkpoint", fg=DIM).grid(
            row=0, column=0, sticky="w", padx=(0, 8))
        names = self._list_checkpoint_names_with_none()
        last = self.state["train"].get("checkpoint", "(new model)")
        if last not in names:
            last = "(new model)"
        self.train_ckpt_dd = RoundedDropdown(
            source, names, value=last, on_select=self._on_train_ckpt_change,
            width=300, height=32, font=self.font_main)
        self.train_ckpt_dd.grid(row=0, column=1, sticky="w")
        self._label(source, "action", fg=DIM).grid(
            row=0, column=2, sticky="e", padx=(12, 8))
        checkpoint_action = self.state["train"].get(
            "checkpoint_action", "continue")
        if checkpoint_action not in ("continue", "fork"):
            checkpoint_action = "continue"
        self.train_checkpoint_action_dd = RoundedDropdown(
            source, ["continue", "fork"], value=checkpoint_action,
            on_select=self._on_train_checkpoint_action,
            width=108, height=32, font=self.font_main)
        self.train_checkpoint_action_dd.grid(row=0, column=3, sticky="w")
        self.train_open_ckpts_btn = RoundedButton(
            source, "open folder", self._open_checkpoints_folder,
            width=112, height=32, font=self.font_dim)
        self.train_open_ckpts_btn.grid(row=0, column=4, padx=(10, 0))

        body = tk.Frame(self.train_setup, bg=BG)
        body.grid(row=1, column=0, sticky="nsew")
        for column in range(3):
            body.grid_columnconfigure(column, weight=1, uniform="setup")
        body.grid_rowconfigure(0, weight=1)

        sections = []
        self.train_section_titles = []
        for column, title in enumerate(("shape", "feed", "automatic")):
            section = tk.Frame(body, bg=BG)
            section.grid(row=0, column=column, sticky="nsew",
                         padx=((0 if column == 0 else 14),
                               (0 if column == 2 else 14)))
            section.grid_columnconfigure(1, weight=1)
            heading = tk.Label(
                section, text=title, font=self.font_main,
                bg=BG, fg=ACCENT, anchor="w")
            heading.grid(
                row=0, column=0, columnspan=2,
                sticky="ew", pady=(0, 8))
            self.train_section_titles.append(heading)
            sections.append(section)
        shape, feed, dynamics = sections

        self.train_fields = {}
        state = self.state["train"]
        defaults = {
            "bands": "48", "capacity": DEFAULT_CAPACITY,
            "base": "1.618033988749895",
            "symbol_dim": "", "scale_rank": "", "scale_depth": "",
            "summaries": "", "model_dim": "", "trunk_depth": "",
            "expansion_dim": "",
            "lr": "auto 0.0003", "grad_clip": "1.0",
            "decimation": "1", "lattice_rate": "0.002",
            "batch": "512", "update_interval": "1",
            "start_byte": "0", "autosave_min": "30",
            "epochs": "1", "dream_every_batches": "64", "dream_length": "128",
            "dream_temperature": "1.0", "save_path": "model.pt",
            "description": "",
            "validation_every_mb": "100",
            "validation_max_mb": "0.25",
        }

        self.train_advanced_controls = []
        self.train_shape_fields = []

        def field(section, label, key, row, width=132, prominent=False,
                  advanced=False, shape_field=False):
            label_font = self.font_main if prominent else self.font_dim
            entry_font = self.font_main
            field_label = self._label(section, label, font=label_font)
            field_label.grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = RoundedEntry(
                section, width=width, height=(32 if prominent else 29),
                value=str(state.get(key, defaults.get(key, ""))),
                font=entry_font)

            def save_value(value, field_key=key):
                self._save_form_field("train", field_key, value)
                if field_key == "start_byte":
                    self._train_start_byte_override = True
                if field_key in ("bands", "base"):
                    self._refresh_train_shape_preview()
                elif field_key == "description":
                    self._refresh_train_description(value)
                self._refresh_train_automatic_summary()

            callback = (self._on_auto_mode_change
                        if key == "auto_mode" else save_value)
            entry.bind_change(callback)
            if key == "start_byte":
                entry.entry.bind(
                    "<KeyRelease>",
                    lambda _event: setattr(
                        self, "_train_start_byte_override", True
                    ),
                    add="+",
                )
            entry.grid(row=row, column=1, sticky="w", pady=3)
            self.train_fields[key] = entry
            if advanced:
                self.train_advanced_controls.append((field_label, entry))
            if shape_field:
                self.train_shape_fields.append((field_label, entry, key))

        field(shape, "checkpoint name", "save_path", 1, 150)
        field(shape, "description", "description", 2, 150, prominent=True)
        self._label(shape, "preset", font=self.font_dim).grid(
            row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        selected_capacity = state.get("capacity", DEFAULT_CAPACITY)
        if selected_capacity not in CAPACITY_NAMES:
            selected_capacity = DEFAULT_CAPACITY
        self.train_capacity_dd = RoundedDropdown(
            shape, CAPACITY_NAMES, value=selected_capacity,
            on_select=self._on_train_capacity_change,
            width=132, height=29, font=self.font_main)
        self.train_capacity_dd.grid(row=3, column=1, sticky="w", pady=3)
        field(shape, "temporal bands", "bands", 4, shape_field=True)
        field(shape, "trace base", "base", 5, shape_field=True)
        field(shape, "symbol dim", "symbol_dim", 6, shape_field=True)
        field(shape, "scale rank", "scale_rank", 7, shape_field=True)
        field(shape, "scale depth", "scale_depth", 8, shape_field=True)
        field(shape, "summaries", "summaries", 9, shape_field=True)
        field(shape, "model dim", "model_dim", 10, shape_field=True)
        field(shape, "trunk depth", "trunk_depth", 11, shape_field=True)
        field(shape, "expansion dim", "expansion_dim", 12, shape_field=True)

        self._label(feed, "source", font=self.font_dim).grid(
            row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        current_mode = state.get("mode", "corpus")
        if current_mode not in ("corpus", "stream"):
            current_mode = "corpus"
        self.train_mode_seg = RoundedDropdown(
            feed, ["corpus", "stream"], value=current_mode,
            on_select=self._on_train_mode_change,
            width=118, height=29, font=self.font_dim)
        self.train_mode_seg.grid(row=1, column=1, sticky="w", pady=3)

        self._label(feed, "data", font=self.font_dim).grid(
            row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.train_data_picker = tk.Frame(feed, bg=BG)
        self.train_data_picker.grid(
            row=3, column=0, columnspan=2, sticky="w", pady=3)
        data_names = [ALL_DATA_OPTION] + self._list_data_names()
        selected_data = state.get("corpus", "")
        if _parse_cycle_spec(selected_data) is not None:
            selected_data = ALL_DATA_OPTION
        if selected_data not in data_names:
            selected_data = data_names[1] if len(data_names) > 1 else ALL_DATA_OPTION
        self.train_corpus_entry = RoundedDropdown(
            self.train_data_picker, data_names, value=selected_data,
            on_select=self._pick_train_corpus,
            width=148, height=29, font=self.font_dim)
        self.train_open_data_btn = RoundedButton(
            self.train_data_picker, "dir", self._open_data_folder,
            width=42, height=29, font=self.font_dim)
        stream_names = [stream["name"] for stream in self.streams] or [
            "(no streams found)"]
        last_stream = state.get("stream_name") or stream_names[0]
        if last_stream not in stream_names:
            last_stream = stream_names[0]
        self.train_stream_dd = RoundedDropdown(
            self.train_data_picker, stream_names, value=last_stream,
            on_select=lambda value: self._save_form_field(
                "train", "stream_name", value),
            width=190, height=29, font=self.font_dim)
        self.train_head_row = tk.Frame(feed, bg=BG)
        self._label(self.train_head_row, "head mb", font=self.font_dim).pack(
            side="left", padx=(0, 8))
        self.train_head_entry = RoundedEntry(
            self.train_head_row, width=90, height=29,
            value=state.get("head_mb", "50"), font=self.font_dim)
        self.train_head_entry.bind_change(
            lambda value: self._save_form_field("train", "head_mb", value))
        self.train_head_entry.pack(side="left")
        if current_mode == "corpus":
            self.train_corpus_entry.pack(side="left")
            self.train_open_data_btn.pack(side="left", padx=(5, 0))
        else:
            self.train_stream_dd.pack(side="left")
            self.train_head_row.grid(
                row=4, column=0, columnspan=2, sticky="w", pady=3)

        field(feed, "passes", "epochs", 5)
        self._label(feed, "validation", font=self.font_dim).grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=3)
        validation_names = [NO_VALIDATION_OPTION] + self._list_data_names()
        selected_validation = state.get("validation_corpus", "")
        if not selected_validation:
            selected_validation = next(
                (
                    name for name in validation_names[1:]
                    if any(marker in name.lower() for marker in (
                        "validation", "heldout", "held-out", "eval"
                    ))
                ),
                NO_VALIDATION_OPTION,
            )
        if selected_validation not in validation_names:
            selected_validation = NO_VALIDATION_OPTION
        self.train_validation_dd = RoundedDropdown(
            feed, validation_names, value=selected_validation,
            on_select=self._on_train_validation_change,
            width=148, height=29, font=self.font_dim)
        self.train_validation_dd.grid(row=6, column=1, sticky="w", pady=3)
        field(feed, "validate every mb", "validation_every_mb", 7,
              advanced=True)
        field(feed, "validation sample mb", "validation_max_mb", 8,
              advanced=True)
        field(feed, "source byte", "start_byte", 9, advanced=True)

        field(dynamics, "batch predictions", "batch", 1)
        field(dynamics, "learning rate", "lr", 2, advanced=True)
        field(dynamics, "decimation gain", "decimation", 3,
              advanced=True)
        field(dynamics, "gradient clip", "grad_clip", 4, advanced=True)
        field(dynamics, "lattice rate", "lattice_rate", 5,
              advanced=True)
        field(dynamics, "dream every updates", "dream_every_batches", 6,
              advanced=True)
        field(dynamics, "dream max tokens", "dream_length", 7,
              advanced=True)
        field(dynamics, "dream temperature", "dream_temperature", 8,
              advanced=True)
        field(dynamics, "autosave minutes", "autosave_min", 9,
              advanced=True)

        self.train_auto_panel = tk.Frame(dynamics, bg=BG)
        self.train_auto_panel.grid(
            row=10, column=0, columnspan=2, sticky="nsew", pady=(1, 0))
        dynamics.grid_rowconfigure(10, weight=1)
        self.train_auto_summary = tk.Label(
            self.train_auto_panel, text="", font=self.font_dim,
            bg=BG, fg=DIM, justify="left", anchor="nw", wraplength=260)
        self.train_auto_summary.pack(fill="both", expand=True, anchor="nw")

        action_row = tk.Frame(self.train_setup, bg=BG)
        action_row.grid(row=2, column=0, sticky="ew", pady=(12, 8))
        self.train_actions = action_row
        self.train_start_btn = RoundedButton(
            action_row, "begin training", self._train_start_safe,
            width=150, height=36, font=self.font_main)
        self.train_start_btn.pack(side="left")
        self.train_profile_btn = RoundedButton(
            action_row, "edit profile", self._edit_train_profile,
            width=132, height=36, font=self.font_main)
        self.train_profile_btn.pack(side="left", padx=(8, 0))
        self.train_profile_btn.set_enabled(
            last != "(new model)")
        self.train_advanced_btn = RoundedButton(
            action_row, "expert settings", self._toggle_train_advanced,
            width=142, height=36, font=self.font_dim)
        self.train_advanced_btn.pack(side="left", padx=(8, 0))
        self.train_setup_status = self._label(action_row, "")
        self.train_setup_status.pack(side="left", padx=(14, 0))

        info_frame = tk.Frame(self.train_setup, bg=SURFACE)
        info_frame.grid(row=3, column=0, sticky="nsew")
        info_frame.grid_rowconfigure(0, weight=1)
        self.train_setup.grid_rowconfigure(3, weight=1)
        self.train_ckpt_panel = info_frame
        self.train_ckpt_columns = []
        for index in range(3):
            pane = tk.Text(
                info_frame, bg=SURFACE, fg=DIM, font=self.font_main,
                width=1, height=1, wrap="word", bd=0,
                highlightthickness=1,
                highlightbackground=GRID_LINE,
                highlightcolor=GRID_LINE,
                padx=14, pady=12, selectbackground=SELECT_BG)
            pane.config(state="disabled")
            self._bind_text_scroll(pane)
            self.train_ckpt_columns.append(pane)
        self.train_ckpt_info = self.train_ckpt_columns[0]
        info_frame.bind(
            "<Configure>",
            lambda event: self._layout_train_ckpt_panel(event.width))
        self._layout_train_ckpt_panel(900)
        self._set_train_ckpt_panel(
            identity="checkpoint\n\nselect a checkpoint to inspect it.",
            description="description\n\n" + (
                state.get("description") or
                "architecture, history, and spectral diagnostics will "
                "appear here."),
            history="history\n\nnot available",
            energy="weight spectrum\n\nnot available")
        self.train_setup.bind("<Configure>", self._reflow_train_setup)
        self._set_train_advanced(
            state.get("advanced") is True, persist=False)
        self._init_shape_fields(selected_capacity)
        self.root.after_idle(self._refresh_train_shape_preview)

    def _init_shape_fields(self, capacity):
        """Populate shape fields from a capacity preset on first layout."""
        checkpoint = getattr(self, "train_ckpt_dd", None)
        is_new = not checkpoint or checkpoint.value == "(new model)"
        if is_new and capacity in CAPACITY_NAMES:
            try:
                bands = int(self.train_fields["bands"].get())
            except (KeyError, ValueError):
                bands = 48
            values = capacity_values(capacity, bands)
            for key in ("symbol_dim", "scale_rank", "scale_depth",
                        "summaries", "model_dim", "trunk_depth",
                        "expansion_dim"):
                if key in self.train_fields:
                    current = self.train_fields[key].get().strip()
                    if not current:
                        self.train_fields[key].set(str(values[key]))

    def _toggle_train_advanced(self):
        self._set_train_advanced(not self._train_advanced)

    def _set_train_advanced(self, enabled, persist=True):
        """Switch between the managed product contract and expert controls."""
        self._train_advanced = bool(enabled)
        for label, entry in getattr(self, "train_advanced_controls", ()):
            if self._train_advanced:
                label.grid()
                entry.grid()
            else:
                label.grid_remove()
                entry.grid_remove()
        visible_shape = {"bands"}
        for label, entry, key in getattr(self, "train_shape_fields", ()):
            if key in visible_shape:
                continue
            if self._train_advanced:
                label.grid()
                entry.grid()
            else:
                label.grid_remove()
                entry.grid_remove()
        if hasattr(self, "train_auto_panel"):
            if self._train_advanced:
                self.train_auto_panel.grid_remove()
            else:
                self.train_auto_panel.grid()
        if hasattr(self, "train_section_titles"):
            self.train_section_titles[2].config(text="dynamics")
        if hasattr(self, "train_advanced_btn"):
            self.train_advanced_btn.set_text(
                "hide settings" if self._train_advanced
                else "expert settings")
        if persist:
            self._save_form_field("train", "advanced", self._train_advanced)
        self._refresh_train_automatic_summary()

    def _refresh_train_automatic_summary(self):
        if not hasattr(self, "train_auto_summary"):
            return
        checkpoint = getattr(self, "train_ckpt_dd", None)
        existing = bool(
            checkpoint and checkpoint.value != "(new model)")
        try:
            capacity = self.train_capacity_dd.get()
            bands = int(self.train_fields["bands"].get())
            memory_limit = _mps_recommended_memory()
            batch = (
                recommended_mps_batch(capacity, bands, memory_limit)
                if memory_limit > 0
                else default_batch_size(capacity)
            )
        except (KeyError, TypeError, ValueError):
            batch = None
        if existing:
            action = self.train_checkpoint_action_dd.get()
            lineage_line = (
                "new lineage with this revision as parent\n"
                if action == "fork"
                else "continuing this lineage\n"
            )
            batch_line = (
                f"batch {batch:,} predictions fitted to this device\n\n"
                if batch is not None else "saved batch restored\n\n")
            policy = (
                "checkpoint policy\n\n"
                f"{lineage_line}"
                "learning rate, learning density, and clipping are restored "
                "from the checkpoint.\n"
                f"{batch_line}"
            )
        else:
            policy = (
                "learning rate, density, and gradient\n"
                "protection are managed by default.\n\n"
            )
        self.train_auto_summary.config(
            text=(
                policy
                + "expand expert settings to configure\n"
                + "learning dynamics, dreams, and\n"
                + "validation."
            ))

    def _apply_simple_training_contract(self, new_model):
        """Materialise every hidden simple-mode choice before launch."""
        if getattr(self, "_train_advanced", False):
            return
        managed = {
            "autosave_min": "30",
            "dream_every_batches": "64",
            "dream_length": "128",
            "dream_temperature": "1.0",
            "validation_every_mb": "100",
            "validation_max_mb": "0.25",
        }
        capacity = self.train_capacity_dd.get()
        try:
            bands = 48 if new_model else int(self.train_fields["bands"].get())
            memory_limit = _mps_recommended_memory()
            batch = (
                recommended_mps_batch(capacity, bands, memory_limit)
                if memory_limit > 0
                else default_batch_size(capacity)
            )
            managed["batch"] = str(batch)
        except (KeyError, TypeError, ValueError):
            # Custom historical shapes have no measured hardware profile.
            pass
        if new_model:
            bands = 48
            managed.update({
                "bands": str(bands),
                "base": "1.618033988749895",
                "lr": "auto 0.0003",
                "decimation": "1",
                "grad_clip": "1.0",
            })
        for key, value in managed.items():
            self.train_fields[key].set(value)
            self._save_form_field("train", key, value)
        self._refresh_train_shape_preview()
        self._refresh_train_automatic_summary()

    def _refresh_train_shape_preview(self):
        if not hasattr(self, "train_ckpt_columns"):
            return
        try:
            bands = max(1, int(self.train_fields["bands"].get()))
            base = max(1.000001, float(self.train_fields["base"].get()))
            capacity = self.train_capacity_dd.get()
            vocab = VOCAB_SIZE
            values = capacity_values(capacity, bands)
            _shape, width, params = _aperture_shape(bands, capacity)
            lens = base ** (bands - 1)
            trace_mb = vocab * bands * 8 / (1024 * 1024)
            batch = int(self.train_fields["batch"].get())
            memory_limit = _mps_recommended_memory()
            if memory_limit > 0:
                fit, memory_bytes = mps_fit_status(
                    capacity, bands, batch, memory_limit
                )
                auto_batch = recommended_mps_batch(
                    capacity, bands, memory_limit
                )
                fit_note = (
                    f"mps fit {fit}  ·  ~{memory_bytes / 1e9:.1f}gb / "
                    f"{memory_limit / 1e9:.1f}gb recommended\n"
                    f"automatic batch {auto_batch:,} predictions"
                )
            else:
                memory_bytes = estimated_mps_training_bytes(
                    capacity, bands, batch
                )
                fit_note = (
                    f"estimated training memory ~{memory_bytes / 1e9:.1f}gb\n"
                    f"portable automatic batch "
                    f"{default_batch_size(capacity):,} predictions"
                )
            profile = getattr(self, "_train_profile", None)
            if not isinstance(profile, dict):
                checkpoint = self.train_ckpt_dd.value
                path = (
                    CHECKPOINTS_DIR / checkpoint
                    if checkpoint and checkpoint != "(new model)"
                    else CHECKPOINTS_DIR / "model.pt")
                profile = _checkpoint_profile(path)
            shape = (
                f"{profile['name']}\n"
                f"{profile['tagline']}\n\n"
                f"{profile['art']}\n\n"
                "shape\n"
                f"{capacity} capacity  ·  {bands} exact temporal bands\n"
                f"symbol d{values['symbol_dim']}  ·  "
                f"{values['scale_depth']} shared scale blocks  ·  rank {width}\n"
                f"{values['summaries']} temporal summaries  ·  "
                f"model d{values['model_dim']}\n"
                f"{values['trunk_depth']} residual swiglu blocks  ·  "
                f"expansion {values['expansion_dim']}\n"
                f"{params / 1e6:.1f}m parameters\n"
                f"trace horizon ~{lens:,.0f} tokens\n"
                f"fixed trace memory {trace_mb:.1f}mb\n"
                f"{fit_note}")
            if profile.get("about"):
                shape += f"\n\n{profile['about']}"
        except (KeyError, TypeError, ValueError, OverflowError):
            shape = "soma\n\nshape unavailable"
        self._set_readonly_text(self.train_ckpt_columns[1], shape, ACCENT)

    def _layout_train_ckpt_panel(self, width):
        """Keep checkpoint columns useful from compact to wide windows."""
        if not hasattr(self, "train_ckpt_columns"):
            return
        for pane in self.train_ckpt_columns:
            pane.grid_forget()
        panel = self.train_ckpt_panel
        for row in range(3):
            panel.grid_rowconfigure(row, weight=0)
        for column in range(3):
            panel.grid_columnconfigure(column, weight=0)
        if width >= 700:
            panel.grid_rowconfigure(0, weight=1)
            for column, pane in enumerate(self.train_ckpt_columns):
                panel.grid_columnconfigure(column, weight=1, uniform="ckpt")
                pane.grid(row=0, column=column, sticky="nsew")
        elif width >= 480:
            panel.grid_rowconfigure(0, weight=1)
            panel.grid_rowconfigure(1, weight=1)
            panel.grid_columnconfigure(0, weight=1, uniform="ckpt")
            panel.grid_columnconfigure(1, weight=1, uniform="ckpt")
            self.train_ckpt_columns[0].grid(
                row=0, column=0, columnspan=2, sticky="nsew")
            self.train_ckpt_columns[1].grid(
                row=1, column=0, sticky="nsew")
            self.train_ckpt_columns[2].grid(
                row=1, column=1, sticky="nsew")
        else:
            panel.grid_columnconfigure(0, weight=1)
            for row, pane in enumerate(self.train_ckpt_columns):
                panel.grid_rowconfigure(row, weight=1)
                pane.grid(row=row, column=0, sticky="nsew")

    def _reflow_train_setup(self, event):
        """Hide the inspector when there is not enough height to read it."""
        if event.widget is not self.train_setup:
            return
        compact = int(event.height) < 560
        if compact == getattr(self, "_train_setup_compact", None):
            return
        self._train_setup_compact = compact
        if compact:
            self.train_ckpt_panel.grid_remove()
            self.train_setup.grid_rowconfigure(3, weight=0)
        else:
            self.train_setup.grid_rowconfigure(3, weight=1)
            self.train_ckpt_panel.grid(row=3, column=0, sticky="nsew")

    def _set_readonly_text(self, widget, text, color=DIM):
        widget.config(state="normal", fg=color)
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _set_train_ckpt_panel(self, identity="", description="",
                              history="", energy=""):
        if not hasattr(self, "train_ckpt_columns"):
            return
        self._train_ckpt_identity_text = identity
        self._train_ckpt_description_text = description
        left = identity
        if description:
            left = f"{left}\n\n{description}" if left else description
        diagnostics = history
        if energy:
            diagnostics = (f"{diagnostics}\n\n{energy}"
                           if diagnostics else energy)
        self._set_readonly_text(self.train_ckpt_columns[0], left)
        self._set_readonly_text(self.train_ckpt_columns[2], diagnostics)
        self._refresh_train_shape_preview()
        self._refresh_train_automatic_summary()

    def _refresh_train_description(self, description):
        if not hasattr(self, "train_ckpt_columns"):
            return
        identity = getattr(
            self, "_train_ckpt_identity_text", "new checkpoint")
        history = self.train_ckpt_columns[2].get("1.0", "end-1c")
        self._set_train_ckpt_panel(
            identity=identity,
            description=f"description\n\n{description or 'none'}",
            history=history)

    def _train_profile_path(self):
        checkpoint = self.train_ckpt_dd.value
        if checkpoint and checkpoint != "(new model)":
            return CHECKPOINTS_DIR / checkpoint
        save = self.train_fields["save_path"].get().strip() or "model.pt"
        return (
            Path(save).expanduser()
            if "/" in save or "\\" in save
            else CHECKPOINTS_DIR / save)

    def _edit_train_profile(self):
        path = self._train_profile_path()
        if not path.exists():
            self.train_setup_status.config(
                text="save the checkpoint before editing its profile")
            return
        metadata = _read_checkpoint_metadata(path)
        profile = _checkpoint_profile(path, metadata)
        try:
            meta_path = _write_checkpoint_profile(path, profile)
            subprocess.Popen(["open", "-t", str(meta_path)])
            self.train_setup_status.config(
                text=f"editing {meta_path.name} · reselect to refresh")
        except OSError as error:
            self.train_setup_status.config(
                text=f"profile open failed: {error}")

    def _set_train_ckpt_info(self, text):
        if not hasattr(self, "train_ckpt_info"):
            return
        if hasattr(self, "train_ckpt_columns"):
            self._set_train_ckpt_panel(identity=text)
            return
        self._set_readonly_text(self.train_ckpt_info, text)

    def _list_checkpoint_names_with_none(self):
        files = sorted(p.name for p in CHECKPOINTS_DIR.glob("*.pt"))
        return ["(new model)"] + files

    def _on_train_capacity_change(self, capacity):
        self._save_form_field("train", "capacity", capacity)
        fields = getattr(self, "train_fields", {})
        bands = 48
        try:
            bands = int(fields["bands"].get()) if "bands" in fields else 48
        except (TypeError, ValueError):
            pass
        shape_values = capacity_values(capacity, bands)
        shape_map = {
            "bands": str(bands),
            "symbol_dim": str(shape_values["symbol_dim"]),
            "scale_rank": str(shape_values["scale_rank"]),
            "scale_depth": str(shape_values["scale_depth"]),
            "summaries": str(shape_values["summaries"]),
            "model_dim": str(shape_values["model_dim"]),
            "trunk_depth": str(shape_values["trunk_depth"]),
            "expansion_dim": str(shape_values["expansion_dim"]),
        }
        for key, value in shape_map.items():
            if key in fields:
                fields[key].set(value)
                self._save_form_field("train", key, value)
        try:
            memory_limit = _mps_recommended_memory()
            batch = (
                recommended_mps_batch(capacity, bands, memory_limit)
                if memory_limit > 0
                else default_batch_size(capacity)
            )
        except (TypeError, ValueError):
            batch = None
        dynamics_defaults = {
            "lr": "auto 0.0003",
            "decimation": "1",
            "grad_clip": "1.0",
            "autosave_min": "30",
            "dream_every_batches": "64",
            "dream_length": "128",
            "dream_temperature": "1.0",
            "validation_every_mb": "100",
            "validation_max_mb": "0.25",
        }
        if batch is not None:
            dynamics_defaults["batch"] = str(batch)
        for key, value in dynamics_defaults.items():
            if key in fields:
                fields[key].set(value)
                self._save_form_field("train", key, value)
        self._refresh_train_shape_preview()
        self._refresh_train_automatic_summary()

    def _on_train_validation_change(self, value):
        self._save_form_field(
            "train", "validation_corpus",
            "" if value == NO_VALIDATION_OPTION else value)
        self._refresh_train_automatic_summary()

    def _on_train_checkpoint_action(self, action):
        """Make continuation and lineage creation mutually explicit."""
        action = action if action in ("continue", "fork") else "continue"
        self._save_form_field("train", "checkpoint_action", action)
        checkpoint = self.train_ckpt_dd.value
        if checkpoint == "(new model)":
            self.train_checkpoint_action_dd.set_enabled(False)
            self.train_fields["save_path"].set_readonly(False)
            return
        self.train_checkpoint_action_dd.set_enabled(True)
        if action == "continue":
            save_name = checkpoint
            readonly = True
        else:
            source = CHECKPOINTS_DIR / checkpoint
            proposed = source.with_name(f"{source.stem}-fork{source.suffix}")
            save_name = _unused_checkpoint_path(proposed).name
            readonly = False
        self.train_fields["save_path"].set(save_name)
        self.train_fields["save_path"].set_readonly(readonly)
        self._save_form_field("train", "save_path", save_name)
        self._refresh_train_automatic_summary()

    def _on_train_mode_change(self, mode):
        """Switch the data-source widget shown in the train form."""
        self._save_form_field("train", "mode", mode)
        if getattr(self, "_train_setup_v2", False):
            self.train_corpus_entry.pack_forget()
            self.train_open_data_btn.pack_forget()
            self.train_stream_dd.pack_forget()
            self.train_head_row.grid_remove()
            if mode == "corpus":
                self.train_corpus_entry.pack(side="left")
                self.train_open_data_btn.pack(side="left", padx=(5, 0))
                self._load_train_resume_position()
            else:
                self.train_stream_dd.pack(side="left")
                self.train_head_row.grid(
                    row=4, column=0, columnspan=2,
                    sticky="w", pady=3)
            return
        if mode == "corpus":
            self.train_stream_dd.pack_forget()
            self.train_corpus_entry.pack(side="left")
            self.train_choose_data_btn.pack(side="left", padx=(8, 0))
            self.train_open_data_btn.pack(side="left", padx=(8, 0))
            self.train_head_row.grid_remove()
        else:
            self.train_corpus_entry.pack_forget()
            self.train_choose_data_btn.pack_forget()
            self.train_open_data_btn.pack_forget()
            self.train_stream_dd.pack(side="left")
            self.train_head_row.grid(
                row=11, column=1, sticky="w", pady=3)
        self._place_train_actions(mode)

    def _on_auto_mode_change(self, value):
        """retired setup-form hook retained only for state migration."""
        self._save_form_field("train", "auto_mode", DEFAULT_AUTO_MODE)
        if "decimation" in self.train_fields:
            decimation = default_decimation_for_mode(DEFAULT_AUTO_MODE)
            self.train_fields["decimation"].set(decimation)
            self._save_form_field("train", "decimation", decimation)

    def _place_train_actions(self, mode=None):
        if not hasattr(self, "train_actions"):
            return
        if getattr(self, "_train_setup_v2", False):
            return
        self.train_actions.grid_forget()
        if mode is None:
            mode = self.train_mode_seg.get()
        if mode == "stream":
            self.train_actions.grid(
                row=11, column=2, columnspan=2, sticky="w",
                padx=(18, 0), pady=3)
        else:
            self.train_actions.grid(
                row=11, column=0, columnspan=4, sticky="w",
                pady=(5, 0))

    def _list_data_names(self):
        names = []
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for path in sorted(DATA_DIR.iterdir(), key=lambda p: p.name.lower()):
                if _is_trainable_corpus(path):
                    names.append(path.name)
        except OSError:
            pass
        return names

    def _show_train_data_menu(self):
        names = self._list_data_names()
        if not names:
            self.train_setup_status.config(text="no data files found")
            return
        menu = tk.Menu(
            self.train_choose_data_btn, tearoff=0,
            bg=SURFACE, fg=FG, font=self.font_dim,
            activebackground=SURFACE_HOVER, activeforeground=FG,
            bd=0, relief="flat")
        for name in names:
            menu.add_command(
                label=name, command=lambda v=name: self._pick_train_corpus(v))
        x = self.train_choose_data_btn.winfo_rootx()
        y = (self.train_choose_data_btn.winfo_rooty()
             + self.train_choose_data_btn.height + 2)
        menu.tk_popup(x, y)

    def _pick_train_corpus(self, name):
        self.train_corpus_entry.set(name)
        self._save_form_field("train", "corpus", name)
        self._load_train_resume_position()

    def _refresh_train_data_options(self):
        if not hasattr(self, "train_corpus_entry"):
            return
        names = [ALL_DATA_OPTION] + self._list_data_names()
        current = self.train_corpus_entry.get()
        if current not in names:
            current = names[1] if len(names) > 1 else ALL_DATA_OPTION
        self.train_corpus_entry.set_options(names, value=current)
        if hasattr(self, "train_validation_dd"):
            validation_names = [NO_VALIDATION_OPTION] + self._list_data_names()
            validation = self.train_validation_dd.get()
            if validation not in validation_names:
                validation = NO_VALIDATION_OPTION
            self.train_validation_dd.set_options(
                validation_names, value=validation)

    def _on_train_corpus_text_change(self, value):
        self._save_form_field("train", "corpus", value)
        self._load_train_resume_position()

    def _set_train_start_byte(self, value):
        text = str(max(0, int(value)))
        if "start_byte" in getattr(self, "train_fields", {}):
            self.train_fields["start_byte"].set(text)
        self._save_form_field("train", "start_byte", text)
        self.train_setup_status.config(text="")

    def _current_train_checkpoint_key(self):
        if not hasattr(self, "train_ckpt_dd"):
            return self.state.get("train", {}).get("checkpoint", "(new model)")
        value = self.train_ckpt_dd.value or "(new model)"
        save = ""
        if hasattr(self, "train_fields") and "save_path" in self.train_fields:
            save = self.train_fields["save_path"].get().strip()
        if value == "(new model)" and save:
            return Path(save).expanduser().name
        return Path(value).name if value else "(new model)"

    def _train_corpus_resume_key(self, corpus_value):
        text = str(corpus_value or "").strip()
        if "/" in text or "\\" in text:
            return Path(text).name
        return text

    def _train_resume_key(self, corpus_value=None, checkpoint_value=None):
        ckpt = checkpoint_value or self._current_train_checkpoint_key()
        corpus = corpus_value
        if corpus is None:
            if hasattr(self, "train_corpus_entry"):
                corpus = self.train_corpus_entry.get().strip()
            else:
                corpus = self.state.get("train", {}).get("corpus", "")
        return f"{ckpt}::{self._train_corpus_resume_key(corpus)}"

    def _resume_positions(self):
        positions = self.state.setdefault("resume_positions", {})
        if not isinstance(positions, dict):
            positions = {}
            self.state["resume_positions"] = positions
        # v12.3.14 used a temporary `new:` namespace before the first save.
        # The checkpoint then acquired its filename and could no longer find
        # that cursor. Fold those keys into their durable names once.
        for key, value in list(positions.items()):
            if not str(key).startswith("new:"):
                continue
            durable = str(key)[4:]
            positions.setdefault(durable, value)
            positions.pop(key, None)
        return positions

    def _load_train_resume_position(self):
        key = self._train_resume_key()
        pos = int(self._resume_positions().get(key, 0) or 0)
        self._set_train_start_byte(pos)
        self._train_start_byte_override = False

    def _save_train_resume_position(self, pos, total=None):
        if not getattr(self, "_active_train_resume_key", ""):
            return
        value = 0
        if total and pos and pos < total:
            value = int(pos)
        self._resume_positions()[self._active_train_resume_key] = value
        save_state(self.state)
        self._set_train_start_byte(value)
        self._train_start_byte_override = False

    def _commit_train_resume_cursor(self):
        if not self._train_resume_cursor_enabled:
            return
        self._save_train_resume_position(
            self._train_last_absolute_pos,
            self._train_last_absolute_total)

    def _open_data_folder(self):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(DATA_DIR)])
        except Exception as e:
            self.train_setup_status.config(text=f"open failed: {e}")

    def _open_checkpoints_folder(self):
        try:
            CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(CHECKPOINTS_DIR)])
        except Exception as e:
            self.train_setup_status.config(text=f"open failed: {e}")

    def _open_streams_folder(self):
        try:
            STREAMS_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.Popen(["open", str(STREAMS_DIR)])
        except Exception as e:
            if hasattr(self, "stream_setup_status"):
                self.stream_setup_status.config(text=f"open failed: {e}")

    def _on_train_ckpt_change(self, value):
        self._save_form_field("train", "checkpoint", value)

        if value == "(new model)":
            # reset all fields to defaults, unlock everything
            new_path = _unused_checkpoint_path(CHECKPOINTS_DIR / "model.pt")
            _DEFAULTS = {
                "bands": "48", "capacity": DEFAULT_CAPACITY,
                "base": "1.618033988749895",
                "symbol_dim": "", "scale_rank": "", "scale_depth": "",
                "summaries": "", "model_dim": "", "trunk_depth": "",
                "expansion_dim": "",
                "lr": "auto 0.0003",
                "decimation": "1", "lattice_rate": "0.002",
                "grad_clip": "1.0", "batch": "512",
                "update_interval": "1",
                "start_byte": "0", "autosave_min": "30",
                "epochs": "1", "dream_every_batches": "64", "dream_length": "128",
                "dream_temperature": "1.0",
                "validation_every_mb": "100",
                "validation_max_mb": "0.25",
                "save_path": new_path.name,
                "description": "",
            }
            for k, v in _DEFAULTS.items():
                if k in self.train_fields:
                    self.train_fields[k].set(v)
                    self.train_fields[k].set_readonly(False)
                    self._save_form_field("train", k, v)
            self.train_capacity_dd.set(DEFAULT_CAPACITY)
            self.train_capacity_dd.set_enabled(True)
            self.train_checkpoint_action_dd.set("continue")
            self.train_checkpoint_action_dd.set_enabled(False)
            self._save_form_field(
                "train", "checkpoint_action", "continue")
            self._on_train_capacity_change(DEFAULT_CAPACITY)
            self.train_setup_status.config(text="")
            self.train_start_btn.set_enabled(True)
            self._train_profile = _checkpoint_profile(
                CHECKPOINTS_DIR / "model.pt")
            if hasattr(self, "train_profile_btn"):
                self.train_profile_btn.set_enabled(False)
            self._load_train_resume_position()
            if hasattr(self, "train_ckpt_columns"):
                self._set_train_ckpt_panel(
                    identity=(
                        "new checkpoint\n\n"
                        "architecture and training settings are editable."),
                    description="description\n\nnone",
                    history="history\n\nnot saved",
                    energy="weight spectrum\n\navailable after the first save")
            else:
                self._set_train_ckpt_info(
                    "new model\n\n"
                    "architecture and training settings are editable. "
                    "checkpoint id history and weight energy will appear "
                    "here after a saved checkpoint is selected.")
            return

        # auto-fill save_path so user saves back to the same file
        self.train_fields["save_path"].set(value)
        self._save_form_field("train", "save_path", value)
        self.train_checkpoint_action_dd.set_enabled(True)
        self._on_train_checkpoint_action(
            self.train_checkpoint_action_dd.get())
        self._load_train_resume_position()

        # load checkpoint metadata on a worker thread
        path = CHECKPOINTS_DIR / value
        if not path.exists():
            self.train_setup_status.config(text=f"not found: {value}")
            self.train_start_btn.set_enabled(False)
            return
        sidecar = _read_checkpoint_metadata(path)
        self._train_profile = _checkpoint_profile(path, sidecar)
        if hasattr(self, "train_profile_btn"):
            self.train_profile_btn.set_enabled(True)
        self._refresh_train_shape_preview()
        self.train_setup_status.config(text=f"reading {value}...")
        self.train_start_btn.set_enabled(False)
        self.root.update_idletasks()

        def thread():
            sidecar_fields = (
                "n_bands", "token_dim", "scale_rank",
                "scale_depth", "model_dim", "language_depth",
                "base", "batch_size")
            if all(field in sidecar for field in sidecar_fields):
                # Current sidecars contain everything needed by this panel.
                # Avoid loading a multi-hundred-megabyte checkpoint merely
                # because the user selected it in a dropdown.
                cfg = dict(sidecar)
            else:
                try:
                    import torch
                    cfg = torch.load(
                        path, map_location='cpu', weights_only=True)
                except Exception as e:
                    def failed(error=e):
                        self.train_setup_status.config(
                            text=f"failed to read: {error}")
                        self.train_start_btn.set_enabled(False)
                    self.root.after(0, failed)
                    return

            # populate fields from checkpoint — same values the CLI shows
            n_bands = int(cfg.get('n_bands',
                cfg.get('num_timescales', 32)))
            vals = {
                "bands": str(n_bands),
                "base": f"{float(cfg.get('base', 1.618033988749895)):.15g}",
                "symbol_dim": str(cfg.get('symbol_dim',
                    cfg.get('token_dim', ''))),
                "scale_rank": str(cfg.get('scale_rank', '')),
                "scale_depth": str(cfg.get('scale_depth', '')),
                "summaries": str(cfg.get('summaries', '')),
                "model_dim": str(cfg.get('model_dim', '')),
                "trunk_depth": str(cfg.get('trunk_depth',
                    cfg.get('language_depth', ''))),
                "expansion_dim": str(cfg.get('expansion_dim',
                    cfg.get('language_expansion', ''))),
                "lr": (
                    f"auto {float(cfg.get('lr_base', 0.0003)):g}"
                    if cfg.get('lr_auto', False)
                    else str(cfg.get('lr', 0.0003))
                ),
                "decimation": str(cfg.get('decimation_gain', 1.0)),
                "lattice_rate": str(_checkpoint_lattice_rate(cfg)),
                "grad_clip": str(cfg.get('grad_clip', 1.0)),
                "update_interval": str(cfg.get('update_interval', 1)),
                "batch": str(cfg.get('batch_size', 1024)),
                "description": str(cfg.get('description', "")),
            }
            capacity = str(cfg.get("capacity") or infer_capacity(cfg))
            seen = cfg.get('bytes_seen', 0)
            ckpt_id = cfg.get('checkpoint_id', '')
            history = cfg.get('checkpoint_history', [])
            try:
                hist_lines = [h[:16] for h in history[-80:]]
            except TypeError:
                hist_lines = []

            energy_lines = []
            saved_spectrum = cfg.get("weight_spectrum")
            if isinstance(saved_spectrum, list) and saved_spectrum:
                try:
                    peak = max(float(value) for value in saved_spectrum) or 1.0
                    for i, spectrum_value in enumerate(saved_spectrum):
                        energy = float(spectrum_value)
                        bar = "█" * max(1, round((energy / peak) * 18))
                        energy_lines.append(f"{i:02d} {bar} {energy:.4g}")
                except (TypeError, ValueError):
                    energy_lines = []
            first_weight = (cfg.get("net") or {}).get("layers.0.weight")
            if not energy_lines and first_weight is not None:
                try:
                    import torch
                    n_bands = int(vals["bands"])
                    first_cutoff = int(
                        (cfg.get("aperture_cutoffs") or [n_bands])[0])
                    col_energy = first_weight.float().pow(2).mean(dim=0)
                    band_energy = torch.stack([
                        col_energy[i::first_cutoff].mean()
                        for i in range(first_cutoff)
                    ])
                    max_energy = float(band_energy.max().item()) or 1.0
                    for i, e in enumerate(band_energy.tolist()):
                        bar = "█" * max(1, round((e / max_energy) * 18))
                        energy_lines.append(f"{i:02d} {bar} {e:.4g}")
                except Exception as e:
                    energy_lines = [f"weight energy unavailable: {e}"]

            # architecture fields are read-only when resuming
            immutable = ("bands", "base", "symbol_dim", "scale_rank",
                         "scale_depth", "summaries", "model_dim",
                         "trunk_depth", "expansion_dim")

            def apply():
                for k, v in vals.items():
                    if k in self.train_fields:
                        self.train_fields[k].set(v)
                        self.train_fields[k].set_readonly(k in immutable)
                self.train_capacity_dd.set(capacity)
                self.train_capacity_dd.set_enabled(False)
                self._on_train_checkpoint_action(
                    self.train_checkpoint_action_dd.get())
                self._save_form_field("train", "capacity", capacity)
                self._refresh_train_shape_preview()
                seen_str = f"{_fmt_bytes(seen)} seen"
                self.train_setup_status.config(
                    text=f"loaded {value} · {seen_str}")
                self.train_start_btn.set_enabled(True)
                identity = "\n".join([
                    "checkpoint",
                    "",
                    value,
                    f"id {ckpt_id[:24] if ckpt_id else 'not saved'}",
                    f"experience {_fmt_bytes(seen)}",
                    "",
                    "training",
                    "",
                    f"lr {vals['lr']}",
                    f"batch {vals['batch']}",
                    f"decimation gain {vals['decimation']}",
                    f"gradclip {vals['grad_clip']}",
                    "controller io2",
                ])
                description = (
                    "description\n\n" + (vals["description"] or "none"))
                history_text = "\n".join([
                    "weight spectrum",
                    "",
                    *(energy_lines or ["not available"]),
                    "",
                    f"id history ({len(history)})",
                    "",
                    *(hist_lines or ["none"]),
                ])
                if hasattr(self, "train_ckpt_columns"):
                    self._train_profile = _checkpoint_profile(path, sidecar)
                    self._set_train_ckpt_panel(
                        identity=identity,
                        description=description,
                        history=history_text)
                else:
                    self._set_train_ckpt_info(
                        "\n\n".join((identity, description, history_text)))

            self.root.after(0, apply)

        threading.Thread(target=thread, daemon=True).start()

    def _build_train_live(self):
        """The live readout view shown during training.

        Left-aligned layout with a model/corpus summary at top,
        headline loss, stats row, sparkline, and stop button."""
        self.train_live = tk.Frame(self.train_card, bg=BG)
        self.train_live.grid(row=0, column=0, sticky="nsew")
        self.train_live.grid_columnconfigure(0, weight=1)
        self.train_live.grid_rowconfigure(0, weight=0)
        self.train_live.grid_rowconfigure(1, weight=0)

        content = tk.Frame(self.train_live, bg=BG)
        content.grid(row=0, column=0, sticky="new", padx=8, pady=8)
        content.grid_columnconfigure(0, weight=1)

        # ── summary block (populated when training starts) ──
        self.train_summary = tk.Label(
            content, text="", font=self.font_dim, bg=BG, fg=DIM,
            anchor="w", justify="left")
        self.train_summary.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        content.bind(
            "<Configure>",
            lambda event: self.train_summary.config(
                wraplength=max(280, event.width - 8)))

        # ── headline numbers ──
        head = tk.Frame(content, bg=BG)
        head.grid(row=1, column=0, sticky="w", pady=(0, 5))

        self.train_loss = tk.Label(
            head, text="—", font=self.font_huge, bg=BG, fg=FG,
            anchor="w")
        self.train_loss.pack(side="left")

        self.train_pct = tk.Label(
            head, text="", font=self.font_med, bg=BG, fg=AMBER,
            anchor="w")
        self.train_pct.pack(side="left", padx=(16, 0))

        # ── responsive instruments ──
        self.primary_stats = ResponsiveTelemetry(
            content,
            (("acc", "next-token exact"), ("top5", "next-token top 5"),
             ("bps", "source bytes / s")),
            self.mono, min_cell=150, value_font=self.font_med)
        self.primary_stats.grid(row=2, column=0, sticky="ew", pady=(0, 3))
        self.cap_acc = self.primary_stats.values["acc"]
        self.cap_top5 = self.primary_stats.values["top5"]
        self.cap_bps = self.primary_stats.values["bps"]

        self.train_safety = ResponsiveTelemetry(
            content,
            (("health", "run health"),
             ("save", "last safe checkpoint"),
             ("validation", "dense held-out"),
             ("remaining", "current pass remaining"),
             ("storage", "storage free")),
            self.mono, min_cell=170, value_font=self.font_dim,
            value_fg=FG)
        self.train_safety.grid(row=3, column=0, sticky="ew", pady=(3, 0))

        self.train_telemetry = ResponsiveTelemetry(
            content,
            (("seen", "this run / time"), ("lr", "learning rate"),
             ("control", "learning drive"),
             ("stride", "prediction spacing"),
             ("pulse", "update opportunity / cost"),
             ("duty", "updates taken"),
             ("updates", "optimizer updates / next fill"),
             ("pressure", "row pressure / spread"),
             ("memory", "mps live / held"),
             ("error", "prediction error energy"),
             ("epoch", "pass")),
            self.mono, min_cell=140, value_font=self.font_dim,
            value_fg=ACCENT)
        self.train_telemetry.grid(row=4, column=0, sticky="ew", pady=(2, 0))
        self.cap_seen = self.train_telemetry.values["seen"]

        # ── outcome + spectral state ──
        plots = tk.Frame(content, bg=BG)
        plots.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        plots.grid_columnconfigure(0, weight=1, uniform="liveplot")
        plots.grid_columnconfigure(1, weight=1, uniform="liveplot")

        loss_plot = tk.Frame(plots, bg=BG)
        loss_plot.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        loss_plot.grid_columnconfigure(0, weight=1)
        self.train_loss_plot = loss_plot
        self.train_loss_plot_label = self._label(
            loss_plot, "estimated bits per source byte · lower is better",
            fg=DIM
        )
        self.train_loss_plot_label.grid(
            row=0, column=0, sticky="w", pady=(0, 3))
        self.train_spark = Sparkline(
            loss_plot, width=360, height=64, capacity=300)
        self.train_spark.grid(row=1, column=0, sticky="ew")

        self.train_spectrum = SpectrumPlot(
            plots, self.mono, height=82)
        self.train_spectrum.grid(
            row=0, column=1, sticky="nsew", padx=(10, 0))

        lexical = tk.Frame(plots, bg=BG)
        self.train_lexical = lexical
        lexical.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        lexical.grid_columnconfigure(0, weight=1, uniform="lexical")
        lexical.grid_columnconfigure(1, weight=1, uniform="lexical")
        self.active_memory_label = tk.Label(
            lexical, text="slow trace deviations  —", font=self.font_micro,
            bg=BG, fg=DIM, anchor="w", justify="left")
        self.active_memory_label.grid(
            row=0, column=0, sticky="ew", padx=(0, 10))
        self.top_predictions_label = tk.Label(
            lexical, text="next  —", font=self.font_micro,
            bg=BG, fg=ACCENT, anchor="w", justify="left")
        self.top_predictions_label.grid(
            row=0, column=1, sticky="ew", padx=(10, 0))

        # ── status + stop ──
        bottom = tk.Frame(content, bg=BG)
        bottom.grid(row=6, column=0, sticky="w", pady=(10, 0))

        self.train_stop_btn = RoundedButton(
            bottom, "stop", self._train_stop,
            width=120, height=36, font=self.font_main)
        self.train_stop_btn.pack(side="left")

        self.train_live_detail_btn = RoundedButton(
            bottom, "show diagnostics", self._toggle_train_live_detail,
            width=156, height=36, font=self.font_dim)
        self.train_live_detail_btn.pack(side="left", padx=(8, 0))

        self.train_run_data_btn = RoundedButton(
            bottom, "run data", self._open_train_run_data,
            width=108, height=36, font=self.font_dim)
        self.train_run_data_btn.pack(side="left", padx=(8, 0))

        self.train_live_status = self._label(bottom, "", fg=DIM)
        self.train_live_status.pack(side="left", padx=(16, 0))

        self.train_log_label = self._label(
            self.train_live, "training log", font=self.font_main, fg=FG)
        self.train_log_label.grid(
            row=1, column=0, sticky="sw", padx=8, pady=(12, 4))

        self.train_log_frame = tk.Frame(self.train_live, bg=SURFACE)
        self.train_log_frame.grid(
            row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.train_log_frame.grid_rowconfigure(0, weight=1)
        self.train_log_frame.grid_columnconfigure(0, weight=1)
        self.train_live.grid_rowconfigure(2, weight=1)

        self.train_log_text = tk.Text(
            self.train_log_frame, bg=SURFACE, fg=DIM, font=self.font_dim,
            height=3, wrap="word", bd=0, highlightthickness=0,
            padx=14, pady=10, selectbackground=SELECT_BG)
        self.train_log_text.grid(row=0, column=0, sticky="nsew")
        self._bind_text_scroll(self.train_log_text)
        saved_log = self._load_text_file(TRAIN_LOG_FILE)
        if saved_log:
            self.train_log_text.insert("1.0", saved_log)
            self.train_log_text.see("end")
        self.train_log_text.config(state="disabled")
        self.train_live.bind("<Configure>", self._reflow_train_live)
        self._set_train_live_detailed(False)

    def _toggle_train_live_detail(self):
        self._set_train_live_detailed(not self._train_live_detailed)

    def _open_train_run_data(self):
        path = self.train_worker.metrics_path
        try:
            if path and path.exists():
                subprocess.Popen(["open", "-R", str(path)])
            else:
                RUNS_DIR.mkdir(parents=True, exist_ok=True)
                subprocess.Popen(["open", str(RUNS_DIR)])
        except OSError as error:
            self.train_live_status.config(text=f"run data open failed: {error}")

    def _set_train_live_detailed(self, enabled):
        self._train_live_detailed = bool(enabled)
        if self._train_live_detailed:
            self.train_telemetry.set_visible()
            self.train_spectrum.grid(
                row=0, column=1, sticky="nsew", padx=(10, 0))
            self.train_loss_plot.grid_configure(
                columnspan=1, padx=(0, 10))
            self.active_memory_label.grid(
                row=0, column=0, sticky="ew", padx=(0, 10))
            self.top_predictions_label.grid_configure(
                row=0, column=1, columnspan=1, padx=(10, 0))
            self.train_live_detail_btn.set_text("hide diagnostics")
        else:
            self.train_telemetry.set_visible(("seen", "epoch"))
            self.train_spectrum.grid_remove()
            self.train_loss_plot.grid_configure(
                columnspan=2, padx=(0, 0))
            self.active_memory_label.grid_remove()
            self.top_predictions_label.grid_configure(
                row=0, column=0, columnspan=2, padx=(0, 0))
            self.train_live_detail_btn.set_text("show diagnostics")

    def _refresh_train_run_safety(self, force=False):
        if not hasattr(self, "train_safety"):
            return
        now = time.time()
        if not force and now - self._train_safety_refresh_at < 1.0:
            return
        self._train_safety_refresh_at = now

        if self._train_state in ("training", "stopping"):
            report_age = (
                now - self._train_last_report_at
                if self._train_last_report_at else None
            )
            if self._train_validating:
                health = "dense validation"
                health_color = AMBER
            elif self._train_saving:
                health = "publishing checkpoint"
                health_color = AMBER
            elif self._train_state == "stopping":
                health = "stopping safely"
                health_color = AMBER
            elif report_age is None:
                health = "starting"
                health_color = AMBER
            elif report_age < 5:
                health = "live"
                health_color = ACCENT
            elif report_age < 30:
                health = f"busy · {_fmt_duration(report_age)} since report"
                health_color = AMBER
            else:
                health = f"stalled · {_fmt_duration(report_age)}"
                health_color = AMBER
        else:
            health = "idle"
            health_color = DIM
        self.train_safety.set_value("health", health)
        self.train_safety.values["health"].config(fg=health_color)

        if self._train_last_save_at:
            save_age = max(0.0, now - self._train_last_save_at)
            save_text = (
                "just now" if save_age < 5
                else f"{_fmt_duration(save_age)} ago"
            )
        else:
            save_text = "creating first checkpoint"
        self.train_safety.set_value("save", save_text)

        if self._train_last_validation_at:
            validation_age = max(0.0, now - self._train_last_validation_at)
            validation_text = (
                f"{self._train_validation_text} · "
                + ("just now" if validation_age < 5
                   else f"{_fmt_duration(validation_age)} ago")
            )
        else:
            validation_text = self._train_validation_text
        self.train_safety.set_value("validation", validation_text)

        if self._train_continuous:
            remaining = "continuous"
        elif self._train_report_total > 0 and self._train_report_bps > 0:
            left = max(0, self._train_report_total - self._train_report_pos)
            remaining = _fmt_duration(left / self._train_report_bps)
        else:
            remaining = "calculating"
        self.train_safety.set_value("remaining", remaining)

        try:
            destination = Path(self._train_save_path).expanduser()
            folder = destination.parent if destination.name else CHECKPOINTS_DIR
            free = shutil.disk_usage(folder).free
            self.train_safety.set_value("storage", _fmt_bytes(free))
            self.train_safety.values["storage"].config(
                fg=AMBER if free < 2_000_000_000 else FG)
        except OSError:
            self.train_safety.set_value("storage", "unavailable")

    def _reflow_train_live(self, event):
        """Keep compact windows intentional rather than partially clipped."""
        compact = event.height < 430
        if compact:
            self.train_log_label.grid_remove()
            self.train_log_frame.grid_remove()
        else:
            self.train_log_label.grid()
            self.train_log_frame.grid()

    def _train_log(self, message):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} · {message}\n"
        self._append_text_file(
            TRAIN_LOG_FILE, line, max_bytes=MAX_TRAIN_LOG_BYTES)
        if hasattr(self, "train_log_text"):
            self.train_log_text.config(state="normal")
            self.train_log_text.insert("end", line)
            self._cap_text_widget(
                self.train_log_text, max_lines=2200, keep_lines=2000,
                max_chars=400_000)
            self.train_log_text.see("end")
            self.train_log_text.config(state="disabled")

    @staticmethod
    def _display_token(value):
        text = str(value).replace("\n", "\\n").replace("\t", "\\t")
        if not text:
            return "<empty>"
        return text if len(text) <= 18 else text[:15] + "..."

    def _train_status(self, message):
        self.train_setup_status.config(text=message)
        try:
            self._train_log(message)
        except Exception:
            pass

    def _set_train_state(self, state):
        self._train_state = state
        if state == 'idle':
            self.train_live.grid_remove()
            self.train_setup.grid()
            self._refresh_train_data_options()
            self.train_setup_status.config(text="")
            self.train_start_btn.set_text("begin training")
            self.train_start_btn.set_enabled(True)
        elif state == 'training':
            self.train_setup.grid_remove()
            self.train_live.grid()
            self.train_stop_btn.set_text("stop")
            self.train_stop_btn.set_enabled(True)
        elif state == 'stopping':
            self.train_stop_btn.set_text("stopping…")
            self.train_stop_btn.set_enabled(False)
        elif state == 'stopped':
            self.train_live.grid_remove()
            self.train_setup.grid()
            self.train_setup.tkraise()
            self._refresh_train_data_options()
            self.train_start_btn.set_text("resume training")
            self.train_start_btn.set_enabled(True)
            # refresh checkpoint list — saved file may be new
            names = self._list_checkpoint_names_with_none()
            cur = self.train_ckpt_dd.value
            if cur not in names:
                cur = names[0]
            self.train_ckpt_dd.set_options(names, value=cur)
            self._refresh_chat_ckpts()
            self._request_native_repaint()

    def _train_start_safe(self):
        """surface setup failures that a frozen gui would otherwise hide."""
        try:
            self._train_status("checking training setup...")
            self._train_start()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            self._train_status(f"start failed · {message}")
            messagebox.showerror(
                "training could not start",
                message,
                parent=self.root,
            )

    def _train_start(self):
        mode = self.train_mode_seg.get()

        # resolve corpus path based on mode
        if mode == "corpus":
            corpus = self.train_corpus_entry.get().strip()
            if not corpus:
                self.train_setup_status.config(
                    text="enter a corpus filename")
                return
            try:
                epochs = max(1, int(float(
                    self.train_fields["epochs"].get())))
            except (ValueError, TypeError, KeyError):
                epochs = 1
            cycle_count = (
                epochs if corpus == ALL_DATA_OPTION
                else _parse_cycle_spec(corpus))
            cycle_files = []
            if cycle_count is not None:
                cycle_files = _data_cycle_files()
                if not cycle_files:
                    self.train_setup_status.config(
                        text=f"no corpus files found in {DATA_DIR}")
                    return
                corpus_path = DATA_DIR
            else:
                if not ('/' in corpus or '\\' in corpus
                        or corpus.startswith('~') or corpus.startswith('.')):
                    corpus_path = DATA_DIR / corpus
                else:
                    corpus_path = Path(os.path.expanduser(corpus))
                if not corpus_path.exists():
                    self.train_setup_status.config(
                        text=f"not found: {corpus_path}")
                    return
            loop_mode = False
        else:  # mode == "stream"
            stream_name = self.train_stream_dd.value
            if not stream_name or stream_name.startswith("("):
                self.train_setup_status.config(
                    text="no stream selected")
                return
            corpus_path = STREAMS_DATA_DIR / f"{stream_name}.txt"
            if not corpus_path.exists():
                self.train_setup_status.config(
                    text=f"{stream_name} hasn't written anything yet — "
                         f"start the stream first")
                return
            loop_mode = True
            try:
                head_mb = max(1, int(float(self.train_head_entry.get())))
            except (ValueError, TypeError):
                head_mb = 50
            self._save_form_field("train", "head_mb", str(head_mb))
            cycle_count = None
            cycle_files = []
            epochs = 1

        validation_value = self.train_validation_dd.get().strip()
        validation_path = ""
        if validation_value and validation_value != NO_VALIDATION_OPTION:
            candidate = Path(os.path.expanduser(validation_value))
            if not candidate.is_absolute() and not (
                    "/" in validation_value or "\\" in validation_value):
                candidate = DATA_DIR / validation_value
            if not candidate.is_file():
                self.train_setup_status.config(
                    text=f"validation file not found: {candidate}")
                return
            candidate = candidate.resolve()
            if mode == "corpus" and cycle_count is None:
                if Path(corpus_path).resolve() == candidate:
                    self.train_setup_status.config(
                        text="validation must be separate from training data")
                    return
            if cycle_files:
                cycle_files = [
                    path for path in cycle_files
                    if path.resolve() != candidate
                ]
                if not cycle_files:
                    self.train_setup_status.config(
                        text="validation leaves no files for training")
                    return
            validation_path = str(candidate)

        # resolve checkpoint (resume) path
        ckpt_name = self.train_ckpt_dd.value
        if ckpt_name == "(new model)":
            ckpt_path = ""
        else:
            ckpt_path = str(CHECKPOINTS_DIR / ckpt_name)
        fork_lineage = bool(
            ckpt_path
            and self.train_checkpoint_action_dd.get() == "fork"
        )
        fresh_model = not bool(ckpt_path)
        started_new_model = fresh_model or fork_lineage
        # resolve save path from its field
        save = self.train_fields["save_path"].get().strip() or "model.pt"
        if not ('/' in save or '\\' in save
                or save.startswith('~') or save.startswith('.')):
            save_path = str(CHECKPOINTS_DIR / save)
        else:
            save_path = os.path.expanduser(save)
        if ckpt_path and not fork_lineage:
            save_path = ckpt_path
            save = Path(ckpt_path).name
            self.train_fields["save_path"].set(save)
        requested_save_path = Path(save_path)
        selected_path = Path(ckpt_path) if ckpt_path else None
        same_lineage = bool(
            not fork_lineage
            and selected_path
            and requested_save_path.resolve() == selected_path.resolve()
        )
        if requested_save_path.exists() and not same_lineage:
            requested_save_path = _unused_checkpoint_path(
                requested_save_path)
            save_path = str(requested_save_path)
            display_save = (
                requested_save_path.name
                if requested_save_path.parent.resolve()
                == CHECKPOINTS_DIR.resolve()
                else str(requested_save_path)
            )
            self.train_fields["save_path"].set(display_save)
            save = display_save
            self.train_setup_status.config(
                text=f"using {requested_save_path.name} to protect the "
                     "existing checkpoint")
        # persist save_path so it survives restarts
        self._save_form_field("train", "save_path", save)

        # read all model params directly from form fields
        def _get(key, default=""):
            if key in self.train_fields:
                return self.train_fields[key].get().strip() or default
            return default

        for key in ("bands", "base", "lr", "decimation", "lattice_rate",
                    "grad_clip",
                    "batch", "epochs",
                    "start_byte", "autosave_min", "dream_every_batches",
                    "dream_length", "dream_temperature", "description",
                    "validation_every_mb", "validation_max_mb",
                    "symbol_dim", "scale_rank", "scale_depth",
                    "summaries", "model_dim", "trunk_depth",
                    "expansion_dim"):
            self._save_form_field("train", key, _get(key, ""))
        capacity = self.train_capacity_dd.get()
        self._save_form_field("train", "capacity", capacity)

        if validation_path:
            try:
                validation_every_mb = float(
                    _get("validation_every_mb", "100"))
                validation_max_mb = float(
                    _get("validation_max_mb", "0.25"))
                if validation_every_mb <= 0 or validation_max_mb <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                self.train_setup_status.config(
                    text="validation cadence and sample size must be positive")
                return

        config = {
            "corpus": str(corpus_path),
            "corpus_label": (
                f"cycle data ×{cycle_count} ({len(cycle_files)} files)"
                if cycle_count is not None else Path(corpus_path).name),
            "cycle_files": [str(p) for p in cycle_files],
            "cycle_count": cycle_count or 1,
            "checkpoint": ckpt_path,
            "fork": fork_lineage,
            "bands": _get("bands", "48"),
            "capacity": capacity,
            "base": _get("base", "1.618033988749895"),
            "symbol_dim": _get("symbol_dim", ""),
            "scale_rank": _get("scale_rank", ""),
            "scale_depth": _get("scale_depth", ""),
            "summaries": _get("summaries", ""),
            "model_dim": _get("model_dim", ""),
            "trunk_depth": _get("trunk_depth", ""),
            "expansion_dim": _get("expansion_dim", ""),
            "lr": _get("lr", "0.0003"),
            "decimation": _get("decimation", "1"),
            "lattice_rate": _get("lattice_rate", "0.002"),
            "grad_clip": _get("grad_clip", "1.0"),
            "description": _get("description", ""),
            "batch": _get("batch", "1024"),
            "update_interval": "1",
            "start_byte": _get("start_byte", "0"),
            "start_byte_override": bool(
                self._train_start_byte_override
            ),
            "autosave_min": _get("autosave_min", "30"),
            "epochs": _get("epochs", "1"),
            "dream_every_batches": _get("dream_every_batches", "0"),
            "dream_length": _get("dream_length", "128"),
            "dream_temperature": _get("dream_temperature", "1.0"),
            "validation_path": validation_path,
            "validation_every_mb": _get("validation_every_mb", "100"),
            "validation_max_mb": _get("validation_max_mb", "0.25"),
            "save_path": save_path,
            "loop": loop_mode,
            "head_bytes": (head_mb if loop_mode else 50) * 1_000_000,
            "cycle_pause": 60.0,
        }
        self._train_resume_cursor_enabled = (
            mode == "corpus" and cycle_count is None
            and epochs == 1 and not loop_mode)
        self._active_train_resume_key = (
            self._train_resume_key(corpus_path)
            if self._train_resume_cursor_enabled else "")
        self._train_last_absolute_pos = int(float(config["start_byte"] or 0))
        try:
            self._train_last_absolute_total = Path(corpus_path).stat().st_size
        except OSError:
            self._train_last_absolute_total = 0
        if (self._train_resume_cursor_enabled
                and self._train_last_absolute_total > 0
                and self._train_last_absolute_pos >= self._train_last_absolute_total):
            self._train_last_absolute_pos = 0
            config["start_byte"] = "0"
            self._set_train_start_byte(0)
            self._save_train_resume_position(0, 0)

        if config["capacity"] in CAPACITY_NAMES:
            try:
                bands = int(config["bands"])
            except (TypeError, ValueError):
                self.train_setup_status.config(
                    text="bands must be a number")
                return
            memory_limit = _mps_recommended_memory()
            if memory_limit > 0:
                try:
                    fit, memory_bytes = mps_fit_status(
                        config["capacity"],
                        bands,
                        int(config["batch"]),
                        memory_limit,
                    )
                except (TypeError, ValueError):
                    self.train_setup_status.config(
                        text="batch must be a supported positive integer"
                    )
                    return
            else:
                fit, memory_bytes = "comfortable", 0
            if fit == "unsupported":
                auto_batch = recommended_mps_batch(
                    config["capacity"], bands, memory_limit
                )
                self.train_setup_status.config(
                    text=(
                        f"this shape needs about {memory_bytes / 1e9:.1f}gb "
                        f"against a {memory_limit / 1e9:.1f}gb mps envelope; "
                        f"use batch {auto_batch} or smaller"
                    ))
                return

        try:
            destination = Path(save_path).expanduser()
            destination.parent.mkdir(parents=True, exist_ok=True)
            free_storage = shutil.disk_usage(destination.parent).free
            if destination.exists():
                checkpoint_estimate = int(destination.stat().st_size * 1.25)
            elif config["capacity"] in CAPACITY_NAMES:
                checkpoint_estimate = int(
                    parameter_count(
                        config["capacity"], int(config["bands"])
                    ) * 16
                )
            else:
                checkpoint_estimate = 2_000_000_000
            required_storage = max(2_000_000_000, checkpoint_estimate)
            if free_storage < required_storage:
                self.train_setup_status.config(
                    text=(
                        f"training needs at least "
                        f"{_fmt_bytes(required_storage)} free for atomic "
                        f"checkpoints; {_fmt_bytes(free_storage)} available"
                    ))
                return
        except OSError as error:
            self.train_setup_status.config(
                text=f"checkpoint storage is unavailable: {error}")
            return

        # reset live view
        self.train_spark.reset()
        self.train_summary.config(text="")
        self.train_pct.config(text="")
        self.train_loss.config(text="—")
        for w in (self.cap_acc, self.cap_top5, self.cap_bps):
            w.config(text="—")
        self.primary_stats.set_label("acc", "next-token exact")
        self.primary_stats.set_label("top5", "next-token top 5")
        self.train_telemetry.reset()
        self.train_safety.reset()
        self.train_spectrum.reset()
        self.active_memory_label.config(text="slow trace deviations  —")
        self.top_predictions_label.config(text="next  —")
        self._set_train_live_detailed(self._train_advanced)
        self._train_last_report_at = 0.0
        self._train_last_save_at = (
            destination.stat().st_mtime if destination.exists() else 0.0)
        self._train_save_path = str(destination)
        self._train_report_pos = 0
        self._train_report_total = 0
        self._train_report_bps = 0.0
        self._train_continuous = bool(loop_mode)
        self._train_saving = False
        self._train_validating = False
        self._train_last_validation_at = 0.0
        self._train_validation_text = (
            "waiting" if validation_path else "off")
        self._refresh_train_run_safety(force=True)
        self.train_live_status.config(text="starting...")

        self._train_log(
            f"start requested · checkpoint "
            f"{Path(ckpt_path).name if ckpt_path else '(new model)'} · "
            f"save {Path(save_path).name} · "
            f"start {config['start_byte']}")

        if self.train_worker.start(config):
            self._train_started_new_model = started_new_model
            self._train_last_saved_checkpoint = ""
            self._set_train_state('training')
            self._set_global_status("training", active=True)
            self._set_active_checkpoint(Path(save_path).stem, "training")
            self._train_log(
                f"started {Path(config['save_path']).name} "
                f"on {Path(config['corpus']).name}")
        else:
            self._train_status("training is already running")

    def _train_stop(self):
        if not self.train_worker.is_running():
            return
        choice = messagebox.askyesnocancel(
            "stop training",
            "save checkpoint before stopping?\n\n"
            "yes saves the latest state. no stops without a final save.")
        if choice is None:
            return
        self.train_worker.stop(save=choice)
        self._set_train_state('stopping')
        if choice:
            self.train_live_status.config(text="stopping and saving...")
        else:
            self.train_live_status.config(text="stopping without saving...")

    def _select_train_checkpoint_after_new_model_save(self):
        if not self._train_started_new_model:
            return
        saved = self._train_last_saved_checkpoint
        self._train_started_new_model = False
        self._train_last_saved_checkpoint = ""
        if not saved:
            return
        path = Path(saved)
        if not path.is_absolute():
            path = CHECKPOINTS_DIR / path.name
        if not path.exists():
            return
        name = path.name
        names = self._list_checkpoint_names_with_none()
        self.train_ckpt_dd.set_options(names, value=name)
        self._on_train_ckpt_change(name)

    # ── stream view ──
    #
    # staged flow:
    #   idle      → pick a stream + corpus, big start button
    #   running   → output log, stop button

    def _build_stream(self):
        f = self.frames["stream"]
        f.grid_rowconfigure(0, weight=1)
        f.grid_columnconfigure(0, weight=1)

        self.stream_card = tk.Frame(f, bg=BG)
        self.stream_card.grid(row=0, column=0, sticky="nsew",
                              padx=20, pady=20)
        self.stream_card.grid_rowconfigure(0, weight=1)
        self.stream_card.grid_columnconfigure(0, weight=1)

        self._build_stream_setup()
        self._build_stream_live()

        self._set_stream_state('idle')

    def _build_stream_setup(self):
        self.stream_setup = tk.Frame(self.stream_card, bg=BG)
        self.stream_setup.grid(row=0, column=0, sticky="nsew")
        self.stream_setup.grid_columnconfigure(0, weight=1)

        names = [s["name"] for s in self.streams] or ["(no streams found)"]
        last = self.state["stream"].get("stream_name") or names[0]
        if last not in names:
            last = names[0]

        # source picker
        src = tk.Frame(self.stream_setup, bg=BG)
        src.grid(row=0, column=0, sticky="w", pady=(0, 16))
        self._label(src, "stream", font=self.font_main, fg=FG
                    ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.stream_dd = RoundedDropdown(
            src, names, value=last,
            on_select=self._on_stream_change,
            width=260, height=32, font=self.font_main)
        self.stream_dd.grid(row=1, column=0, sticky="w")
        self.stream_desc = self._label(src, "")
        self.stream_desc.grid(row=2, column=0, sticky="w", pady=(6, 0))
        self._update_stream_desc()

        # destination note (read-only — streams write to predetermined paths)
        dest = tk.Frame(self.stream_setup, bg=BG)
        dest.grid(row=1, column=0, sticky="w", pady=(0, 16))
        self._label(dest, "writes to", font=self.font_main, fg=FG
                    ).grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.stream_dest_label = self._label(dest, "", fg=ACCENT,
                                             font=self.font_main)
        self.stream_dest_label.grid(row=1, column=0, sticky="w")
        self._update_stream_dest()
        self._label(
            dest,
            "  the file rolls when it reaches its size cap (1 GB by default).",
            fg=DIM
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        # action
        act = tk.Frame(self.stream_setup, bg=BG)
        act.grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.stream_start_btn = RoundedButton(
            act, "start stream", self._stream_start,
            width=160, height=36, font=self.font_main)
        self.stream_start_btn.pack(side="left")
        self.stream_open_btn = RoundedButton(
            act, "open streams", self._open_streams_folder,
            width=150, height=36, font=self.font_dim)
        self.stream_open_btn.pack(side="left", padx=(10, 0))
        self.stream_setup_status = self._label(act, "")
        self.stream_setup_status.pack(side="left", padx=(16, 0))

    def _build_stream_live(self):
        self.stream_live = tk.Frame(self.stream_card, bg=BG)
        self.stream_live.grid(row=0, column=0, sticky="nsew")
        self.stream_live.grid_rowconfigure(1, weight=1)
        self.stream_live.grid_columnconfigure(0, weight=1)

        # header with stop button
        head = tk.Frame(self.stream_live, bg=BG)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.stream_live_label = self._label(
            head, "", font=self.font_main, fg=FG)
        self.stream_live_label.pack(side="left")

        self.stream_stop_btn = RoundedButton(
            head, "stop stream", self._stream_stop,
            width=160, height=32, font=self.font_dim)
        self.stream_stop_btn.pack(side="right")

        # log
        self.stream_log = tk.Text(
            self.stream_live, bg=SURFACE, fg=FG, font=self.font_dim,
            wrap="char", bd=0, highlightthickness=0,
            padx=14, pady=10, state="disabled",
            selectbackground=SELECT_BG)
        self.stream_log.grid(row=1, column=0, sticky="nsew")

    def _on_stream_change(self, value):
        self._save_form_field("stream", "stream_name", value)
        self._update_stream_desc()
        self._update_stream_dest()

    def _update_stream_desc(self):
        name = self.stream_dd.value
        for s in self.streams:
            if s["name"] == name:
                self.stream_desc.config(text=s["description"])
                return
        self.stream_desc.config(text="")

    def _update_stream_dest(self):
        name = self.stream_dd.value
        if not name or name.startswith("("):
            self.stream_dest_label.config(text="—")
            return
        path = STREAMS_DATA_DIR / f"{name}.txt"
        # show a path relative to the runtime folder for readability
        try:
            rel = path.relative_to(RUNTIME_DIR)
            self.stream_dest_label.config(text=str(rel))
        except ValueError:
            self.stream_dest_label.config(text=str(path))

    def _set_stream_state(self, state):
        if state == 'idle':
            self.stream_live.grid_remove()
            self.stream_setup.grid()
        elif state == 'running':
            self.stream_setup.grid_remove()
            self.stream_live.grid()

    def _stream_start(self):
        name = self.stream_dd.value
        stream = next((s for s in self.streams if s["name"] == name), None)
        if not stream:
            self.stream_setup_status.config(text="no stream selected")
            return
        # no corpus arg → stream uses its default (data/streams/<name>.txt)
        if self.stream_proc.start(
                [stream["path"]],
                cwd=str(RUNTIME_DIR)):
            dest = STREAMS_DATA_DIR / f"{name}.txt"
            try:
                rel = dest.relative_to(RUNTIME_DIR)
                dest_str = str(rel)
            except ValueError:
                dest_str = str(dest)
            self.stream_live_label.config(
                text=f"running · {name} → {dest_str}")
            self._stream_log_append(
                f"started {name} → {dest_str}\n")
            self._set_stream_state('running')

    def _stream_stop(self):
        self.stream_proc.stop()
        self._stream_log_append("stop requested\n")

    def _stream_log_append(self, text):
        self.stream_log.config(state="normal")
        self.stream_log.insert("end", text)
        idx = float(self.stream_log.index("end").split(".")[0])
        if idx > 2200:
            self.stream_log.delete("1.0", f"{idx - 2000}.0")
        self.stream_log.see("end")
        self.stream_log.config(state="disabled")

    # ── logOS view ──

    def _build_logos(self):
        f = self.frames["logOS"]
        f.grid_rowconfigure(1, weight=1)
        f.grid_columnconfigure(0, weight=1)

        top = tk.Frame(f, bg=BG)
        top.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        self.logos_refresh_btn = RoundedButton(
            top, "refresh", self._logos_refresh,
            width=100, height=30, font=self.font_dim)
        self.logos_refresh_btn.pack(side="left")

        self.logos_detail_title = tk.Label(
            top, text="select a checkpoint", font=self.font_med,
            bg=BG, fg=FG, anchor="w", justify="left")
        self.logos_detail_title.pack(
            side="left", fill="x", expand=True, padx=(20, 0))

        body = tk.Frame(f, bg=BG)
        body.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=0, minsize=260)
        body.grid_columnconfigure(1, weight=1, minsize=360)

        left = tk.Frame(body, bg=BG, width=260)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        left.grid_propagate(False)
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.logos_list = tk.Listbox(
            left, bg=SURFACE, fg=FG, font=self.font_main,
            selectbackground=SELECT_BG, selectforeground=FG,
            activestyle="none", bd=0, highlightthickness=0,
            relief="flat", exportselection=False)
        self.logos_list.grid(row=0, column=0, sticky="nsew")
        self.logos_list.bind("<<ListboxSelect>>", self._logos_pick)

        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.logos_detail = tk.Text(
            right, bg=SURFACE, fg=FG, font=self.font_dim,
            wrap="char", bd=0, highlightthickness=0,
            padx=14, pady=10, state="disabled",
            selectbackground=SELECT_BG)
        self.logos_detail.grid(row=0, column=0, sticky="nsew",
                               pady=(0, 12))

        actions = tk.Frame(right, bg=BG)
        actions.grid(row=1, column=0, sticky="ew")

        self.logos_download_btn = RoundedButton(
            actions, "download", self._logos_download,
            width=130, height=34, font=self.font_main)
        self.logos_download_btn.pack(side="left")
        self.logos_download_btn.set_enabled(False)

        self.logos_tab_status = self._label(actions, "")
        self.logos_tab_status.pack(side="left", padx=(16, 0))

        self._logos_refresh()

    def _logos_set_detail(self, text):
        self.logos_detail.config(state="normal")
        self.logos_detail.delete("1.0", "end")
        self.logos_detail.insert("1.0", text)
        self.logos_detail.config(state="disabled")

    def _logos_refresh(self):
        if self.logos_downloading:
            return
        self.logos_refresh_btn.set_enabled(False)
        self.logos_tab_status.config(text="loading...")
        self.logos_list.delete(0, "end")
        self.logos_list.insert("end", "loading checkpoints...")
        self.logos_selected = None
        self.logos_download_btn.set_enabled(False)

        def thread():
            try:
                url = f"{LOGOS_API}/api/checkpoints?limit=50&sort=newest"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(
                        req, timeout=12, context=URL_CONTEXT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if not isinstance(data, list):
                    raise ValueError("unexpected response")
                self.root.after(0, lambda: self._logos_loaded(data))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: self._logos_load_failed(msg))

        threading.Thread(target=thread, daemon=True).start()

    def _logos_loaded(self, items):
        self.logos_checkpoints = items
        self.logos_list.delete(0, "end")
        if not items:
            self.logos_list.insert("end", "no checkpoints found")
        for item in items:
            size = _fmt_bytes(item.get("file_size_bytes", 0))
            creator = item.get("creator_username") or "unknown"
            name = item.get("name") or "untitled"
            self.logos_list.insert("end", f"{name} · {creator} · {size}")
        self.logos_tab_status.config(text=f"{len(items)} checkpoints")
        self.logos_refresh_btn.set_enabled(True)

    def _logos_load_failed(self, msg):
        self.logos_checkpoints = []
        self.logos_list.delete(0, "end")
        self.logos_list.insert("end", "could not load logOS")
        self.logos_detail_title.config(text="logOS unavailable")
        self._logos_set_detail(msg)
        self.logos_tab_status.config(text="load failed")
        self.logos_refresh_btn.set_enabled(True)

    def _logos_pick(self, event=None):
        sel = self.logos_list.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(self.logos_checkpoints):
            return
        item = self.logos_checkpoints[idx]
        self.logos_selected = item
        name = item.get("name") or "untitled"
        creator = item.get("creator_username") or "unknown"
        tags = ", ".join(item.get("corpus_tags") or []) or "untagged"
        checkpoint_id = item.get("checkpoint_id") or ""
        detail = "\n".join([
            item.get("description") or "no description",
            "",
            f"creator: {creator}",
            f"tags: {tags}",
            f"version: {item.get('soma_version') or 'unknown'}",
            f"bands: {item.get('n_bands')}  "
            f"d{item.get('token_dim')} / a{item.get('aperture_width')}  "
            f"depth: {item.get('language_depth')}",
            f"base: {item.get('base')}  direct: {item.get('direct_readout')}",
            f"trained: {_fmt_bytes(item.get('bytes_seen', 0))}",
            f"size: {_fmt_bytes(item.get('file_size_bytes', 0))}",
            f"downloads: {item.get('download_count', 0)}  stars: {item.get('star_count', 0)}",
            "",
            checkpoint_id,
        ])
        self.logos_detail_title.config(text=f"{name} · {creator}")
        self._logos_set_detail(detail)
        self.logos_download_btn.set_enabled(True)

    def _logos_download_name(self, item):
        raw = item.get("name") or item.get("checkpoint_id", "checkpoint")[:12]
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in raw)
        safe = safe.strip("._") or "checkpoint"
        if not safe.endswith(".pt"):
            safe += ".pt"
        return safe

    def _logos_download(self):
        item = self.logos_selected
        if not item or self.logos_downloading:
            return
        checkpoint_id = item.get("checkpoint_id")
        if not checkpoint_id:
            self.logos_tab_status.config(text="missing checkpoint id")
            return

        filename = self._logos_download_name(item)
        dest = CHECKPOINTS_DIR / filename
        if dest.exists():
            if not messagebox.askyesno(
                    "download checkpoint",
                    f"replace local checkpoint {filename}?",
                    parent=self.root):
                return

        self.logos_downloading = True
        self.logos_download_btn.set_enabled(False)
        self.logos_refresh_btn.set_enabled(False)
        self.logos_tab_status.config(text="requesting download...")

        def thread():
            tmp = dest.with_suffix(dest.suffix + ".download")
            try:
                api_url = f"{LOGOS_API}/api/checkpoints/{checkpoint_id}/download"
                req = urllib.request.Request(api_url, data=b"{}", method="POST")
                req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(
                        req, timeout=12, context=URL_CONTEXT) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                download_url = payload.get("download_url")
                if not download_url:
                    raise ValueError("missing download url")

                with urllib.request.urlopen(
                        download_url, timeout=30,
                        context=URL_CONTEXT) as resp:
                    total = int(resp.headers.get("Content-Length") or
                                item.get("file_size_bytes") or 0)
                    got = 0
                    with open(tmp, "wb") as f:
                        while True:
                            chunk = resp.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                            got += len(chunk)
                            if total:
                                pct = 100 * got / total
                                self.root.after(0, lambda p=pct:
                                    self.logos_tab_status.config(
                                        text=f"downloading {p:.0f}%"))
                tmp.replace(dest)
                self.root.after(0, lambda: self._logos_download_done(dest))
            except Exception as e:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                msg = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: self._logos_download_failed(msg))

        threading.Thread(target=thread, daemon=True).start()

    def _logos_download_done(self, path):
        self.logos_downloading = False
        self.logos_download_btn.set_enabled(True)
        self.logos_refresh_btn.set_enabled(True)
        self.logos_tab_status.config(text=f"downloaded {path.name}")
        self._refresh_chat_ckpts()
        names = self._list_checkpoint_names_with_none()
        if hasattr(self, "train_ckpt_dd"):
            self.train_ckpt_dd.set_options(names, value=path.name)
            self._on_train_ckpt_change(path.name)

    def _logos_download_failed(self, msg):
        self.logos_downloading = False
        self.logos_download_btn.set_enabled(True)
        self.logos_refresh_btn.set_enabled(True)
        self.logos_tab_status.config(text="download failed")
        self._logos_set_detail(msg)

    def _flush_chat_chars(self, parts):
        """paint a streamed response once per chunk, not once per token."""
        if not parts:
            return
        chunk = "".join(parts)
        parts.clear()
        self._chat_append(chunk, "soma", persist=False)
        self.chat_response_buf.append(chunk)
        self.voice.feed(chunk)
        if self._logos_mode:
            self.logos_response_buf.append(chunk)

    # ── poll loop ──

    def _poll(self):
        # stream subprocess output
        for line in self.stream_proc.drain():
            self._stream_log_append(line + "\n")
        if not self.stream_proc.is_running() and \
                self._train_state != 'training':
            # stream might have died on its own
            try:
                if self.stream_live.winfo_ismapped():
                    if hasattr(self, '_stream_was_running'):
                        self._stream_log_append("(stream stopped)\n")
                        self._set_stream_state('idle')
                        del self._stream_was_running
            except Exception:
                pass
        if self.stream_proc.is_running():
            self._stream_was_running = True

        # strictly on-device speech recognition. partial transcripts replace
        # only the microphone-owned suffix, leaving any typed prefix intact.
        for kind, text in self.speech.drain():
            if kind == "status":
                self.chat_voice_status.config(text=text)
            elif kind == "partial":
                self._set_speech_input(text)
            elif kind == "final":
                self._speech_received_final = True
                self._set_speech_input(text)
                self.chat_voice_status.config(text="local transcript ready")
                self.chat_input.focus_set()
                self.chat_input.mark_set("insert", "end-1c")
            elif kind == "error":
                self.chat_voice_status.config(text=f"speech: {text}")
                self._chat_append(f"speech: {text}\n", "dim")
            elif kind == "stopped":
                self.chat_mic_btn.set_text("speak")
                self.chat_mic_btn.set_enabled(True)
                if not self._speech_received_final:
                    self.chat_voice_status.config(text="microphone idle")

        # chat outputs. generation can enqueue hundreds of small decoded
        # fragments between polls. coalesce adjacent characters so transcript
        # maintenance cannot monopolise tkinter and starve pointer events.
        chat_parts = []
        for kind, *rest in self.chat.drain():
            if kind == 'char':
                chat_parts.append(rest[0])
                continue
            self._flush_chat_chars(chat_parts)
            if kind == 'status':
                self.chat_status.config(text=rest[0])
            elif kind == 'done':
                response = "".join(self.chat_response_buf)
                self.chat_response_buf.clear()
                self.voice.finish()
                self._chat_append("\n\n", "soma", persist=False)
                self._append_text_file(
                    CHAT_LOG_FILE, response + "\n\n",
                    max_bytes=MAX_CHAT_LOG_BYTES)
                self._set_chat_pending(False)
                mode = "online" if self.chat.online else "context"
                self.chat_status.config(text=f"loaded · {mode}")
                # if this was a logOS prompt, send the full response back.
                # json encoding keeps the stdin protocol one-line-safe even
                # when the model response contains newlines.
                if self._logos_mode and self.logos_response_buf:
                    response = "".join(self.logos_response_buf)
                    self.logos_response_buf.clear()
                    try:
                        if (self.logos_bridge.proc and
                                self.logos_bridge.proc.stdin):
                            payload = json.dumps(response)
                            self.logos_bridge.proc.stdin.write(
                                f"RESPONSE:{payload}\n")
                            self.logos_bridge.proc.stdin.flush()
                    except Exception as e:
                        self.logos_status.config(
                            text=f"send failed: {e}")
                else:
                    self.logos_response_buf.clear()
            elif kind == 'error':
                self.voice.stop()
                self._chat_append(f"\n  error: {rest[0]}\n\n", "dim")
                self._set_chat_pending(False)
                self.chat_response_buf.clear()
                self.logos_response_buf.clear()
        self._flush_chat_chars(chat_parts)

        # logOS bridge outputs
        for line in self.logos_bridge.drain():
            if line == "LOGOS:connected":
                self._logos_connected = True
                self.logos_status.config(text="connected")
            elif line == "LOGOS:started":
                self.logos_status.config(text="started")
            elif line.startswith("LOGOS:prompt:"):
                json_str = line[len("LOGOS:prompt:"):]
                try:
                    prompt = json.loads(json_str)
                except (json.JSONDecodeError, ValueError):
                    prompt = json_str
                if self.chat.model is not None and not self._chat_pending:
                    self.logos_status.config(text="thinking...")
                    speaker = getattr(self, '_chat_speaker', None) or 'soma'
                    w = max(4, len(speaker))
                    you_label = "web".ljust(w)
                    soma_label = speaker.ljust(w)
                    self._chat_append(
                        f"{you_label} › {prompt}\n", "you")
                    self._chat_append(f"{soma_label} › ", "soma")
                    self.logos_response_buf.clear()
                    self._set_chat_pending(True)
                    self.chat.online = (
                        self.chat_online_cb.get()
                        and not self._chat_readonly_snapshot
                    )
                    try:
                        self.chat.max_length = int(
                            self.chat_maxlen_entry.get())
                    except (ValueError, TypeError):
                        self.chat.max_length = 200
                    self.voice.enabled = self.chat_voice_cb.get()
                    if self.voice.enabled:
                        self.voice.begin()
                    self.chat.submit(prompt)
            elif line == "LOGOS:posted":
                self.logos_status.config(text="connected")
            elif line == "LOGOS:thinking":
                pass  # web user is typing, no action needed
            elif line.startswith("LOGOS:error:"):
                err = line[len("LOGOS:error:"):]
                self.logos_status.config(text=f"error: {err}")
            elif line == "LOGOS:stopped":
                self.logos_status.config(text="disconnected")
                self._logos_mode = False
                self.logos_cb.value = False
                self.logos_cb._update_display()

        # train outputs
        for kind, *rest in self.train_worker.drain():
            if kind == 'start':
                self.train_live_status.config(text="training")
                self._set_global_status("training", active=True)
                self._train_log("training worker started")
            elif kind == 'summary':
                d = rest[0]
                # format param count
                p = d["params"]
                if p >= 1e6:
                    p_str = f"{p/1e6:.1f}M"
                elif p >= 1e3:
                    p_str = f"{p/1e3:.1f}K"
                else:
                    p_str = str(p)
                active = int(d.get("active_params", p))
                if active >= 1e6:
                    active_str = f"{active/1e6:.1f}M"
                elif active >= 1e3:
                    active_str = f"{active/1e3:.1f}K"
                else:
                    active_str = str(active)
                mode = "loop" if d["loop"] else "single"
                lines = [
                    f"run {str(d.get('run_id', ''))[:12] or 'local'}  ·  "
                    f"{mode}  ·  {d['corpus']}",
                    f"{d['bands']} bands  ·  base={d['base']:.4f}"
                    f"  ·  {d['hidden']}  ·  {p_str} params",
                    f"dictionary: {d.get('tokenizer', 1024):,} tokens"
                    f"  ·  {_fmt_tokens(d.get('tokens_seen', 0))} tokens seen",
                    f"device: {d['device']}  ·  {d['dtype']}"
                    f"  ·  batch={d['batch']:,}",
                    f"saving to: {d['save_path']}"
                    f"  ·  {_fmt_bytes(d['bytes_seen'])} source seen",
                    f"v14 adaptive lens · lr "
                    f"{d.get('lr_base', d.get('lr', 0.0003)):.4g}"
                    f"  ·  decimation gain {d.get('decimation_gain', 1):g}",
                ]
                self.train_summary.config(text="\n".join(lines))
                self._train_save_path = str(
                    d.get("save_path_full") or d.get("save_path", ""))
                self._set_active_checkpoint(
                    Path(d['save_path']).stem,
                    f"{p_str} · {d['device']}")
                self.system_status.config(
                    text=f"{d['device']} · {d['dtype']} · batch {d['batch']:,}")
                self._train_log(
                    f"loaded {d['save_path']} · {d['corpus']} · {mode}")
            elif kind == 'report':
                d = rest[0]
                self._train_saving = False
                self._train_last_report_at = time.time()
                self._train_report_pos = int(d.get("pos", 0))
                self._train_report_total = int(d.get("total", 0))
                self._train_report_bps = float(d.get("bps", 0.0))
                pct = (100 * d["pos"] / d["total"]) if d["total"] else 0
                self._train_last_absolute_pos = int(
                    d.get("absolute_pos", d["pos"]))
                self._train_last_absolute_total = int(
                    d.get("absolute_total", d["total"]))
                sampled = d.get("metric_mode") == "sampled"
                coverage = 100.0 * float(d.get("prediction_coverage", 1.0))
                self.train_loss.config(
                    text=f"{d['bpb']:.2f} bpb"
                )
                self.primary_stats.set_label(
                    "acc",
                    "sampled next-token exact"
                    if sampled else "next-token exact",
                )
                self.primary_stats.set_label(
                    "top5",
                    "sampled next-token top 5"
                    if sampled else "next-token top 5",
                )
                self.train_loss_plot_label.config(
                    text=(
                        f"bpb · {coverage:.1f}% coverage"
                        if sampled
                        else "bpb · dense"
                    )
                )
                self.train_pct.config(text=f"{pct:.1f}%")

                self.cap_acc.config(text=f"{d['acc']:.1f}%")
                self.cap_top5.config(text=f"{d.get('top5_acc', 0.0):.1f}%")
                self.cap_bps.config(text=f"{d['bps']:,.0f}")
                telemetry = {
                    "seen": (f"{_fmt_bytes(d['life_bytes'])} / "
                             f"{_fmt_duration(d.get('life_seconds', 0))}"),
                    "lr": f"{d['lr']:.5f}",
                    "control": f"{d.get('io2', 1.0):.3f}",
                    "stride": (
                        f"{d.get('stride', 1)}t · "
                        f"{d.get('byte_stride', 1.0):.1f}b"
                    ),
                    "duty": f"{100*d.get('update_duty', 1.0):.0f}%",
                    "pulse": (
                        f"{100*d.get('pulse_frequency', 1.0):.0f}% · "
                        f"{100*d.get('update_cost_fraction', 0.0):.0f}%"
                    ),
                    "updates": (
                        f"{d.get('optimizer_updates', 0):,} · "
                        f"{100*d.get('optimizer_fill', 0.0):.0f}%"
                    ),
                    "pressure": (
                        f"{100*d.get('radial_pressure', 0.0):.1f}% · "
                        f"{d.get('row_spread', 1.0):.1f}x"
                    ),
                    "memory": (
                        f"{d.get('mps_allocated_gb', 0.0):.1f} / "
                        f"{d.get('mps_driver_gb', 0.0):.1f} gb"
                    ),
                    "error": f"{d.get('error_energy', 0.0):.3f}",
                    "epoch": f"{d['epoch']} / {d['epochs']}",
                }
                for key, value in telemetry.items():
                    self.train_telemetry.set_value(key, value)
                self.train_spectrum.set(
                    d.get('input_spectrum', []),
                    d.get('weight_spectrum', []),
                    d.get('error_spectrum', []),
                    d.get('target_band', 0.0),
                    d.get('input_coherence', 0.0),
                    d.get('output_coherence', 0.0),
                    d.get('error_energy', 0.0))
                memory = d.get('active_memory', [])
                predictions = d.get('top_predictions', [])
                self.active_memory_label.config(text=(
                    "slow trace deviations  " + "  ".join(
                        self._display_token(item.get('token', ''))
                        for item in memory[:6])
                    if memory else "slow trace deviations  —"))
                self.top_predictions_label.config(text=(
                    "next  " + "  ".join(
                        f"{self._display_token(item.get('token', ''))} "
                        f"{100*float(item.get('probability', 0.0)):.0f}%"
                        for item in predictions[:5])
                    if predictions else "next  —"))
                self.train_spark.push(
                    d["bpb"],
                    d.get("recent_loss", d["loss"]) / math.log(2.0))
                self._set_global_status(
                    f"training · {d['bpb']:.2f} bpb · "
                    f"{d['acc']:.1f}% · {d['bps']:,.0f} b/s", active=True)
            elif kind == 'autosaved':
                # quiet, periodic, just shows the user the system is
                # alive and saving without their input
                t = time.strftime("%H:%M:%S")
                self.train_live_status.config(
                    text=f"autosaved · {t}")
                self._train_log(f"autosaved {Path(rest[0]).name}")
                self._train_last_saved_checkpoint = rest[0]
                self._train_saving = False
                self._train_last_save_at = time.time()
                self._train_save_path = str(rest[0])
                self._commit_train_resume_cursor()
                self._refresh_chat_ckpts()
                self._refresh_model_library()
            elif kind == 'worker_recycle':
                details = rest[0] if rest and isinstance(rest[0], dict) else {}
                count = int(details.get("count", 1))
                self.train_live_status.config(
                    text=f"training engine refreshed · {count}"
                )
                self._train_log(
                    "training engine refreshed to release native memory; "
                    "resuming from the saved cursor"
                )
            elif kind == 'dream':
                d = rest[0]
                seen = _fmt_bytes(d["bytes_seen"])
                token_note = ""
                if d.get("input_tokens", 0):
                    token_note = (
                        f" · {d.get('output_tokens', 0)}/"
                        f"{d['input_tokens']} feedback tokens")
                self._train_log(
                    f"dream at update {d.get('update', d.get('batch', 0))} "
                    f"· {seen} seen"
                    f"{token_note}\n"
                    f"> {d['text']}")
            elif kind == 'checkpoint_saved':
                self.train_live_status.config(
                    text=f"saved {Path(rest[0]).name}")
                self._train_log(f"saved {Path(rest[0]).name}")
                self._train_last_saved_checkpoint = rest[0]
                self._train_saving = False
                self._train_last_save_at = time.time()
                self._train_save_path = str(rest[0])
                self._commit_train_resume_cursor()
                self._refresh_chat_ckpts()
                self._refresh_model_library()
            elif kind == 'saving':
                self._train_saving = True
                self.train_live_status.config(
                    text=f"publishing {Path(rest[0]).name}...")
                self._refresh_train_run_safety(force=True)
            elif kind == 'validation_start':
                details = rest[0] if rest and isinstance(rest[0], dict) else {}
                self._train_validating = True
                self.train_live_status.config(text="running dense validation...")
                self._train_log(
                    f"dense validation started · {details.get('reason', 'interval')}")
                self._refresh_train_run_safety(force=True)
            elif kind == 'validation':
                result = rest[0]
                self._train_validating = False
                self._train_last_validation_at = float(
                    result.get("timestamp", time.time()))
                self._train_validation_text = (
                    f"{float(result.get('bpb', 0.0)):.2f} est. bpb · "
                    f"{100 * float(result.get('top1', 0.0)):.1f}% exact"
                )
                self.train_live_status.config(
                    text=f"dense validation · {self._train_validation_text}")
                self._train_log(
                    f"dense validation · {self._train_validation_text} · "
                    f"{_fmt_duration(result.get('seconds', 0.0))}")
                self._set_global_status("training", active=True)
                self._refresh_train_run_safety(force=True)
            elif kind == 'info':
                self.train_live_status.config(text=rest[0])
                self._train_log(rest[0])
            elif kind == 'warning':
                self._train_validating = False
                self.train_live_status.config(text=f"warning: {rest[0]}")
                self._train_log(f"warning: {rest[0]}")
                self._set_global_status(
                    "training continues with a warning", warning=True)
            elif kind == 'error':
                self.train_live_status.config(text=f"error: {rest[0]}")
                self._train_log(f"error: {rest[0]}")
                self._set_global_status("training error", warning=True)
            elif kind == 'done':
                self._train_log("training worker stopped")
                self._set_train_state('stopped')
                run_status = str(
                    self.train_worker.manifest().get("status") or "stopped")
                if run_status in (
                        "failed", "interrupted", "stopped_without_save"):
                    readable = run_status.replace("_", " ")
                    self._set_global_status(
                        f"training {readable}", warning=True)
                    self.train_setup_status.config(
                        text=(
                            f"run {readable}; resume from the last safe "
                            "checkpoint"
                        ))
                else:
                    self._set_global_status("training stopped")
                self._select_train_checkpoint_after_new_model_save()

        self._refresh_train_run_safety()
        self.root.after(80, self._poll)

    # ── lifecycle ──

    def _on_close(self):
        self._closing = True
        # If chat has changed runtime state, ask before implicitly saving it.
        if self.chat.model is not None:
            path = getattr(self, '_chat_loaded_path', None)
            save_on_close = True
            if (getattr(self.chat, 'dirty', False)
                    and not self._chat_readonly_snapshot):
                choice = messagebox.askyesnocancel(
                    "quit soma",
                    "save chat model changes before quitting?\n\n"
                    "yes: save & quit\n"
                    "no: quit without saving\n"
                    "cancel: keep running",
                    parent=self.root,
                )
                if choice is None:
                    return
                save_on_close = bool(choice)
            if path and save_on_close and not self._chat_readonly_snapshot:
                try:
                    self._restore_chat_training_config()
                    self.chat.model.save(path)
                    self.chat.dirty = False
                except Exception as e:
                    print(f"chat save on quit failed: {e}", file=sys.stderr)

        # A long run may outlive the cockpit. Closing the window is not an
        # implicit request to stop learning.
        if self.train_worker.is_running():
            choice = messagebox.askyesnocancel(
                "quit soma",
                "stop the active training run?\n\n"
                "yes: save, stop, and quit\n"
                "no: leave training running and close the cockpit\n"
                "cancel: keep the cockpit open",
                parent=self.root,
            )
            if choice is None:
                self._closing = False
                return
            if choice:
                self.train_worker.stop(save=True)
                # A large checkpoint can take materially longer than two
                # seconds. Keep the cockpit alive until publication completes.
                import time as _t
                for _ in range(600):  # up to 60s
                    if not self.train_worker.is_running():
                        break
                    self.root.update_idletasks()
                    _t.sleep(0.1)
                saved = False
                for event in self.train_worker.drain():
                    if event and event[0] == "checkpoint_saved":
                        saved = True
                        self._train_last_saved_checkpoint = event[1]
                if saved:
                    self._commit_train_resume_cursor()

        # stop the other session-bound accessories.
        self.stream_proc.stop()
        self.logos_bridge.stop()
        self.speech.cancel()
        self.voice.shutdown()

        self.chat.shutdown()
        save_state(self.state)
        self.root.after(100, self.root.destroy)

    def run(self):
        self.root.mainloop()


def main():
    sys.path.insert(0, str(BUNDLE_DIR))
    App().run()


if __name__ == "__main__":
    main()

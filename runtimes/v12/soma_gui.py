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
import os
import sys
import json
import time
import queue
import signal
import threading
import subprocess
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

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

BG = "#000000"           # main page background
SURFACE = "#141414"      # rounded pill surface for inputs and buttons
SURFACE_HOVER = "#222222"
SURFACE_ACTIVE = "#2a2a2a"
FG = "#e8e8e8"           # primary text
DIM = "#777777"          # secondary text and labels
ACCENT = "#bbbbbb"
SELECT_BG = "#222222"
TAB_ACTIVE = "#1a1a1a"
SIDEBAR_BG = "#050505"
SPARK_LINE = "#cccccc"
SPARK_FILL = "#1a1a1a"

MONO_PREFS = ("SF Mono", "SFMono-Regular", "Menlo", "Monaco",
              "Courier New", "monospace")


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
CHAT_LOG_FILE = LOGS_DIR / "chat_history.txt"
TRAIN_LOG_FILE = LOGS_DIR / "train_history.txt"
MAX_CHAT_LOG_BYTES = 1_500_000
MAX_TRAIN_LOG_BYTES = 600_000
PUBLIC_DIR = Path.home() / "Documents" / "soma"
PUBLIC_DATA_LINK = PUBLIC_DIR / "data"
LOGOS_API = "https://logossoma.com"
DEFAULT_AUTO_MODE = "io2"
AUTO_MODE_DECIMATION_DEFAULTS = {
    "io2": "1",
    "model": "12",
    "wallclock": "12",
    "off": "0",
}


def default_decimation_for_mode(mode):
    return AUTO_MODE_DECIMATION_DEFAULTS.get(
        str(mode or DEFAULT_AUTO_MODE).strip().lower(),
        AUTO_MODE_DECIMATION_DEFAULTS[DEFAULT_AUTO_MODE])


def ensure_dirs():
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    STREAMS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    STREAMS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    STATE_DIR.mkdir(exist_ok=True)
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
    """Compact byte-count formatter. 1500 → '1.5K', 2_000_000 → '2.0M'."""
    n = float(n)
    for unit in ("B", "K", "M", "G"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}T"


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


def _data_cycle_files():
    if not DATA_DIR.exists():
        return []
    files = []
    for path in sorted(DATA_DIR.iterdir(), key=lambda p: p.name.lower()):
        if path.name.startswith(".") or path.name == "streams":
            continue
        if path.is_file():
            files.append(path)
    return files


# ──────────────────────────────────────────────────────────────────────
# state persistence
# ──────────────────────────────────────────────────────────────────────

DEFAULT_STATE = {
    "train": {
        "checkpoint": "",
        "mode": "corpus",        # "corpus" or "stream"
        "corpus": "",            # used when mode == "corpus"
        "stream_name": "",       # used when mode == "stream"
        "save_path": "model.pt",
        "start_byte": "0",
        "head_mb": "50",
        "autosave_min": "30",
        "dream_every_batches": "100",
        "dream_length": "auto 300",
        "dream_temperature": "1.0",
        "layers": "3",
        "auto_mode": "io2",
        "decimation": "1",
    },
    "chat": {
        "checkpoint": "",
        "online": False,
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
    "last_view": "chat",
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
        self.radius = height // 2
        self._draw_surface()

        self.entry = tk.Entry(
            self, bg=surface, fg=fg, font=font,
            insertbackground=fg, bd=0, highlightthickness=0,
            relief="flat", selectbackground=SELECT_BG,
            selectforeground=fg)
        if value:
            self.entry.insert(0, value)
        # place the Entry inside the pill, leaving room for the curves
        pad = self.radius // 2 + 4
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


class RoundedButton(tk.Canvas):
    """A pill-shaped button, drawn on a canvas with a label."""

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
        self.radius = height // 2
        self._current_surface = surface

        self._surface_id = None
        self._text_id = None
        self.text = text
        self.font = font
        self._redraw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _redraw(self):
        self.delete("all")
        pts = _round_rect_points(
            1, 1, self.width - 1, self.height - 1, self.radius - 1)
        self._surface_id = self.create_polygon(
            *pts, fill=self._current_surface, outline="", smooth=True)
        fg = self.fg if self._enabled else DIM
        self._text_id = self.create_text(
            self.width // 2, self.height // 2,
            text=self.text, fill=fg, font=self.font)

    def set_text(self, text):
        self.text = text
        self._redraw()

    def set_enabled(self, enabled):
        self._enabled = enabled
        self.config(cursor="hand2" if enabled else "")
        self._redraw()

    def _on_enter(self, event):
        if not self._enabled:
            return
        self._current_surface = self.hover
        self._redraw()
        self.config(cursor="hand2")

    def _on_leave(self, event):
        self._current_surface = self.surface
        self._redraw()

    def _on_press(self, event):
        if not self._enabled:
            return
        self._current_surface = self.active
        self._redraw()

    def _on_release(self, event):
        if not self._enabled:
            return
        self._current_surface = self.hover
        self._redraw()
        # only fire if the release is still over the button
        x, y = event.x, event.y
        if 0 <= x <= self.width and 0 <= y <= self.height:
            try:
                self.command()
            except Exception as e:
                print(f"button error: {e}", file=sys.stderr)


class RoundedDropdown(tk.Canvas):
    """A pill-shaped dropdown. Click reveals a tk.Menu; selection 
    fires on_select(value) and updates the visible label."""

    def __init__(self, parent, options, value=None, on_select=None,
                 width=240, height=32, font=None):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.radius = height // 2
        self.font = font
        self.options = list(options)
        self.value = value if value in self.options else (
            self.options[0] if self.options else "")
        self.on_select = on_select
        self._current_surface = SURFACE
        self._redraw()

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._show_menu)

    def _redraw(self):
        self.delete("all")
        pts = _round_rect_points(
            1, 1, self.width - 1, self.height - 1, self.radius - 1)
        self.create_polygon(
            *pts, fill=self._current_surface, outline="", smooth=True)
        # left-aligned label
        pad = self.radius // 2 + 4
        self.create_text(
            pad, self.height // 2,
            anchor="w",
            text=self._ellipsize(self.value),
            fill=FG, font=self.font)
        # right-side caret
        cx = self.width - self.radius
        cy = self.height // 2
        self.create_polygon(
            cx - 4, cy - 2, cx + 4, cy - 2, cx, cy + 3,
            fill=DIM, outline="")

    def _ellipsize(self, text):
        text = str(text)
        try:
            measure_font = tkfont.Font(font=self.font)
            pad = self.radius // 2 + 4
            available = max(20, self.width - pad - self.radius - 16)
            if measure_font.measure(text) <= available:
                return text
            suffix = "..."
            while text and measure_font.measure(text + suffix) > available:
                text = text[:-1]
            return text + suffix if text else suffix
        except Exception:
            return text

    def _on_enter(self, event):
        self._current_surface = SURFACE_HOVER
        self._redraw()
        self.config(cursor="hand2")

    def _on_leave(self, event):
        self._current_surface = SURFACE
        self._redraw()

    def _show_menu(self, event):
        if not self.options:
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
        self._redraw()
        if self.on_select:
            self.on_select(value)

    def set_options(self, options, value=None):
        self.options = list(options)
        if value is not None and value in self.options:
            self.value = value
        elif self.value not in self.options:
            self.value = self.options[0] if self.options else ""
        self._redraw()


class RoundedCheckbox(tk.Canvas):
    """Pill-style checkbox with text label."""

    def __init__(self, parent, text, value=False,
                 on_change=None, width=200, height=24, font=None):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.value = bool(value)
        self.text = text
        self.font = font
        self.on_change = on_change
        self._redraw()
        self.bind("<Button-1>", self._toggle)
        self.config(cursor="hand2")

    def _redraw(self):
        self.delete("all")
        # checkbox box
        size = 14
        x1 = 4
        y1 = (self.height - size) // 2
        x2 = x1 + size
        y2 = y1 + size
        r = 4
        pts = _round_rect_points(x1, y1, x2, y2, r)
        fill = FG if self.value else SURFACE
        self.create_polygon(
            *pts, fill=fill, outline=DIM, width=1, smooth=True)
        if self.value:
            # check mark
            self.create_line(
                x1 + 3, (y1 + y2) // 2,
                x1 + 6, y2 - 3,
                x2 - 3, y1 + 3,
                fill=BG, width=2,
                capstyle="round", joinstyle="round")
        # label
        self.create_text(
            x2 + 8, self.height // 2,
            anchor="w", text=self.text, fill=FG, font=self.font)

    def _toggle(self, event):
        self.value = not self.value
        self._redraw()
        if self.on_change:
            self.on_change(self.value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = bool(value)
        self._redraw()


class RoundedSegmented(tk.Canvas):
    """A two-or-more-option segmented pill. The currently-selected
    segment is filled with the active color; others are dim. Click a
    segment to select it."""

    def __init__(self, parent, options, value=None, on_change=None,
                 width=240, height=30, font=None):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0)
        self.width = width
        self.height = height
        self.radius = height // 2
        self.options = list(options)
        self.value = value if value in self.options else self.options[0]
        self.on_change = on_change
        self.font = font
        self._segment_bounds = []  # filled by _redraw
        self._redraw()
        self.bind("<Button-1>", self._on_click)
        self.config(cursor="hand2")

    def _redraw(self):
        self.delete("all")
        # outer pill (dim track)
        pts = _round_rect_points(
            1, 1, self.width - 1, self.height - 1, self.radius - 1)
        self.create_polygon(
            *pts, fill=SURFACE, outline="", smooth=True)

        n = len(self.options)
        if n == 0:
            return
        seg_w = self.width / n
        self._segment_bounds = []
        for i, opt in enumerate(self.options):
            x1 = i * seg_w
            x2 = (i + 1) * seg_w
            self._segment_bounds.append((x1, x2, opt))
            if opt == self.value:
                # active segment fill
                inset = 2
                pts = _round_rect_points(
                    x1 + inset, inset,
                    x2 - inset, self.height - inset,
                    self.radius - inset - 1)
                self.create_polygon(
                    *pts, fill=SURFACE_ACTIVE, outline="", smooth=True)
            fg = FG if opt == self.value else DIM
            self.create_text(
                (x1 + x2) / 2, self.height / 2,
                text=opt, fill=fg, font=self.font)

    def _on_click(self, event):
        for x1, x2, opt in self._segment_bounds:
            if x1 <= event.x <= x2:
                if opt != self.value:
                    self.value = opt
                    self._redraw()
                    if self.on_change:
                        self.on_change(opt)
                return

    def get(self):
        return self.value

    def set(self, value):
        if value in self.options:
            self.value = value
            self._redraw()


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

    def stop(self):
        if not self.is_running():
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        except (OSError, ProcessLookupError):
            pass

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
        self.dirty = False

    def load(self, ckpt_path, online=False,
             temperature=0.8, max_length=200):
        import soma_v12_2 as soma_runtime
        self.release_model()
        model = soma_runtime.SOMA(device="auto")
        model.load(ckpt_path)
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
        self.tasks.put(text)

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
                for ch in self.model.generate(
                        length=self.max_length,
                        temperature=self.temperature,
                        prelude_source="chat"):
                    self.outbox.put(('char', ch))
                # Prompt ingestion and generation both advance the trace bank;
                # online mode may also update weights. Treat a completed turn
                # as dirty so unload can offer save/discard/cancel.
                self.dirty = True
                self.outbox.put(('done',))
            except Exception as e:
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

class InlineTrainWorker:
    def __init__(self):
        self.outbox = queue.Queue(maxsize=10000)
        self._thread = None
        self._stop = False
        self._save_on_stop = True
        self.model = None
        self._last_memory_hygiene = 0.0

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, config):
        if self.is_running():
            return False
        self._stop = False
        self._save_on_stop = True
        self._thread = threading.Thread(
            target=self._run, args=(config,), daemon=True)
        self._thread.start()
        return True

    def stop(self, save=True):
        self._save_on_stop = bool(save)
        self._stop = True

    def _memory_hygiene(self, model=None, force=False):
        now = time.time()
        if not force and now - self._last_memory_hygiene < 60.0:
            return
        self._last_memory_hygiene = now
        device = getattr(getattr(model, "device", None), "type", None)
        release_torch_memory(device)

    def _release_model(self):
        model = self.model
        device = getattr(getattr(model, "device", None), "type", None)
        self.model = None
        if model is not None:
            del model
        release_torch_memory(device)

    def _run(self, config):
        try:
            try:
                import soma_v12_2 as soma_runtime
            except Exception as e:
                self.outbox.put(('error', f"import failed: {e}"))
                return

            self.outbox.put(('start',))

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
                self.outbox.put(('info', f'resumed {Path(ckpt).name}'))
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
                self.outbox.put(('info', 'fresh model'))

            self.model = model

            # compute model stats for the summary display
            if hasattr(model, "net") and hasattr(model, "params"):
                n_params = model.params()
            elif hasattr(model, "W_list"):
                n_params = sum(W.numel() for W in model.W_list)
                n_params += sum(U.numel() for U in getattr(model, "U_list", []))
                n_params += sum(G.numel() for G in getattr(model, "G_list", []))
                n_params += sum(gb.numel() for gb in getattr(model, "gb_list", []))
                if model.Wd is not None:
                    n_params += model.Wd.numel()
            elif model.hidden_dim > 0:
                n_params = model.U.numel() + model.W.numel()
                if model.Wd is not None:
                    n_params += model.Wd.numel()
            else:
                n_params = model.W.numel()
            hidden_str = (f"hidden={model.hidden_dim:,}"
                          if model.hidden_dim > 0 else "linear")
            if hasattr(model, "depth"):
                hidden_str += f" · depth={model.depth}"
            elif model.hidden_dim > 0 and model.Wd is not None:
                hidden_str += " + direct"
            if not hasattr(model, "depth") and getattr(model, "n_layers", 1) > 1:
                hidden_str += f" · stacks={model.n_layers}"
            if not hasattr(model, "depth") and getattr(model, "scale_gate", False):
                hidden_str += " · gated"
            if not hasattr(model, "depth") and getattr(model, "clock", 1) > 1:
                hidden_str += f" · clock={model.clock}"

            corpus_name = config.get("corpus_label") or Path(config["corpus"]).name
            outbox = self.outbox
            outbox.put(('summary', {
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
                "auto_mode": getattr(model, "auto_mode", DEFAULT_AUTO_MODE),
                "decimation_range": getattr(
                    model, "decimation_range",
                    default_decimation_for_mode(DEFAULT_AUTO_MODE)),
                "bytes_seen": model.bytes_seen,
                "save_path": Path(config["save_path"]).name,
                "loop": bool(config.get("loop", False)),
            }))
            stop_flag = lambda: self._stop
            ln2 = __import__('math').log(2)

            save_path = config["save_path"]
            start_byte = int(config.get("start_byte", 0))

            # session lifetime tracking (across cycles in loop mode).
            session_t0 = time.time()
            session_bytes_start = model.bytes_seen

            # Autosave is wall-clock based and independent of batch cadence.
            # 0 disables periodic autosave; stop/completion still respects the
            # user's save/discard choice.
            autosave_min = float(config.get("autosave_min", 0.5))
            autosave_interval_s = max(0.0, autosave_min * 60.0)
            last_save_t = [time.time()]
            self._autosave_interval_s = autosave_interval_s
            self._last_save_t = last_save_t

            # rate-limit reports to the GUI — fast models can fire
            # hundreds of batches per second; the Tk main loop can't
            # keep up with that many widget updates.
            MIN_EMIT_S = 0.15
            last_emit_t = [0.0]

            def patched_report(epoch, epochs, pos, total, loss,
                               correct, samples, t0):
                """Called once per batch by train(). All args are
                already-synced Python numbers (no device tensors).
                Compute stats and push to the GUI, rate-limited."""
                if stop_flag():
                    raise KeyboardInterrupt("gui requested stop")

                now = time.time()

                # only emit to the GUI if enough time has passed
                if now - last_emit_t[0] >= MIN_EMIT_S:
                    cum_avg = loss / samples if samples > 0 else 0
                    cum_bpb = cum_avg / ln2
                    cum_acc = 100 * correct / samples if samples > 0 else 0
                    bps = pos / max(1e-9, now - t0)

                    life_dt = max(1e-9, now - session_t0)
                    life_bytes = model.bytes_seen - session_bytes_start
                    life_bps = life_bytes / life_dt

                    outbox.put(('report', {
                        "pos": int(pos),
                        "total": int(total),
                        "loss": float(cum_avg),
                        "bpb": float(cum_bpb),
                        "acc": float(cum_acc),
                        "bps": float(bps),
                        "life_bytes": int(life_bytes),
                        "life_bps": float(life_bps),
                        "life_seconds": float(life_dt),
                        "epoch": int(epoch + 1),
                        "epochs": int(epochs),
                        "lr": float(getattr(model, "lr", 0.001)),
                        "auto_mode": getattr(
                            model, "auto_mode", DEFAULT_AUTO_MODE),
                        "grad_clip": float(getattr(model, "grad_clip", 1.0)),
                        "decimation_band": float(getattr(
                            model, "decimation_band", 0.0)),
                        "stride": int(getattr(model, "_stride", 1)),
                        "sampled_stride": int(getattr(
                            model, "_sampled_stride",
                            getattr(model, "_stride", 1))),
                        "io2": float(getattr(
                            model, "_io2_plasticity", 0.0)),
                        "motor": float(getattr(model, "_motor_value", 0.0)),
                        "motor_delta": float(getattr(
                            model, "_motor_delta", 0.0)),
                        "motor_energy": float(getattr(
                            model, "_motor_energy_push", 0.0)),
                        "row_clip": float(getattr(
                            model, "_row_clip_fraction", 0.0)),
                    }))
                    last_emit_t[0] = now

                self._memory_hygiene(model)

                # wall-clock autosave (runs regardless of rate limit)
                if (autosave_interval_s > 0
                        and now - last_save_t[0] >= autosave_interval_s):
                    try:
                        model.save(save_path)
                        self._memory_hygiene(model, force=True)
                        outbox.put(('autosaved', save_path))
                    except Exception as e:
                        outbox.put(('error', f"autosave failed: {e}"))
                    last_save_t[0] = now

            model._report = patched_report
            model._dream_callback = lambda text, batch, seen: outbox.put(
                ('dream', {
                    "text": text,
                    "batch": batch,
                    "bytes_seen": seen,
                }))

            corpus_path = config["corpus"]
            loop_mode = bool(config.get("loop", False))
            head_bytes = int(config.get("head_bytes", 50_000_000))
            cycle_pause = float(config.get("cycle_pause", 60.0))
            self._dream_every_batches = int(
                config.get("dream_every_batches", 0))
            self._dream_length = str(config.get("dream_length", "200"))
            self._dream_temperature = float(
                config.get("dream_temperature", 0.8))

            if loop_mode:
                self._run_loop(model, corpus_path, save_path,
                               head_bytes, cycle_pause)
            elif config.get("cycle_files"):
                self._run_cycle(model, config["cycle_files"],
                                int(config.get("cycle_count", 1)),
                                save_path)
            else:
                self._run_single(model, corpus_path, save_path,
                                 start_byte)

        except Exception as e:
            self.outbox.put(('error', f"{type(e).__name__}: {e}"))
        finally:
            self._release_model()
            self.outbox.put(('done',))

    def _run_single(self, model, corpus_path, save_path, start_byte=0):
        try:
            model.train(
                corpus_path,
                epochs=1,
                save_every=0,
                save_path=save_path,
                start_byte=start_byte,
                report_every=1,
                dream_every_batches=self._dream_every_batches,
                dream_length=self._dream_length,
                dream_temperature=self._dream_temperature,
                dream_callback=model._dream_callback,
            )
        except KeyboardInterrupt:
            pass

        # final save on completion or normal stop
        if not self._save_on_stop:
            self.outbox.put(('info', 'stopped without saving'))
            return
        try:
            model.save(save_path)
            self._memory_hygiene(model, force=True)
            self.outbox.put(('checkpoint_saved', save_path))
        except Exception as e:
            self.outbox.put(('error', f"save failed: {e}"))

    def _run_cycle(self, model, corpus_paths, cycle_count, save_path):
        cycle_count = max(1, int(cycle_count))
        files = [str(p) for p in corpus_paths]
        for cycle_idx in range(cycle_count):
            if self._stop:
                break
            self.outbox.put(('info',
                f"cycle {cycle_idx + 1}/{cycle_count} · {len(files)} files"))
            for idx, corpus_path in enumerate(files, 1):
                if self._stop:
                    break
                self.outbox.put(('info',
                    f"file {idx}/{len(files)} · {Path(corpus_path).name}"))
                try:
                    model.train(
                        corpus_path,
                        epochs=1,
                        save_every=0,
                        save_path=save_path,
                        start_byte=0,
                        report_every=1,
                        dream_every_batches=self._dream_every_batches,
                        dream_length=self._dream_length,
                        dream_temperature=self._dream_temperature,
                        dream_callback=model._dream_callback,
                    )
                except KeyboardInterrupt:
                    self._stop = True
                    break
                except Exception as e:
                    self.outbox.put(('error',
                        f"cycle file failed: {Path(corpus_path).name}: {e}"))

        if not self._save_on_stop:
            self.outbox.put(('info', 'stopped without saving'))
            return
        try:
            model.save(save_path)
            self._memory_hygiene(model, force=True)
            self.outbox.put(('checkpoint_saved', save_path))
        except Exception as e:
            self.outbox.put(('error', f"save failed: {e}"))

    def _run_loop(self, model, corpus_path, save_path,
                  head_bytes, cycle_pause):
        """Perpetual training on a rolling corpus.

        Each cycle:
          - snapshot the head N bytes of the corpus to a temp file
          - train on that snapshot
          - save the checkpoint
          - pause for cycle_pause seconds (to let the corpus grow)
          - repeat
        """
        tmp_head = Path(save_path).parent / f".{Path(save_path).stem}_head.tmp"

        def snapshot_head():
            corpus = Path(corpus_path)
            if not corpus.exists():
                return None
            n = min(head_bytes, corpus.stat().st_size)
            if n < 1_000_000:  # too small to bother
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
        while not self._stop:
            cycle += 1
            n_written = snapshot_head()
            if n_written is None:
                self.outbox.put(('info',
                    'corpus too small or missing — waiting'))
                self._sleep_responsive(cycle_pause)
                continue

            self.outbox.put(('info', f'cycle {cycle} · '
                f'head={n_written // 1_000_000}M'))

            try:
                model.train(
                    str(tmp_head),
                    epochs=1,
                    save_every=0,
                    save_path=save_path,
                    start_byte=0,
                    report_every=1,
                    dream_every_batches=self._dream_every_batches,
                    dream_length=self._dream_length,
                    dream_temperature=self._dream_temperature,
                    dream_callback=model._dream_callback,
                )
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.outbox.put(('error', f"cycle error: {e}"))
                self._sleep_responsive(cycle_pause)
                continue
            finally:
                try:
                    tmp_head.unlink()
                except FileNotFoundError:
                    pass

            # cycle-end save (in addition to wall-clock autosaves
            # during the cycle). For large checkpoints this is not cheap:
            # high decimation can finish cycles quickly and otherwise trigger
            # repeated multi-GB saves. Respect the autosave interval here.
            if (getattr(self, "_autosave_interval_s", 0.0) > 0
                    and time.time() - self._last_save_t[0]
                    >= self._autosave_interval_s):
                try:
                    model.save(save_path)
                    self._memory_hygiene(model, force=True)
                    self._last_save_t[0] = time.time()
                    self.outbox.put(('checkpoint_saved', save_path))
                except Exception as e:
                    self.outbox.put(('error', f"save failed: {e}"))

            if not self._stop:
                self.outbox.put(('info',
                    f'pausing {int(cycle_pause)}s before next cycle'))
                self._sleep_responsive(cycle_pause)

        # final save on graceful stop
        if not self._save_on_stop:
            self.outbox.put(('info', 'stopped without saving'))
            return
        try:
            model.save(save_path)
            self._memory_hygiene(model, force=True)
            self.outbox.put(('checkpoint_saved', save_path))
        except Exception as e:
            self.outbox.put(('error', f"save failed: {e}"))

    def _sleep_responsive(self, seconds):
        """Sleep but check the stop flag every second."""
        slept = 0.0
        while slept < seconds and not self._stop:
            time.sleep(min(1.0, seconds - slept))
            slept += 1.0

    def drain(self, max_items=500):
        out = []
        while len(out) < max_items:
            try:
                out.append(self.outbox.get_nowait())
            except queue.Empty:
                break
        return out


class TrainWorker:
    """subprocess-backed train worker.

    training is deliberately isolated from the gui process. when a run stops,
    the child process exits and releases torch/mps state at the os boundary.
    """

    def __init__(self):
        self.proc = ManagedProcess("train")
        self.config_path = None
        self.stop_path = None
        self._done_seen = False

    def is_running(self):
        return self.proc.is_running()

    def start(self, config):
        if self.is_running():
            return False
        token = f"{int(time.time() * 1000)}"
        self.config_path = STATE_DIR / f"train_worker_{token}.json"
        self.stop_path = STATE_DIR / f"train_stop_{token}.json"
        try:
            if self.stop_path.exists():
                self.stop_path.unlink()
            self.config_path.write_text(
                json.dumps(config), encoding="utf-8")
        except OSError as e:
            self.proc.queue.put(f'["error", "worker config failed: {e}"]')
            return False
        self._done_seen = False
        return self.proc.start([
            str(BUNDLE_DIR / "soma_train_worker.py"),
            str(self.config_path),
            str(self.stop_path),
        ], cwd=str(BUNDLE_DIR))

    def stop(self, save=True):
        try:
            if self.stop_path:
                self.stop_path.write_text(
                    json.dumps({"save": bool(save)}), encoding="utf-8")
        except OSError:
            pass

    def drain(self, max_items=500):
        out = []
        for line in self.proc.drain(max_items):
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
        if (not self.proc.is_running()
                and not self._done_seen
                and self.config_path is not None):
            self._done_seen = True
            out.append(("done",))
        return out


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

    def push(self, v):
        self.values.append(float(v))
        self._redraw()

    def reset(self):
        self.values.clear()
        self.delete("all")

    def _redraw(self):
        self.delete("all")
        if len(self.values) < 2:
            return
        vmin = min(self.values)
        vmax = max(self.values)
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
        self.create_line(*pts, fill=SPARK_LINE, width=1.5, smooth=False)


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
        self.root.minsize(820, 560)
        self.root.configure(bg=BG)

        self.mono = pick_mono(self.root)
        self.font_main = (self.mono, 12)
        self.font_dim = (self.mono, 11)
        self.font_title = (self.mono, 13)
        self.font_huge = (self.mono, 28)
        self.font_med = (self.mono, 14)

        self.stream_proc = ManagedProcess("stream")
        self.logos_bridge = ManagedProcess("logos_bridge")
        self.chat = ChatWorker()
        self.train_worker = TrainWorker()
        self._chat_pending = False
        self._logos_mode = False
        self._logos_connected = False
        self.chat_response_buf = []
        self.logos_response_buf = []
        self.logos_checkpoints = []
        self.logos_selected = None
        self.logos_downloading = False
        self._closing = False
        self._train_resume_cursor_enabled = False
        self._active_train_resume_key = ""
        self._train_last_absolute_pos = 0
        self._train_last_absolute_total = 0
        self._train_started_new_model = False
        self._train_last_saved_checkpoint = ""
        self._train_discarded_without_save = False
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
        self._build_chat()
        self._build_train()
        self._build_stream()
        self._build_logos()

        # if a checkpoint is pre-selected from saved state, populate the
        # train form fields from it (the dropdown's on_select callback
        # only fires on click, not on initial value assignment)
        initial_ckpt = self.train_ckpt_dd.value
        if initial_ckpt and initial_ckpt != "(new model)":
            self._on_train_ckpt_change(initial_ckpt)

        # validate saved view — earlier versions had different tabs
        last = self.state.get("last_view", "chat")
        if last not in self.frames:
            last = "chat"
        self._show(last)
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── layout ──

    def _build_layout(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        self.sidebar = tk.Frame(self.root, bg=SIDEBAR_BG, width=136)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)

        tk.Label(self.sidebar, text="░▒▓ soma ▓▒░",
                 font=self.font_title, bg=SIDEBAR_BG, fg=FG,
                 anchor="w", padx=18, pady=20).pack(fill="x")

        self.tabs = {}
        for label in ("chat", "train", "stream", "logOS"):
            btn = tk.Label(
                self.sidebar, text=f"  {label}",
                font=self.font_main, bg=SIDEBAR_BG, fg=DIM,
                anchor="w", padx=18, pady=6, cursor="hand2")
            btn.pack(fill="x")
            btn.bind("<Button-1>", lambda e, n=label: self._show(n))
            self.tabs[label] = btn

        # quit button anchored at the bottom of the sidebar
        quit_wrap = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        quit_wrap.pack(side="bottom", fill="x", padx=18, pady=18)
        self.quit_btn = RoundedButton(
            quit_wrap, "quit", self._quit_app,
            width=104, height=30, font=self.font_dim)
        self.quit_btn.pack(anchor="w")

        self.main = tk.Frame(self.root, bg=BG)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_rowconfigure(0, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for name in ("chat", "train", "stream", "logOS"):
            f = tk.Frame(self.main, bg=BG)
            f.grid(row=0, column=0, sticky="nsew")
            self.frames[name] = f

    def _quit_app(self):
        """Triggered by the sidebar quit button. Same flow as window close."""
        self._on_close()

    def _show(self, which):
        self.frames[which].tkraise()
        for name, btn in self.tabs.items():
            if name == which:
                btn.config(fg=FG, bg=TAB_ACTIVE)
            else:
                btn.config(fg=DIM, bg=SIDEBAR_BG)
        self.state["last_view"] = which
        save_state(self.state)

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

    # ── chat view ──

    def _build_chat(self):
        f = self.frames["chat"]
        f.grid_rowconfigure(3, weight=1)
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

        self._label(settings, "response max").pack(
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

        # row 2b — logOS remote chat
        logos_row = tk.Frame(f, bg=BG)
        logos_row.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 10))

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

        # transcript (shifted to row 3)
        transcript = tk.Frame(f, bg=BG)
        transcript.grid(row=3, column=0, sticky="nsew",
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
        bottom.grid(row=4, column=0, sticky="ew", padx=20, pady=(0, 20))
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
        self.chat_send_btn.grid(row=0, column=1, sticky="s")

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
        self.root.update_idletasks()

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
        online_mode = self.chat_online_cb.get()

        def thread():
            try:
                self.chat.load(
                    str(path),
                    online=online_mode,
                    temperature=temperature,
                    max_length=max_length)
                self._chat_speaker = speaker
                self._chat_loaded_path = str(path)
                mode = "online" if online_mode else "context"
                self.root.after(0, lambda: self.chat_status.config(
                    text=f"loaded {name} · {mode}"))
                self.root.after(0, lambda: self._chat_append(
                    f"loaded {name}\n", "dim"))
                self.root.after(0,
                    lambda: self.chat_unload_btn.set_enabled(True))
            except Exception as e:
                msg = f"{type(e).__name__}: {e}"
                self.root.after(0, lambda: self.chat_status.config(
                    text="load failed"))
                self.root.after(0, lambda: self._chat_append(
                    f"load failed: {msg}\n", "dim"))

        threading.Thread(target=thread, daemon=True).start()

    def _restore_chat_training_config(self):
        """Restore checkpoint hyperparameters before saving chat state.

        Older builds temporarily overrode chat decimation. Current chat uses
        the checkpoint controller state, but keep this guard so older loaded
        checkpoints still save with their original training settings.
        """
        model = self.chat.model
        if model is None:
            return
        if hasattr(model, 'checkpoint_batch_size'):
            model.batch_size = model.checkpoint_batch_size
        if hasattr(model, 'checkpoint_decimation_band'):
            model.decimation_band = model.checkpoint_decimation_band
            model._update_decimation()

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

        if getattr(self.chat, 'dirty', False):
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
            self._finish_chat_unload(f"unloaded {name} without saving\n")

    def _finish_chat_unload(self, message=None):
        """Clear chat model/UI state after save or discard."""
        if message:
            self._chat_append(message, "dim")
        self.chat.release_model()
        self.chat.dirty = False
        self._chat_speaker = None
        self._chat_loaded_path = None
        self.chat_status.config(text="no model loaded")
        self.chat_unload_btn.set_enabled(False)

    def _chat_input_return(self, event):
        self._chat_send()
        return "break"

    def _chat_input_newline(self, event):
        self.chat_input.insert("insert", "\n")
        self._resize_chat_input()
        return "break"

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
        text = self.chat_input.get("1.0", "end-1c").strip()
        if not text or self.chat.model is None or self._chat_pending:
            return
        self.chat_input.delete("1.0", "end")
        self._resize_chat_input()
        # pick up live settings each send — user may have changed them
        self.chat.online = self.chat_online_cb.get()
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
        # right-align the two speaker labels by padding to a common width
        w = max(4, len(speaker))
        you_label = "you".ljust(w)
        soma_label = speaker.ljust(w)
        self._chat_append(f"{you_label} › {text}\n", "you")
        self._chat_append(f"{soma_label} › ", "soma")
        self._chat_pending = True
        self.chat.submit(text)

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
                self.logos_cb._redraw()
                self._logos_mode = False
                return

            pass_prompt = self.logos_pass_entry.get().strip()
            if not pass_prompt:
                self.logos_status.config(text="enter a pass prompt")
                self.logos_cb.value = False
                self.logos_cb._redraw()
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

        self._build_train_setup()
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
            "bands": "20", "base": "1.6180", "hidden": "1536",
            "layers": "3",
            "lr": "auto 0.001", "grad_clip": "1.0",
            "batch": "512", "auto_mode": DEFAULT_AUTO_MODE,
            "decimation": default_decimation_for_mode(DEFAULT_AUTO_MODE),
            "start_byte": "0", "autosave_min": "30",
            "dream_every_batches": "100", "dream_length": "auto 300",
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
        field("batch", "batch", 6, 0)
        field("lr", "lr", 6, 2)
        field("gradclip", "grad_clip", 7, 0)
        field("auto mode", "auto_mode", 7, 2)
        field("decimation", "decimation", 8, 0)
        field("start byte", "start_byte", 8, 2)
        field("autosave min", "autosave_min", 9, 0)
        field("dream every batches", "dream_every_batches", 9, 2)
        field("dream length", "dream_length", 10, 0)
        field("temperature", "dream_temperature", 10, 2)

        # ── action row ──
        self.train_actions = tk.Frame(params, bg=BG)

        self.train_start_btn = RoundedButton(
            self.train_actions, "start", self._train_start,
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

    def _set_train_ckpt_info(self, text):
        if not hasattr(self, "train_ckpt_info"):
            return
        self.train_ckpt_info.config(state="normal")
        self.train_ckpt_info.delete("1.0", "end")
        self.train_ckpt_info.insert("1.0", text)
        self.train_ckpt_info.config(state="disabled")

    def _list_checkpoint_names_with_none(self):
        files = sorted(p.name for p in CHECKPOINTS_DIR.glob("*.pt"))
        return ["(new model)"] + files

    def _on_train_mode_change(self, mode):
        """Switch the data-source widget shown in the train form."""
        self._save_form_field("train", "mode", mode)
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
        """Keep the visible decimation range meaningful for each controller."""
        mode = str(value or DEFAULT_AUTO_MODE).strip().lower()
        if mode not in AUTO_MODE_DECIMATION_DEFAULTS:
            mode = DEFAULT_AUTO_MODE
            self.train_fields["auto_mode"].set(mode)
        self._save_form_field("train", "auto_mode", mode)
        if "decimation" in self.train_fields:
            decimation = default_decimation_for_mode(mode)
            self.train_fields["decimation"].set(decimation)
            self._save_form_field("train", "decimation", decimation)

    def _place_train_actions(self, mode=None):
        if not hasattr(self, "train_actions"):
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
                if path.name.startswith(".") or path.name == "streams":
                    continue
                if path.is_file():
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
        try:
            menu.tk_popup(x, y)
        finally:
            menu.grab_release()

    def _pick_train_corpus(self, name):
        self.train_corpus_entry.set(name)
        self._save_form_field("train", "corpus", name)
        self._load_train_resume_position()

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
            return f"new:{save}"
        return value or "(new model)"

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
        return positions

    def _load_train_resume_position(self):
        key = self._train_resume_key()
        pos = int(self._resume_positions().get(key, 0) or 0)
        self._set_train_start_byte(pos)

    def _save_train_resume_position(self, pos, total=None):
        if not getattr(self, "_active_train_resume_key", ""):
            return
        value = 0
        if total and pos and pos < total:
            value = int(pos)
        self._resume_positions()[self._active_train_resume_key] = value
        save_state(self.state)
        self._set_train_start_byte(value)

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
            _DEFAULTS = {
                "bands": "20", "base": "1.6180", "hidden": "1536",
                "layers": "3", "lr": "auto 0.001",
                "grad_clip": "1.0", "batch": "512",
                "auto_mode": DEFAULT_AUTO_MODE,
                "decimation": default_decimation_for_mode(DEFAULT_AUTO_MODE),
                "start_byte": "0", "autosave_min": "30",
                "dream_every_batches": "100", "dream_length": "auto 300",
                "dream_temperature": "1.0",
                "save_path": "model.pt",
                "description": "",
            }
            for k, v in _DEFAULTS.items():
                if k in self.train_fields:
                    self.train_fields[k].set(v)
                    self.train_fields[k].set_readonly(False)
                    self._save_form_field("train", k, v)
            self.train_setup_status.config(text="")
            self._load_train_resume_position()
            self._set_train_ckpt_info(
                "new model\n\n"
                "architecture and training settings are editable. "
                "checkpoint id history and weight energy will appear here "
                "after a saved checkpoint is selected.")
            return

        # auto-fill save_path so user saves back to the same file
        self.train_fields["save_path"].set(value)
        self._save_form_field("train", "save_path", value)
        self._load_train_resume_position()

        # load checkpoint metadata on a worker thread
        path = CHECKPOINTS_DIR / value
        if not path.exists():
            self.train_setup_status.config(text=f"not found: {value}")
            return
        self.train_setup_status.config(text=f"reading {value}...")
        self.root.update_idletasks()

        def thread():
            try:
                import torch
                cfg = torch.load(
                    path, map_location='cpu', weights_only=False)
            except Exception as e:
                self.root.after(0, lambda: self.train_setup_status.config(
                    text=f"failed to read: {e}"))
                return

            # populate fields from checkpoint — same values the CLI shows
            vals = {
                "bands": str(cfg.get('n_bands',
                    cfg.get('num_timescales', 32))),
                "base": f"{float(cfg.get('base', 1.6180)):.4f}",
                "hidden": str(cfg.get('hidden_dim', 256)),
                "layers": str(cfg.get('depth',
                    cfg.get('n_stacks', cfg.get('n_layers', 1)))),
                "lr": ("auto "
                       f"{cfg.get('lr_base', cfg.get('lr', 0.001))}"
                       if cfg.get('lr_auto', True)
                       else str(cfg.get('lr', 0.001))),
                "grad_clip": str(cfg.get('grad_clip', 1.0)),
                "auto_mode": str(cfg.get('auto_mode', DEFAULT_AUTO_MODE)),
                "decimation": str(cfg.get(
                    'decimation_range',
                    default_decimation_for_mode(
                        cfg.get('auto_mode', DEFAULT_AUTO_MODE)))),
                "batch": str(cfg.get('batch_size', 10000)),
                "description": str(cfg.get('description', "")),
            }
            seen = cfg.get('bytes_seen', 0)
            ckpt_id = cfg.get('checkpoint_id', '')
            history = cfg.get('checkpoint_history', [])
            try:
                hist_lines = [h[:16] for h in history[-80:]]
            except TypeError:
                hist_lines = []

            energy_lines = []
            net = cfg.get("net") or {}
            first_weight = net.get("layers.0.weight")
            if first_weight is not None:
                try:
                    import torch
                    n_bands = int(vals["bands"])
                    col_energy = first_weight.float().pow(2).mean(dim=0)
                    band_energy = torch.stack([
                        col_energy[i::n_bands].mean()
                        for i in range(n_bands)
                    ])
                    max_energy = float(band_energy.max().item()) or 1.0
                    for i, e in enumerate(band_energy.tolist()):
                        bar = "█" * max(1, round((e / max_energy) * 18))
                        energy_lines.append(f"{i:02d} {bar} {e:.4g}")
                except Exception as e:
                    energy_lines = [f"weight energy unavailable: {e}"]

            # architecture fields are read-only when resuming
            immutable = ("bands", "base", "hidden", "layers")

            def apply():
                for k, v in vals.items():
                    if k in self.train_fields:
                        self.train_fields[k].set(v)
                        self.train_fields[k].set_readonly(k in immutable)
                # nicely-formatted "X.XB seen"
                if seen >= 1_000_000_000:
                    seen_str = f"{seen/1e9:.2f}B seen"
                elif seen >= 1_000_000:
                    seen_str = f"{seen/1e6:.0f}M seen"
                else:
                    seen_str = f"{seen} seen"
                self.train_setup_status.config(
                    text=f"loaded {value} · {seen_str}")
                info = [
                    f"{value}",
                    f"id: {ckpt_id[:24] if ckpt_id else 'not saved'}",
                    f"bytes seen: {seen_str}",
                    "",
                    "description:",
                    vals["description"] or "none",
                    "",
                    "settings:",
                    f"bands {vals['bands']} · base {vals['base']}",
                    f"hidden {vals['hidden']} · depth {vals['layers']}",
                    f"lr {vals['lr']} · gradclip {vals['grad_clip']}",
                    f"batch {vals['batch']} · decimation {vals['decimation']}",
                    f"auto mode {vals['auto_mode']}",
                    "",
                    f"id history ({len(history)}):",
                ]
                info.extend(hist_lines or ["none"])
                info.extend(["", "band energy:"])
                info.extend(energy_lines or ["not available"])
                self._set_train_ckpt_info("\n".join(info))

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
        self.train_live.grid_rowconfigure(1, weight=1)

        content = tk.Frame(self.train_live, bg=BG)
        content.grid(row=0, column=0, sticky="nw", padx=8, pady=8)

        # ── summary block (populated when training starts) ──
        self.train_summary = tk.Label(
            content, text="", font=self.font_dim, bg=BG, fg=DIM,
            anchor="w", justify="left")
        self.train_summary.grid(row=0, column=0, sticky="w", pady=(0, 16))

        # ── headline numbers ──
        head = tk.Frame(content, bg=BG)
        head.grid(row=1, column=0, sticky="w", pady=(0, 8))

        self.train_loss = tk.Label(
            head, text="—", font=self.font_huge, bg=BG, fg=FG,
            anchor="w")
        self.train_loss.pack(side="left")

        self.train_pct = tk.Label(
            head, text="", font=self.font_med, bg=BG, fg=DIM,
            anchor="w")
        self.train_pct.pack(side="left", padx=(16, 0))

        # ── stats row ──
        stats = tk.Frame(content, bg=BG)
        stats.grid(row=2, column=0, sticky="w", pady=(0, 4))

        def stat_pair(parent, label_text):
            """Create a 'label  value' pair, return the value widget."""
            self._label(parent, label_text).pack(side="left", padx=(0, 4))
            val = tk.Label(parent, text="—", font=self.font_med,
                           bg=BG, fg=FG, anchor="w")
            val.pack(side="left", padx=(0, 16))
            return val

        self.cap_nats = stat_pair(stats, "nats")
        self.cap_bpb = stat_pair(stats, "bpb")
        self.cap_acc = stat_pair(stats, "acc")
        self.cap_bps = stat_pair(stats, "b/s")

        # ── session line ──
        session_row = tk.Frame(content, bg=BG)
        session_row.grid(row=3, column=0, sticky="w", pady=(4, 0))

        self._label(session_row, "session b/s").pack(side="left", padx=(0, 4))
        self.cap_life_bps = tk.Label(
            session_row, text="—", font=self.font_dim, bg=BG, fg=DIM)
        self.cap_life_bps.pack(side="left", padx=(0, 16))

        self.cap_seen = tk.Label(
            session_row, text="", font=self.font_dim, bg=BG, fg=DIM)
        self.cap_seen.pack(side="left")

        # ── lr / epoch line ──
        self.cap_lr = tk.Label(
            content, text=" ", font=self.font_dim, bg=BG, fg=DIM,
            anchor="w")
        self.cap_lr.grid(row=4, column=0, sticky="w", pady=(8, 0))

        # ── sparkline ──
        self._label(content, "loss", fg=DIM).grid(
            row=5, column=0, sticky="w", pady=(16, 4))
        self.train_spark = Sparkline(content, width=500, height=80,
                                     capacity=300)
        self.train_spark.grid(row=6, column=0, sticky="w")

        # ── status + stop ──
        bottom = tk.Frame(content, bg=BG)
        bottom.grid(row=7, column=0, sticky="w", pady=(16, 0))

        self.train_stop_btn = RoundedButton(
            bottom, "stop", self._train_stop,
            width=120, height=36, font=self.font_main)
        self.train_stop_btn.pack(side="left")

        self.train_live_status = self._label(bottom, "", fg=DIM)
        self.train_live_status.pack(side="left", padx=(16, 0))

        log_label = self._label(
            self.train_live, "training log", font=self.font_main, fg=FG)
        log_label.grid(row=1, column=0, sticky="sw", padx=8, pady=(12, 4))

        log_frame = tk.Frame(self.train_live, bg=SURFACE)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        self.train_live.grid_rowconfigure(2, weight=1)

        self.train_log_text = tk.Text(
            log_frame, bg=SURFACE, fg=DIM, font=self.font_dim,
            height=7, wrap="word", bd=0, highlightthickness=0,
            padx=14, pady=10, selectbackground=SELECT_BG)
        self.train_log_text.grid(row=0, column=0, sticky="nsew")
        self._bind_text_scroll(self.train_log_text)
        saved_log = self._load_text_file(TRAIN_LOG_FILE)
        if saved_log:
            self.train_log_text.insert("1.0", saved_log)
            self.train_log_text.see("end")
        self.train_log_text.config(state="disabled")

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

    def _train_status(self, message):
        self.train_setup_status.config(text=message)
        try:
            self._train_log(message)
        except Exception:
            pass

    def _set_train_state(self, state):
        self._train_state = state
        if state == 'idle':
            self.train_setup.tkraise()
            self.train_setup_status.config(text="")
            self.train_start_btn.set_text("start")
            self.train_start_btn.set_enabled(True)
        elif state == 'training':
            self.train_live.tkraise()
            self.train_stop_btn.set_text("stop")
            self.train_stop_btn.set_enabled(True)
        elif state == 'stopping':
            self.train_stop_btn.set_text("stopping…")
            self.train_stop_btn.set_enabled(False)
        elif state == 'stopped':
            self.train_setup.tkraise()
            self.train_start_btn.set_text("start")
            self.train_start_btn.set_enabled(True)
            # refresh checkpoint list — saved file may be new
            names = self._list_checkpoint_names_with_none()
            cur = self.train_ckpt_dd.value
            if cur not in names:
                cur = names[0]
            self.train_ckpt_dd.set_options(names, value=cur)
            self._refresh_chat_ckpts()

    def _train_start(self):
        mode = self.train_mode_seg.get()

        # resolve corpus path based on mode
        if mode == "corpus":
            corpus = self.train_corpus_entry.get().strip()
            if not corpus:
                self.train_setup_status.config(
                    text="enter a corpus filename")
                return
            cycle_count = _parse_cycle_spec(corpus)
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

        # resolve checkpoint (resume) path
        ckpt_name = self.train_ckpt_dd.value
        if ckpt_name == "(new model)":
            ckpt_path = ""
        else:
            ckpt_path = str(CHECKPOINTS_DIR / ckpt_name)
        started_new_model = not bool(ckpt_path)

        # resolve save path from its field
        save = self.train_fields["save_path"].get().strip() or "model.pt"
        if not ('/' in save or '\\' in save
                or save.startswith('~') or save.startswith('.')):
            save_path = str(CHECKPOINTS_DIR / save)
        else:
            save_path = os.path.expanduser(save)
        # persist save_path so it survives restarts
        self._save_form_field("train", "save_path", save)

        # read all model params directly from form fields
        def _get(key, default=""):
            if key in self.train_fields:
                return self.train_fields[key].get().strip() or default
            return default

        for key in ("bands", "base", "hidden", "layers", "lr", "grad_clip",
                    "batch", "auto_mode", "decimation",
                    "start_byte", "autosave_min", "dream_every_batches",
                    "dream_length", "dream_temperature", "description"):
            self._save_form_field("train", key, _get(key, ""))

        config = {
            "corpus": str(corpus_path),
            "corpus_label": (
                f"cycle data ×{cycle_count} ({len(cycle_files)} files)"
                if cycle_count is not None else Path(corpus_path).name),
            "cycle_files": [str(p) for p in cycle_files],
            "cycle_count": cycle_count or 1,
            "checkpoint": ckpt_path,
            "bands": _get("bands", "20"),
            "base": _get("base", "1.6180"),
            "hidden": _get("hidden", "1536"),
            "layers": _get("layers", "3"),
            "lr": _get("lr", "auto 0.001"),
            "grad_clip": _get("grad_clip", "1.0"),
            "description": _get("description", ""),
            "batch": _get("batch", "512"),
            "auto_mode": _get("auto_mode", DEFAULT_AUTO_MODE),
            "decimation": _get(
                "decimation",
                default_decimation_for_mode(_get("auto_mode", DEFAULT_AUTO_MODE))),
            "start_byte": _get("start_byte", "0"),
            "autosave_min": _get("autosave_min", "30"),
            "dream_every_batches": _get("dream_every_batches", "100"),
            "dream_length": _get("dream_length", "auto 300"),
            "dream_temperature": _get("dream_temperature", "1.0"),
            "save_path": save_path,
            "loop": loop_mode,
            "head_bytes": (head_mb if loop_mode else 50) * 1_000_000,
            "cycle_pause": 60.0,
        }
        self._train_resume_cursor_enabled = (
            mode == "corpus" and cycle_count is None and not loop_mode)
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

        if not ckpt_path:
            try:
                bands = int(config["bands"])
                hidden = int(config["hidden"])
                layers = int(config["layers"])
            except (TypeError, ValueError):
                self.train_setup_status.config(
                    text="bands, hidden, and layers must be numbers")
                return
            # v12.2 serial backprop: one trace-bank projection, then
            # depth-1 hidden matrices, then byte readout.
            params = (bands * 256 * hidden
                      + max(0, layers - 1) * hidden * hidden
                      + hidden * 256)
            raw_gb = params * 4 / 1_000_000_000
            # mps training needs far more than checkpoint size: weights,
            # update deltas, traces, activations, and matmul temporaries.
            # empirical failure point on an m3 with a 9gb mps cap was a
            # 405m-param model (~1.6gb raw), which needed >9gb at update time.
            est_train_gb = raw_gb * 6.2
            try:
                import torch
                mps_active = torch.backends.mps.is_available()
            except Exception:
                mps_active = False
            if mps_active and est_train_gb > 8.0:
                self.train_setup_status.config(
                    text=(
                        f"~{params/1e6:.0f}m params needs about "
                        f"{est_train_gb:.1f}gb to train on mps; "
                        "lower hidden/layers or use cloud"
                    ))
                return

        # reset live view
        self.train_spark.reset()
        self.train_summary.config(text="")
        self.train_pct.config(text="")
        self.train_loss.config(text="—")
        for w in (self.cap_nats, self.cap_bpb,
                  self.cap_acc, self.cap_bps,
                  self.cap_life_bps):
            w.config(text="—")
        self.cap_lr.config(text=" ")
        self.cap_seen.config(text="")
        self.train_live_status.config(text="starting...")

        self._train_log(
            f"start requested · checkpoint "
            f"{Path(ckpt_path).name if ckpt_path else '(new model)'} · "
            f"save {Path(save_path).name} · "
            f"start {config['start_byte']}")

        if self.train_worker.start(config):
            self._train_started_new_model = started_new_model
            self._train_last_saved_checkpoint = ""
            self._train_discarded_without_save = False
            self._set_train_state('training')
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
            self.stream_setup.tkraise()
        elif state == 'running':
            self.stream_live.tkraise()

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

        tk.Label(top, text="logOS", font=self.font_title, bg=BG, fg=FG,
                 anchor="w").pack(side="left")

        self.logos_refresh_btn = RoundedButton(
            top, "refresh", self._logos_refresh,
            width=100, height=30, font=self.font_dim)
        self.logos_refresh_btn.pack(side="left", padx=(18, 0))

        self.logos_detail_title = tk.Label(
            top, text="select a checkpoint", font=self.font_med,
            bg=BG, fg=FG, anchor="w", justify="left")
        self.logos_detail_title.pack(
            side="left", fill="x", expand=True, padx=(28, 0))

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
            f"bands: {item.get('n_bands')}  hidden: {item.get('hidden_dim')}",
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

        # chat outputs
        for kind, *rest in self.chat.drain():
            if kind == 'char':
                self._chat_append(rest[0], "soma", persist=False)
                self.chat_response_buf.append(rest[0])
                if self._logos_mode:
                    self.logos_response_buf.append(rest[0])
            elif kind == 'status':
                self.chat_status.config(text=rest[0])
            elif kind == 'done':
                response = "".join(self.chat_response_buf)
                self.chat_response_buf.clear()
                self._chat_append("\n\n", "soma", persist=False)
                self._append_text_file(
                    CHAT_LOG_FILE, response + "\n\n",
                    max_bytes=MAX_CHAT_LOG_BYTES)
                self._chat_pending = False
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
                self._chat_append(f"\n  error: {rest[0]}\n\n", "dim")
                self._chat_pending = False
                self.chat_response_buf.clear()
                self.logos_response_buf.clear()

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
                    self._chat_pending = True
                    self.chat.online = self.chat_online_cb.get()
                    try:
                        self.chat.max_length = int(
                            self.chat_maxlen_entry.get())
                    except (ValueError, TypeError):
                        self.chat.max_length = 200
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
                self.logos_cb._redraw()

        # train outputs
        for kind, *rest in self.train_worker.drain():
            if kind == 'start':
                self.train_live_status.config(text="training")
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
                mode = "loop" if d["loop"] else "single"
                lines = [
                    f"corpus: {d['corpus']}  ·  {mode}",
                    f"{d['bands']} bands  ·  base={d['base']:.4f}"
                    f"  ·  {d['hidden']}  ·  {p_str} params",
                    f"device: {d['device']}  ·  {d['dtype']}"
                    f"  ·  batch={d['batch']:,}",
                    f"saving to: {d['save_path']}"
                    f"  ·  {_fmt_bytes(d['bytes_seen'])} seen",
                    f"{d.get('auto_mode', DEFAULT_AUTO_MODE)} · "
                    f"lr {'auto ' if d.get('lr_auto', True) else ''}"
                    f"{d.get('lr_base', d.get('lr', 0.001)):.4g}"
                    f"  ·  decimation {d.get('decimation_range', 12.0):.2f}"
                    f"  ·  gradclip {d.get('grad_clip', 1.0):.4g}",
                ]
                self.train_summary.config(text="\n".join(lines))
                self._train_log(
                    f"loaded {d['save_path']} · {d['corpus']} · {mode}")
            elif kind == 'report':
                d = rest[0]
                pct = (100 * d["pos"] / d["total"]) if d["total"] else 0
                self._train_last_absolute_pos = int(
                    d.get("absolute_pos", d["pos"]))
                self._train_last_absolute_total = int(
                    d.get("absolute_total", d["total"]))
                self.train_loss.config(text=f"{d['loss']:.3f} nats")
                self.train_pct.config(text=f"{pct:.1f}%")

                self.cap_nats.config(text=f"{d['loss']:.3f}")
                self.cap_bpb.config(text=f"{d['bpb']:.2f}")
                self.cap_acc.config(text=f"{d['acc']:.1f}%")
                self.cap_bps.config(text=f"{d['bps']:,.0f}")
                self.cap_life_bps.config(text=f"{d['life_bps']:,.0f}")

                sampled_stride = int(d.get(
                    "sampled_stride", d.get("stride", 1)))
                base_stride = int(d.get("stride", 1))
                stride_text = str(base_stride)
                if sampled_stride != base_stride:
                    stride_text = f"{sampled_stride}/{base_stride}"
                mode_name = d.get('auto_mode', DEFAULT_AUTO_MODE)
                control_name = (
                    'motor' if mode_name == 'model'
                    else ('wallclock' if mode_name == 'wallclock' else 'io2'))
                control_value = (
                    d.get('motor_delta', 0.0) if mode_name == 'model'
                    else (d.get('sampled_stride_float', d.get('stride', 1))
                          if mode_name == 'wallclock' else d.get('io2', 0.0)))
                self.cap_lr.config(
                    text=f"lr {d['lr']:.4f}  ·  "
                         f"{control_name} {control_value:+.3f}  ·  "
                         f"decimation {d.get('decimation_band', 0.0):.2f} "
                         f"(stride {stride_text})  ·  "
                         f"rowclip {100*d.get('row_clip', 0.0):.1f}%  ·  "
                         f"gradclip {d['grad_clip']:.4f}  ·  "
                         f"epoch {d['epoch']}/{d['epochs']}")
                self.cap_seen.config(
                    text=f"{_fmt_bytes(d['life_bytes'])} "
                         f"in {_fmt_duration(d.get('life_seconds', 0))}")
                self.train_spark.push(d["loss"])
            elif kind == 'autosaved':
                # quiet, periodic, just shows the user the system is
                # alive and saving without their input
                t = time.strftime("%H:%M:%S")
                self.train_live_status.config(
                    text=f"autosaved · {t}")
                self._train_log(f"autosaved {Path(rest[0]).name}")
                self._refresh_chat_ckpts()
            elif kind == 'dream':
                d = rest[0]
                seen = _fmt_bytes(d["bytes_seen"])
                self._train_log(
                    f"dream at batch {d['batch']} · {seen} seen\n"
                    f"> {d['text']}")
            elif kind == 'checkpoint_saved':
                self.train_live_status.config(
                    text=f"saved {Path(rest[0]).name}")
                self._train_log(f"saved {Path(rest[0]).name}")
                self._train_last_saved_checkpoint = rest[0]
                self._refresh_chat_ckpts()
            elif kind == 'info':
                self.train_live_status.config(text=rest[0])
                self._train_log(rest[0])
                if rest[0] == 'stopped without saving':
                    self._train_discarded_without_save = True
            elif kind == 'error':
                self.train_live_status.config(text=f"error: {rest[0]}")
                self._train_log(f"error: {rest[0]}")
            elif kind == 'done':
                self._train_log("training worker stopped")
                if (self._train_resume_cursor_enabled
                        and not self._train_discarded_without_save):
                    self._save_train_resume_position(
                        self._train_last_absolute_pos,
                        self._train_last_absolute_total)
                self._train_discarded_without_save = False
                self._set_train_state('stopped')
                self._select_train_checkpoint_after_new_model_save()

        self.root.after(80, self._poll)

    # ── lifecycle ──

    def _on_close(self):
        self._closing = True
        # If chat has changed runtime state, ask before implicitly saving it.
        if self.chat.model is not None:
            path = getattr(self, '_chat_loaded_path', None)
            save_on_close = True
            if getattr(self.chat, 'dirty', False):
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
            if path and save_on_close:
                try:
                    self._restore_chat_training_config()
                    self.chat.model.save(path)
                    self.chat.dirty = False
                except Exception as e:
                    print(f"chat save on quit failed: {e}", file=sys.stderr)

        # stop the stream subprocess (its sigint handler saves cleanly)
        self.stream_proc.stop()

        # stop the logOS bridge
        self.logos_bridge.stop()

        # stop the train worker — ask whether to save its final state
        if self.train_worker.is_running():
            choice = messagebox.askyesnocancel(
                "quit soma",
                "save training checkpoint before quitting?\n\n"
                "yes: save & quit\n"
                "no: quit without saving\n"
                "cancel: keep running",
                parent=self.root,
            )
            if choice is None:
                self._closing = False
                return
            self.train_worker.stop(save=bool(choice))
            # wait a moment for the worker to flush its final save
            import time as _t
            for _ in range(20):  # up to 2s
                if not self.train_worker.is_running():
                    break
                _t.sleep(0.1)
            if self.train_worker.is_running():
                self.train_worker.proc.stop()

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

"""strictly local speech input and output for the macos soma app."""

import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path


_SENTENCE_BOUNDARY = re.compile(
    r"(?s)^(.+?(?:[.!?]+[\"')\]]*\s+|\n+))(.*)$")


def split_speakable(buffer):
    """return complete spoken phrases and the unfinished tail."""
    phrases = []
    while True:
        match = _SENTENCE_BOUNDARY.match(buffer)
        if not match:
            break
        phrase, buffer = match.groups()
        phrase = phrase.strip()
        if phrase:
            phrases.append(phrase)
    return phrases, buffer


def speech_helper_path():
    roots = [Path(__file__).resolve().parent]
    if getattr(sys, "frozen", False):
        roots.extend((roots[0].parent / "Frameworks", roots[0].parent / "MacOS"))
    roots.append(Path(__file__).resolve().parent / "macos" / "bin")
    for root in roots:
        candidate = root / "soma-speech"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


class OnDeviceRecognizer:
    def __init__(self):
        self.outbox = queue.Queue()
        self.proc = None
        self._thread = None
        self._lock = threading.Lock()

    @property
    def listening(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, locale="auto"):
        previous = self._thread
        self.cancel()
        if previous is not None and previous.is_alive():
            previous.join(timeout=0.5)
        self.drain()
        helper = speech_helper_path()
        if helper is None:
            self.outbox.put(("error", "local speech helper is not installed"))
            return False
        try:
            proc = subprocess.Popen(
                [str(helper), str(locale or "auto")],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)
        except OSError as error:
            self.outbox.put(("error", f"local speech could not start: {error}"))
            return False
        with self._lock:
            self.proc = proc
        self._thread = threading.Thread(
            target=self._read, args=(proc,), daemon=True)
        self._thread.start()
        return True

    def _read(self, proc):
        try:
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                    kind = str(event.get("kind", "error"))
                    text = str(event.get("text", ""))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    kind, text = "error", "invalid response from local speech"
                self.outbox.put((kind, text))
        finally:
            stderr = proc.stderr.read().strip() if proc.stderr else ""
            proc.wait()
            with self._lock:
                if self.proc is proc:
                    self.proc = None
            if proc.returncode and stderr:
                self.outbox.put(("error", stderr.splitlines()[-1]))
            self.outbox.put(("stopped", ""))

    def stop(self):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError, AttributeError):
            pass

    def cancel(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        with self._lock:
            if self.proc is proc:
                self.proc = None

    def drain(self, limit=100):
        events = []
        while len(events) < limit:
            try:
                events.append(self.outbox.get_nowait())
            except queue.Empty:
                break
        return events


class LocalSpeaker:
    def __init__(self):
        self.enabled = False
        self._buffer = ""
        self._queue = queue.Queue()
        self._proc = None
        self._epoch = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def begin(self):
        self.stop()
        self._buffer = ""

    def feed(self, text):
        if not self.enabled or not text:
            return
        self._buffer += text
        phrases, self._buffer = split_speakable(self._buffer)
        for phrase in phrases:
            self._queue.put((self._epoch, phrase))

    def finish(self):
        if self.enabled:
            phrase = self._buffer.strip()
            if phrase:
                self._queue.put((self._epoch, phrase))
        self._buffer = ""

    def stop(self):
        self._buffer = ""
        with self._lock:
            self._epoch += 1
            proc = self._proc
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if proc is not None and proc.poll() is None:
            proc.terminate()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            epoch, phrase = item
            if not self.enabled:
                continue
            proc = None
            try:
                with self._lock:
                    if epoch != self._epoch:
                        continue
                    proc = subprocess.Popen(
                        ["/usr/bin/say", phrase], stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL)
                    self._proc = proc
                proc.wait()
            except OSError:
                pass
            finally:
                with self._lock:
                    if proc is not None and self._proc is proc:
                        self._proc = None

    def shutdown(self):
        self.stop()
        self._queue.put(None)

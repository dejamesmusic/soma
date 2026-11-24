"""Many seats on one model at once (Nov 24; LANES.md): a chat in a lane per active seat. The chat runtime's prompt, file
reads and reply, for many chats, on the lane engine's clock: every step, one learning round (lanes whose tick filled)
and one batched decode step for every chat reading its prompt or writing its reply (lanes_decode). A chat is exactly
what SOMA's session_text / session_file + session_reply + generate would make of it alone: the same tokens into its
stream, the same reading context, the same samples from the same distributions (one chat: identical to the single-stream
runtime, tests/lane_chats.py).

  chats = LaneChats(soma, lanes=24)
  chats.attach("ch_1", state)                  # a chat's saved context (the chat bridge's format), or None
  chats.prompt("ch_1", parts=[{"text": "hi"}, {"file": "/data/notes.md"}], learn=True, length=200, temperature=0.8)
  while chats.busy(): for chat, kind, payload in chats.step(): ...
        # kinds: delta (reply text), done, read (a file's progress), read_done, paused, resumed
  chats.pause("ch_1"); chats.resume("ch_1"); chats.skip("ch_1"); chats.cancel("ch_1")
  state = chats.detach("ch_1")

A file is read as the single runtime reads it: <|soma_state|>, then its bytes in the stream's ticks (no token-by-token
reading context while it streams; the context is rebuilt from the stream when the file ends). Its ticks are learned in
the shared rounds beside every other lane's.

The attention window a reply reads was computed position by position as its tokens arrived, as training's cached window
is: other lanes' learning moves the shared weights between those positions, and the window keeps the keys each position
had when it was read (the cache is rebuilt when this chat's own tick is learned, as in the single stream).
"""
import dataclasses
import io
import json
import threading
import time
from pathlib import Path

import numpy as np

from .data import encode_bytes, read_segment
from .lane_engine import LaneEngine
from .lanes_decode import decode_lanes

SESSION_FILE = "soma_state"
TRACES_SUFFIX = ".traces.jsonl"                           # 1.5.3: recorded turns, read as turns (SOMA._session_traces)
FORBIDDEN = ("imessage", "mytext")


@dataclasses.dataclass
class FilePart:
    path: Path
    size: int
    handle: object = None
    pos: int = 0
    bits: float = 0.0                                      # the read's ticks: bits and bytes (for its bpb)
    bytes: float = 0.0
    tokens: int = 0
    win_bits: float = 0.0                                  # since the last report
    win_bytes: float = 0.0
    started: float = 0.0
    reported: float = 0.0
    eof: bool = False
    traces: bool = False
    quiet: bool = False                # a skill's answer: read like a file, no read events (as _read_bytes)
    data: bytes = None


@dataclasses.dataclass
class Job:
    parts: list                        # ("ids", [tokens]) to read and join, ("file", FilePart) to stream in
    learn: bool
    length: int
    temperature: float
    report_seconds: float = 0.5
    backlog: list = dataclasses.field(default_factory=list)   # in the stream already; still to read into a rebuilt context
    made: int = 0
    cancelled: bool = False
    paused: bool = False
    told_paused: bool = False
    skip: bool = False
    rebuild: bool = False              # the reading context is rebuilt from the stream before the next read
    decoder: object = None
    skills: object = None              # this chat's search and run (runtime skills.py), when on
    asks: int = 0
    written: list = dataclasses.field(default_factory=list)
    asking: object = None              # a request running: {"answer": bytes or None}


class LaneChats:
    def __init__(self, soma, lanes: int):
        self.soma, self.E = soma, LaneEngine(soma, lanes)
        self.E.bank_cores = [ln.core for ln in self.E.lanes]
        self.tok = soma.tokenizer
        self.where = {}                                    # chat -> lane
        self.jobs = {}                                     # lane -> Job
        self.forbidden = list(self.tok.generation_forbidden_ids)
        self.turn = self.tok.control_ids["turn"]
        self.asks_file = self.tok.control_ids[SESSION_FILE]
        self.skill_sets = {}                               # chat -> Skills (links read in one chat aren't another's)

    # ---- chats in and out of lanes ----
    def attach(self, chat, state=None) -> int:
        lane = self.E.attach(chat, state)
        self.where[chat] = lane
        return lane

    def detach(self, chat) -> dict:
        self.skill_sets.pop(chat, None)
        lane = self.where.pop(chat)
        assert lane not in self.jobs, "the chat is mid-answer: cancel it first"
        return self.E.detach(lane)

    def state(self, chat) -> dict:
        return self.E.state(self.where[chat])

    # ---- a prompt: text and files, then the reply ----
    def prompt(self, chat, text=None, parts=None, learn=True, length=200, temperature=0.8, report_seconds=0.5,
               skills=None, on_event=None):
        """skills: {"search": bool, "run": bool} (1.5.0), with on_event(kind, payload) hearing them as they happen"""
        lane = self.where[chat]
        assert lane not in self.jobs, "one prompt at a time per chat"
        parts = parts if parts is not None else [{"text": text}]
        sk = None
        if skills and (skills.get("search") or skills.get("run")):
            try:
                from .skills import Skills
            except ImportError:                            # a runtime without skills
                Skills = None
            if Skills is not None:
                sk = self.skill_sets.get(chat)
                if sk is None:
                    sk = self.skill_sets[chat] = Skills({}, on_event)
                sk.on_event = on_event or sk.on_event
                sk.settings.update({"search": bool(skills.get("search")), "run": bool(skills.get("run"))})
        out = []
        for p in parts:
            if p.get("file"):
                path = Path(str(p["file"])).expanduser().resolve()
                assert not any(w in path.name.lower() for w in FORBIDDEN), "never read iMessage-derived data"
                traces = path.name.endswith(TRACES_SUFFIX) and hasattr(self.soma, "_trace_ids")
                if not traces:                             # a file opens with its marker; recorded turns carry their own
                    out.append(("ids", [self.tok.control_ids[SESSION_FILE]]))
                out.append(("file", FilePart(path=path, size=path.stat().st_size, traces=traces)))
            elif str(p.get("text", "")).strip():
                body = list(encode_bytes(self.tok, str(p["text"]).encode("utf-8")))
                out.append(("ids", [self.tok.control_ids["user"]] + body))
                if sk is not None:
                    sk.observe(str(p["text"]))
        out.append(("ids", [self.tok.control_ids["soma"]]))
        if sk is not None:
            sk.new_turn()
        self.jobs[lane] = Job(parts=out, learn=bool(learn), length=int(length), temperature=float(temperature),
                              report_seconds=float(report_seconds), decoder=self.tok.incremental_decoder(), skills=sk)

    def _job(self, chat):
        lane = self.where.get(chat)
        return self.jobs.get(lane) if lane is not None else None

    def cancel(self, chat):
        j = self._job(chat)
        if j is not None:
            j.cancelled, j.paused = True, False

    def pause(self, chat):
        j = self._job(chat)
        if j is not None and j.parts and j.parts[0][0] == "file":
            j.paused = True

    def resume(self, chat):
        j = self._job(chat)
        if j is not None:
            j.paused = False

    def skip(self, chat):
        j = self._job(chat)
        if j is not None:
            j.skip, j.paused = True, False

    def forget(self, chat) -> None:
        """links read in one chat aren't another's to follow (a chat's skills start over)"""
        sk = self.skill_sets.get(chat)
        if sk is not None:
            sk.forget()

    def busy(self) -> bool:
        return bool(self.jobs)

    def reading(self, chat):
        """the file this chat is reading now, or None"""
        j = self._job(chat)
        return j.parts[0][1].path.name if j is not None and j.parts and j.parts[0][0] == "file" else None

    # ---- one step of the clock ----
    def _join(self, lane, token, learn):
        ln = self.E.lanes[lane]
        ln.pending = np.concatenate([ln.pending, np.asarray([token], np.int64)])
        ln.learn = learn

    def _rebuild(self, lane, job):
        """the reading context from the learned stream; the unlearned tail is read into it (batched, in decode steps)"""
        ln = self.E.lanes[lane]
        ln.chat_ctx = ln.stream_ctx.clone()
        ln.synced = True
        job.backlog = [int(t) for t in ln.pending] + job.backlog
        job.rebuild = False

    def _file_done(self, lane, job, events):
        part = job.parts.pop(0)[1]
        if part.handle is not None:
            part.handle.close()
        chat = self.E.lanes[lane].chat
        if part.quiet:                                     # a skill's answer: no read events
            job.rebuild = True
            return
        events.append((chat, "read", {"file": str(part.path), "bytes": part.size, "report": self._report(part, True)}))
        events.append((chat, "read_done", {"file": str(part.path), "bytes": part.size, "read": part.pos,
                                           "bpb": part.bits / part.bytes if part.bytes else None,
                                           "seconds": time.time() - part.started, "tokens": part.tokens,
                                           "learned": job.learn, "stopped": job.skip or job.cancelled}))
        job.skip = False
        job.rebuild = True                                 # as session_file ends: the context comes from the stream

    def _report(self, part, final=False):
        now = time.time()
        dt = max(1e-6, now - (part.reported or part.started))
        r = {"pos": part.pos, "total": part.size, "bpb": part.win_bits / part.win_bytes if part.win_bytes else None,
             "session_bpb": part.bits / part.bytes if part.bytes else None, "bps": part.win_bytes / dt}
        part.reported, part.win_bits, part.win_bytes = now, 0.0, 0.0
        return r

    def step(self) -> list:
        events = []
        E, C = self.E, self.E.C
        # 1. files stream into their lanes' ticks (a lane reads up to a whole tick ahead of what's learned)
        for lane, job in list(self.jobs.items()):
            if not job.parts or job.parts[0][0] != "file":
                continue
            ln, part = E.lanes[lane], job.parts[0][1]
            if job.paused:
                if not job.told_paused:
                    events.append((ln.chat, "paused", {"file": str(part.path)}))
                    job.told_paused = True
                continue
            if job.told_paused:
                events.append((ln.chat, "resumed", {"file": str(part.path)}))
                job.told_paused = False
            if part.handle is None:
                part.handle = io.BytesIO(part.data) if part.data is not None else part.path.open("rb")
                part.started = time.time()
                if not part.quiet:
                    events.append((ln.chat, "status", f"reading {part.path.name}" + ("" if job.learn else " (no weight updates)")))
            while len(ln.pending) < C and not part.eof and not (job.skip or job.cancelled):
                raw = part.handle.readline() if part.traces else read_segment(part.handle, part.size, 4096)
                if not raw:
                    part.eof = True
                    break
                part.pos += len(raw)
                if part.traces:                            # one recorded turn, as the stream holds it
                    try:
                        ids = np.asarray(self.soma._trace_ids(json.loads(raw)), np.int64)
                    except (ValueError, AttributeError, TypeError):
                        continue                           # not a turn: skipped
                    if not len(ids):
                        continue
                else:
                    ids = np.asarray(encode_bytes(self.tok, raw), np.int64)
                    if job.skills is not None and not part.quiet:
                        job.skills.observe_bytes(raw)
                ln.pending = np.concatenate([ln.pending, ids])
                ln.learn, ln.synced = job.learn, False
            if part.eof or job.skip or job.cancelled:
                self._file_done(lane, job, events)
        for lane, job in self.jobs.items():                # a skill's answer has come: read it, then its turn goes on
            if job.asking is not None and job.asking.get("done"):
                data = job.asking.get("answer") or b""
                job.asking = None
                job.parts = [("ids", [self.asks_file]), ("file", FilePart(path=Path("answer"), size=len(data), data=data, quiet=True)),
                             ("ids", [self.tok.control_ids["soma"]])] + job.parts
        # 2. every lane whose tick filled learns it (one batched round)
        learned = E.ready()
        if learned:
            bits = E.round()
            for lane in learned:
                job = self.jobs.get(lane)
                ln = E.lanes[lane]
                if job is not None and job.parts and job.parts[0][0] == "file":
                    part = job.parts[0][1]
                    b, nb = float(bits[lane].sum()), float(self.soma.blen[ln.last_tick].sum())
                    part.bits += b; part.bytes += nb; part.win_bits += b; part.win_bytes += nb; part.tokens += C
                    if not part.quiet and time.time() - part.reported >= job.report_seconds:
                        events.append((ln.chat, "read", {"file": str(part.path), "bytes": part.size, "report": self._report(part)}))
                    continue                               # mid-file: its context is rebuilt when the file ends
                if job is not None:
                    self._rebuild(lane, job)
                else:
                    ln.chat_ctx = ln.stream_ctx.clone()
                    ln.synced = True
                    if len(ln.pending):                    # an idle chat's tail: read now, alone (rare)
                        getattr(ln.core, "read", ln.core.observe)(ln.chat_ctx, ln.pending)
        for lane, job in self.jobs.items():                # a file has ended and its last tick is learned (or waits)
            if job.rebuild and len(E.lanes[lane].pending) < C:
                self._rebuild(lane, job)
        # 3. one decode step for every chat reading or writing whose tick isn't waiting to be learned
        active = [lane for lane, job in self.jobs.items()
                  if len(E.lanes[lane].pending) < C and not job.paused and not job.rebuild and job.asking is None
                  and not (job.parts and job.parts[0][0] == "file")]
        if not active:
            return events
        ps = decode_lanes([E.lanes[l].core for l in active], [E.lanes[l].chat_ctx for l in active])
        for lane, p in zip(active, ps):
            ln, job = E.lanes[lane], self.jobs[lane]
            chat = ln.chat
            if job.backlog:                                # a rebuilt context catching up with its stream
                ln.core.observe(ln.chat_ctx, [job.backlog.pop(0)])
                continue
            if job.cancelled and job.parts:                # cancelled before its reply began: no turn is written
                events.append((chat, "done", {"cancelled": True, "tokens": 0}))
                del self.jobs[lane]
                continue
            if job.parts:                                  # the prompt's words (and markers), read and joined
                ids = job.parts[0][1]
                t = ids.pop(0)
                if not ids:
                    job.parts.pop(0)
                ln.core.observe(ln.chat_ctx, [t])
                self._join(lane, t, job.learn)
                continue
            # the reply. Skills (1.5.0): where its most likely next token opens a file, it has asked for one: what it
            # wrote since its turn began (or its last file) goes to search or run, on its own thread, and the answer is
            # read in as a file (the lane waits; the clock doesn't). At most 16 asks a turn; files don't count
            if (job.skills is not None and not job.cancelled and job.asks < 16 and job.made < job.length
                    and int(p.argmax()) == self.asks_file):
                job.asks += 1
                ask, job.written = self.tok.decode(job.written), []
                job.asking = box = {"done": False}
                def go(sk=job.skills, ask=ask, box=box):
                    try:
                        box["answer"] = str(sk.request(ask)).encode("utf-8")
                    except Exception as error:             # noqa: BLE001 - a failed ask is read as its error
                        box["answer"] = f"(the request failed: {type(error).__name__})".encode()
                    box["done"] = True
                threading.Thread(target=go, daemon=True).start()
                continue
            if job.cancelled or job.made >= job.length:
                t = self.turn                              # the turn closes (as generate's finally does)
            else:
                t = self.soma._sample(p, job.temperature, self.forbidden)
                job.made += 1
            ln.core.observe(ln.chat_ctx, [t])
            self._join(lane, t, job.learn)
            if t == self.turn:
                text = "".join(job.decoder.finish())
                if text:
                    events.append((chat, "delta", text))
                events.append((chat, "done", {"cancelled": job.cancelled, "tokens": job.made}))
                del self.jobs[lane]
            else:
                job.written.append(t)
                text = "".join(job.decoder.push(t))
                if text:
                    events.append((chat, "delta", text))
        return events

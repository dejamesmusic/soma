"""skills (somatic core 1.5.0, James, Nov 22): a model's only two skills. Search is where to look; run is what to do.

A model asks for a file: while it writes a turn, <|soma_state|> (the control that opens a file) is its most likely next
token. The runtime hands what it wrote since its turn began (or since its last file) to Skills.request, and reads
whatever comes back into the stream as a file. What it wrote decides what it gets:

    a url it has already seen        the page, as text, with its links (fetch: a crawler that only follows links)
    a code block at its end          what the code printed, run in a sandbox with no network (run)
    anything else                    search results for its last line (search)

The safeguards live here, beside the model, wherever it runs (a computer or a gpu); none of it passes through logOS.

    fetch    GET only; http and https; public addresses only (never local, private or cloud-metadata addresses, checked
             on the address actually connected to, every redirect); robots.txt honoured; one request a second per site;
             text only, ~200 KB of it; only urls the model has read (typed to it, in a file, a page or results), so it
             can't carry a chat out inside a url it made up
    search   wikipedia's search api (no key); its results' urls become links it has read
    run      python compiled to webassembly (sandbox/ in this runtime, when installed): no network, a scratch folder
             emptied after every run, 10 s, 256 MB, 20 KB of output
    caps     per turn: 3 fetches or searches, 10 runs; per hour: 30 fetches or searches (settable by the engine)

Every request gets a file back, refusals included ("only links it has read"), so a model learns the form from its
answers. The engine hears each one (on_event) and shows it in the chat.
"""
from __future__ import annotations

import html
import html.parser
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import ssl
import tempfile
import threading
import time
import urllib.parse
import urllib.robotparser
from collections import OrderedDict, deque
from pathlib import Path
from typing import Callable

UA = "soma/1.5.3 (somatic core; +https://logossoma.com/about)"
ROBOT_NAME = "soma"
URL_RE = re.compile(r"https?://[^\s<>\"'`|\\^{}]+")
FENCE_RE = re.compile(r"```[ \t]*(?:python|py)?[ \t]*\n(.*?)```\s*$", re.S | re.I)
PAGE_CHARS = 200_000
DOWNLOAD_BYTES = 2_000_000
LINKS_MAX = 60
TIMEOUT = 15.0
REDIRECTS = 5
SEEN_MAX = 20_000
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "application/rss", "application/atom")
SANDBOX = Path(__file__).resolve().parent / "sandbox"
MEDIA_MAX = 40
EXCERPT_CHARS = 3000
STARTUP_NOISE = ("Could not find platform dependent libraries", "WARN: Could not find the platform standard library")


class Refusal(Exception):
    """a request that gets a note back instead of a page"""


# ------------------------------------------------------------------ the crawler's connection: public addresses only

def public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%", 1)[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast and not addr.is_reserved


def _address(host: str, port: int, allow_private: bool) -> str:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        raise Refusal(f"{host} doesn't resolve")
    ips = [i[4][0] for i in infos]
    if not ips:
        raise Refusal(f"{host} doesn't resolve")
    if not allow_private and not all(public(ip) for ip in ips):
        raise Refusal(f"{host} isn't on the public internet")
    return ips[0]


class _HTTP(http.client.HTTPConnection):
    def __init__(self, host, port, ip, timeout):
        super().__init__(host, port, timeout=timeout)
        self._ip = ip

    def connect(self):                                   # to the address that was checked, never a fresh lookup
        self.sock = socket.create_connection((self._ip, self.port), self.timeout)


class _HTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, ip, timeout, context):
        super().__init__(host, port, timeout=timeout, context=context)
        self._ip = ip

    def connect(self):
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def _tls() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:                                    # noqa: BLE001 - the system's own
        return ssl.create_default_context()


def http_get(url: str, *, allow_private: bool = False, limit: int = DOWNLOAD_BYTES, accept: str = "") -> tuple:
    """GET a url, following redirects, each hop checked -> (final url, content type, body bytes)"""
    for _ in range(REDIRECTS + 1):
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise Refusal("only http and https links")
        if parts.username or parts.password:
            raise Refusal("no links with passwords in them")
        host, https = parts.hostname, parts.scheme == "https"
        port = parts.port or (443 if https else 80)
        ip = _address(host, port, allow_private)
        conn = _HTTPS(host, port, ip, TIMEOUT, _tls()) if https else _HTTP(host, port, ip, TIMEOUT)
        path = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        try:
            conn.request("GET", path, headers={"User-Agent": UA, "Accept-Encoding": "identity",
                                               "Accept": accept or "text/html,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.1"})
            r = conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and r.getheader("Location"):
                url = urllib.parse.urljoin(url, r.getheader("Location"))
                continue
            if r.status >= 400:
                raise Refusal(f"{host} answered {r.status}")
            data = r.read(limit + 1)
            return url, (r.getheader("Content-Type") or "").lower(), data[:limit]
        except (OSError, http.client.HTTPException, ssl.SSLError) as error:
            raise Refusal(f"{host} couldn't be read ({type(error).__name__})")
        finally:
            conn.close()
    raise Refusal("too many redirects")


# ------------------------------------------------------------------ pages as text

class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "template", "iframe", "head", "nav", "footer", "form", "button", "select"}
    BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section", "article",
             "blockquote", "pre", "hr", "dd", "dt", "header", "main", "figcaption"}

    def __init__(self, base: str):
        super().__init__(convert_charrefs=True)
        self.base, self.out, self.links, self.title = base, [], [], ""
        self.skip, self.in_title = 0, False
        self.main, self.main_out, self.main_links = 0, [], []      # a page's <main> or <article>, when it has one
        self.media: list = []                                          # 1.5.2: images and videos, where they appear
        self.in_video = None

    def _put(self, s: str) -> None:
        self.out.append(s)
        if self.main:
            self.main_out.append(s)

    def _media(self, kind: str, src: str, alt: str) -> None:
        """an image or a video, in the text where it appears: [image: alt](url). the model reads the alt text there;
        the chat shows the picture there"""
        src = urllib.parse.urljoin(self.base, (src or "").strip())
        if not src.startswith(("http://", "https://")) or len(self.media) >= MEDIA_MAX or any(m["src"] == src for m in self.media):
            return
        alt = re.sub(r"\s+", " ", alt or "").strip()[:200]
        self.media.append({"kind": kind, "src": src, "alt": alt})
        self._put(f"\n[{kind}: {alt or kind}]({src})\n")

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("main", "article") or a.get("role") == "main":
            self.main += 1
        if not self.skip:
            if tag == "img":
                small = any((a.get(k) or "").isdigit() and int(a[k]) < 40 for k in ("width", "height"))
                src = a.get("src") or a.get("data-src") or (a.get("srcset") or "").split(" ")[0]
                if src and not small and not src.startswith("data:"):
                    self._media("image", src, a.get("alt") or a.get("title") or "")
            elif tag == "video":
                self.in_video = {"title": a.get("title") or a.get("aria-label") or "", "got": bool(a.get("src"))}
                if a.get("src"):
                    self._media("video", a["src"], self.in_video["title"])
            elif tag == "source" and self.in_video is not None and not self.in_video["got"] and a.get("src"):
                self.in_video["got"] = True                      # one video per <video>: its first source
                self._media("video", a["src"], self.in_video["title"])
            elif tag == "iframe" and re.match(r"https?://(www\.)?(youtube(-nocookie)?\.com/embed/|player\.vimeo\.com/video/)", urllib.parse.urljoin(self.base, a.get("src") or "")):
                self._media("video", a.get("src") or "", a.get("title") or "")
        if tag in self.SKIP:
            self.skip += 1
        if tag == "title":
            self.in_title = True
        if tag in self.BLOCK:
            self.out.append("\n")
            if self.main:
                self.main_out.append("\n")
        if tag == "li":
            self.out.append("- ")
            if self.main:
                self.main_out.append("- ")
        if tag == "a" and not self.skip:
            href = dict(attrs).get("href") or ""
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                link = urllib.parse.urldefrag(urllib.parse.urljoin(self.base, href))[0]
                if link.startswith(("http://", "https://")) and link not in self.links:
                    self.links.append(link)
                if link.startswith(("http://", "https://")) and self.main and link not in self.main_links:
                    self.main_links.append(link)

    def handle_endtag(self, tag):
        if tag == "video":
            self.in_video = None
        if tag in ("main", "article") and self.main:
            self.main -= 1
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "title":
            self.in_title = False
        if tag in self.BLOCK:
            self.out.append("\n")
            if self.main:
                self.main_out.append("\n")

    def handle_data(self, data):
        if self.in_title:
            self.title += data
        elif not self.skip:
            self.out.append(data)
            if self.main:
                self.main_out.append(data)


def page_text(url: str, ctype: str, data: bytes) -> tuple:
    """-> (text, links, media)"""
    charset = (re.search(r"charset=([\w-]+)", ctype) or [None, "utf-8"])[1]
    try:
        raw = data.decode(charset, errors="replace")
    except LookupError:
        raw = data.decode("utf-8", errors="replace")
    if "html" not in ctype and not raw.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
        return raw[:PAGE_CHARS], [u for u in dict.fromkeys(URL_RE.findall(raw))][:LINKS_MAX], []
    p = _Text(url)
    try:
        p.feed(raw)
        p.close()
    except Exception:                                    # noqa: BLE001 - a broken page: what was read so far
        pass
    body = p.main_out if len("".join(p.main_out).strip()) > 200 else p.out      # the page's main part, when it has one
    links = p.main_links if body is p.main_out and p.main_links else p.links
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(body))
    text = re.sub(r"(?m)^\s*-\s*$", "", text)                                    # empty list items (menus, icons)
    text = re.sub(r"\n\s*\n+", "\n\n", "\n".join(line.strip() for line in text.split("\n"))).strip().replace("\n\n- ", "\n- ")
    head = (html.unescape(p.title.strip()) + "\n") if p.title.strip() else ""
    host = urllib.parse.urlsplit(url).hostname or ""
    links = sorted(links, key=lambda u: (urllib.parse.urlsplit(u).hostname or "") != host)   # the site's own first
    text = (head + text)[:PAGE_CHARS]
    media = [{k: v for k, v in m.items() if not k.startswith("_")} for m in p.media if m["src"] in text]
    return text, links[:LINKS_MAX], media


# ------------------------------------------------------------------ run: python in webassembly

def sandbox_ready() -> bool:
    return (SANDBOX / "python.wasm").is_file() and (SANDBOX / "lib" / "python315.zip").is_file() and _wasmtime() is not None


def _wasmtime():
    try:
        import wasmtime
        return wasmtime
    except Exception:                                    # noqa: BLE001
        return None


_MODULE: dict = {}


def run_python(code: str, *, seconds: float = 10.0, memory: int = 256 * 1024 ** 2, output: int = 20_000) -> str:
    """code in the sandbox: python compiled to webassembly, no network (wasi gives it none), only a scratch folder
    (emptied after) and the standard library (read-only). -> what it printed, then any error"""
    wt = _wasmtime()
    if not sandbox_ready() or wt is None:
        raise Refusal("run isn't installed on this machine yet")
    if "engine" not in _MODULE:
        cfg = wt.Config()
        cfg.epoch_interruption = True
        engine = wt.Engine(cfg)
        # compiled once per process, in memory: a precompiled module is native code, so none is ever read from disk
        module = wt.Module.from_file(engine, str(SANDBOX / "python.wasm"))
        _MODULE.update(engine=engine, module=module)
    engine, module = _MODULE["engine"], _MODULE["module"]
    scratch = Path(tempfile.mkdtemp(prefix="soma-run-"))
    try:
        (scratch / "main.py").write_text(code, encoding="utf-8")
        out, err = scratch / ".out", scratch / ".err"
        wasi = wt.WasiConfig()
        wasi.argv = ["python", "-I", "-B", "/work/main.py"]
        wasi.env = [["PYTHONHOME", "/"], ["PYTHONDONTWRITEBYTECODE", "1"], ["HOME", "/work"]]   # stdlib: /lib/python315.zip
        wasi.preopen_dir(str(SANDBOX / "lib"), "/lib", False)               # the standard library, read-only
        wasi.preopen_dir(str(scratch), "/work", True)                      # its scratch folder, emptied after
        wasi.stdout_file, wasi.stderr_file = str(out), str(err)
        store = wt.Store(engine)
        store.set_limits(memory_size=memory)
        store.set_wasi(wasi)
        store.set_epoch_deadline(1)
        linker = wt.Linker(engine)
        linker.define_wasi()
        timer = threading.Timer(seconds, engine.increment_epoch)
        timer.start()
        note = ""
        try:
            linker.instantiate(store, module).exports(store)["_start"](store)
        except wt.ExitTrap as e:
            note = "" if e.code == 0 else f"\n(exit {e.code})"
        except wt.Trap as e:
            note = f"\n(stopped after {seconds:.0f} s)" if "interrupt" in str(e).lower() else f"\n(stopped: {str(e).splitlines()[0][:200]})"
        except wt.WasmtimeError as e:
            note = f"\n(stopped: {str(e).splitlines()[0][:200]})"
        finally:
            timer.cancel()
        text = out.read_text(errors="replace") if out.exists() else ""
        errs = err.read_text(errors="replace") if err.exists() else ""
        errs = "".join(x for x in errs.splitlines(True) if not x.startswith(STARTUP_NOISE))   # the build's own path notes
        body = text + (("\n" if text and not text.endswith("\n") else "") + errs if errs else "") + note
        return body[:output] + ("\n(output cut at 20 KB)" if len(body) > output else "")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ------------------------------------------------------------------ the skills, with their caps

class Skills:
    """one model's search and run. settings: search, run (on/off); per_turn, per_hour (fetches and searches);
    runs_per_turn; allow_private (tests only). on_event(kind, payload) hears every request."""

    def __init__(self, settings: dict | None = None, on_event: Callable | None = None):
        self.settings = {"search": True, "run": True, "per_turn": 3, "per_hour": 30, "runs_per_turn": 10, "allow_private": False}
        self.settings.update(settings or {})
        self.on_event = on_event
        self.seen: OrderedDict = OrderedDict()
        self.media: OrderedDict = OrderedDict()          # 1.5.2: images and videos it has read: url -> {kind, alt}
        self.hour: deque = deque()
        self.turn = {"look": 0, "run": 0}
        self.last_at: dict = {}
        self.robots: dict = {}

    # what the model has read: its links are the only ones it may follow
    def observe(self, text: str) -> None:
        for url in URL_RE.findall(text or ""):
            url = url.rstrip(".,;:!?)]}'\"")
            self.seen[url] = True
            self.seen.move_to_end(url)
        while len(self.seen) > SEEN_MAX:
            self.seen.popitem(last=False)

    def observe_bytes(self, raw: bytes) -> None:
        if b"http" in raw:
            self.observe(raw.decode("utf-8", errors="replace"))

    def forget(self) -> None:
        """a new chat: links read in another chat don't carry over"""
        self.seen.clear()
        self.media.clear()

    def media_in(self, text: str) -> list:
        """images and videos it has read that it shows in what it wrote (the chat shows them)"""
        out = []
        for url in URL_RE.findall(text or ""):
            url = url.rstrip(".,;:!?)]}'\"")
            m = self.media.get(url)
            if m and not any(x["src"] == url for x in out):
                out.append({"src": url, **m})
        return out[:MEDIA_MAX]

    def new_turn(self) -> None:
        self.turn = {"look": 0, "run": 0}

    def _event(self, event: str, **payload) -> None:
        if self.on_event:
            try:
                self.on_event(event, payload)
            except Exception:                            # noqa: BLE001 - the chat display never stops a turn
                pass

    # what the model asked for
    def request(self, written: str) -> str:
        """-> the file the model gets back (a page, results, output, or a note)"""
        text = (written or "").rstrip()
        block = FENCE_RE.search(text)
        line = next((x.strip() for x in reversed(text.split("\n")) if x.strip()), "")
        urls = [u.rstrip(".,;:!?)]}'\"") for u in URL_RE.findall(line)]
        kind = "run" if block else "fetch" if urls else "search"
        try:
            if block:
                return self._run(block.group(1))
            if urls:
                return self._fetch(urls[-1])
            query = re.sub(r"^(search|look up|lookup|find)\s*[:\-]?\s*", "", line, flags=re.I).strip()
            if not query:
                raise Refusal("nothing to search for: write a query, a link you've read, or code, then ask")
            return self._search(query[:300])
        except Refusal as r:
            self._event("skill", kind=kind, ok=False, what=(urls[-1] if urls else line)[:300], note=str(r))
            return f"(soma: {r})\n"

    def _spend(self, kind: str) -> None:
        if kind == "run":
            if not self.settings["run"]:
                raise Refusal("run is off for this model")
            if self.turn["run"] >= int(self.settings["runs_per_turn"]):
                raise Refusal(f"that's {self.settings['runs_per_turn']} runs this turn; no more until the next")
            self.turn["run"] += 1
            return
        if not self.settings["search"]:
            raise Refusal("search is off for this model")
        now = time.time()
        while self.hour and self.hour[0] < now - 3600:
            self.hour.popleft()
        if self.turn["look"] >= int(self.settings["per_turn"]):
            raise Refusal(f"that's {self.settings['per_turn']} searches and fetches this turn; no more until the next")
        if len(self.hour) >= int(self.settings["per_hour"]):
            raise Refusal(f"that's {self.settings['per_hour']} searches and fetches this hour; wait a while")
        self.turn["look"] += 1
        self.hour.append(now)

    def _polite(self, host: str) -> None:
        wait = self.last_at.get(host, 0) + 1.0 - time.time()
        if wait > 0:
            time.sleep(wait)
        self.last_at[host] = time.time()

    def _allowed(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        root = f"{parts.scheme}://{parts.netloc}"
        if root not in self.robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                self._polite(parts.hostname or "")
                _, _, data = http_get(root + "/robots.txt", allow_private=self.settings["allow_private"], limit=500_000)
                rp.parse(data.decode("utf-8", errors="replace").splitlines())
            except Refusal:
                rp.parse([])                             # no robots.txt (or unreadable): nothing disallowed
            self.robots[root] = rp
        return self.robots[root].can_fetch(ROBOT_NAME, url)

    def _fetch(self, url: str) -> str:
        if url not in self.seen:
            raise Refusal("soma can only fetch links it has read (typed to it, in a file, a page or search results)")
        self._spend("look")
        if not self._allowed(url):
            raise Refusal(f"{urllib.parse.urlsplit(url).hostname} asks robots like soma not to read that page")
        self._polite(urllib.parse.urlsplit(url).hostname or "")
        final, ctype, data = http_get(url, allow_private=self.settings["allow_private"])
        if ctype and not ctype.startswith(TEXT_TYPES):
            raise Refusal(f"that link is {ctype.split(';')[0]}, not text")
        text, links, media = page_text(final, ctype, data)
        self.observe(final)
        for link in links:
            self.observe(link)
        for m in media:
            self.media[m["src"]] = {"kind": m["kind"], "alt": m["alt"]}
            self.media.move_to_end(m["src"])
            self.seen[m["src"]] = True
        while len(self.media) > SEEN_MAX:
            self.media.popitem(last=False)
        body = f"{final}\n\n{text}\n" + (("\nlinks:\n" + "\n".join(f"- {u}" for u in links) + "\n") if links else "")
        excerpt = text[:EXCERPT_CHARS]
        self._event("skill", kind="fetch", ok=True, what=final, bytes=len(body.encode()), excerpt=excerpt,
                    media=[m for m in media if m["src"] in excerpt] + [m for m in media if m["src"] not in excerpt][:8])
        return body

    def _search(self, query: str) -> str:
        self._spend("look")
        q = urllib.parse.urlencode({"action": "query", "list": "search", "srsearch": query, "srlimit": 8, "format": "json",
                                    "utf8": 1, "srprop": "snippet"})
        self._polite("en.wikipedia.org")
        _, _, data = http_get(f"https://en.wikipedia.org/w/api.php?{q}", accept="application/json", limit=500_000)
        try:
            hits = json.loads(data.decode("utf-8"))["query"]["search"]
        except (ValueError, KeyError):
            raise Refusal("the search didn't answer")
        lines = [f"search: {query}", ""]
        for i, h in enumerate(hits, 1):
            url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(h["title"].replace(" ", "_"))
            snippet = html.unescape(re.sub(r"<[^>]+>", "", h.get("snippet") or ""))
            lines += [f"{i}. {h['title']}", f"   {url}", f"   {snippet}", ""]
            self.observe(url)
        if not hits:
            lines.append("nothing found.")
        body = "\n".join(lines) + "\n"
        self._event("skill", kind="search", ok=True, what=query, results=len(hits), bytes=len(body.encode()))
        return body

    def _run(self, code: str) -> str:
        if not sandbox_ready():
            raise Refusal("run isn't installed on this machine yet")
        self._spend("run")
        self._event("skill", kind="run", ok=None, what=code)
        out = run_python(code)
        self._event("skill", kind="run", ok=True, what=code, output=out[:2000], bytes=len(out.encode()))
        return f"ran:\n{out}" if out.strip() else "ran: (it printed nothing)\n"

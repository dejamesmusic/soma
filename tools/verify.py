#!/usr/bin/env python3
"""Check a runtime release in this repository against what logOS publishes.

    python3 tools/verify.py somatic-core-1.6.0        # a tag
    python3 tools/verify.py --all                     # every release tag

For a tag, it:
  1. reads runtimes/<family>/LOGOS_RELEASE.json at that tag,
  2. fetches logOS's own record of that release and checks the file hashes agree,
  3. where the release is signed (somatic core from 1.4.0), checks release.sig against release.json with the runtime
     signing key's public half (needs `pip install cryptography`), and that release.json names the same bundle hash,
  4. downloads the bundle from logOS, checks its sha256, unpacks it and compares it file by file with the tag's tree.

Nothing here trusts this repository: every check is against logOS's published record, and the bundle bytes.
Standard library only, except the signature check.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

SITE = "https://logossoma.com"
RELEASE_KEY = "0lLdgB3LtdBakH70/B8uQZmPYb0pxyOvVXullhq2jNU="   # the runtime signing key's public half (ed25519)
FOLDERS = {"soma.somatic_core": "somatic_core", "soma.rma_v1": "rma_v1", "soma.v14": "v14", "soma.v12.3": "v12.3"}
ADDED = {"LOGOS_RELEASE.json", "release.json", "release.sig", "SPEC.md"}   # beside the source, not in the bundle


def fetch(path: str) -> bytes:
    req = urllib.request.Request(path if path.startswith("http") else SITE + path, headers={"User-Agent": "soma-verify"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def tree(tag: str, folder: str) -> dict[str, bytes]:
    names = git("ls-tree", "-r", "--name-only", tag, f"runtimes/{folder}").decode().splitlines()
    return {n[len(f"runtimes/{folder}/"):]: git("show", f"{tag}:{n}") for n in names}


def unpacked(name: str, data: bytes) -> dict[str, bytes]:
    files = {}
    if name.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(data))
        for i in z.infolist():
            if not i.is_dir() and not i.filename.startswith("__MACOSX/"):
                files[i.filename] = z.read(i)
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():
                if m.isfile():
                    files[m.name] = t.extractfile(m).read()
    tops = {n.split("/", 1)[0] for n in files}
    if len(tops) == 1 and all("/" in n for n in files):              # one folder at the top: the repo holds its contents
        files = {n.split("/", 1)[1]: b for n, b in files.items()}
    return files


def verify(tag: str) -> bool:
    found = [f for f in FOLDERS.values() if git("ls-tree", "--name-only", tag, f"runtimes/{f}/LOGOS_RELEASE.json").strip()]
    rec = None
    for f in found:
        r = json.loads(git("show", f"{tag}:runtimes/{f}/LOGOS_RELEASE.json"))
        if tag.endswith(r["version"]):
            rec, folder = r, f
    if not rec:
        print(f"{tag}: not a logOS release tag (v8, v10 and v12 were published on github only)")
        return True
    ok = True
    say = lambda good, text: print(f"  {'ok ' if good else 'BAD'} {text}") or good  # noqa: E731
    print(f"{tag}: {rec['runtime_id']} {rec['version']}")
    logos = json.loads(fetch(f"/api/runtimes/{rec['runtime_id']}"))
    pub = next((r for r in logos["releases"] if r["version"] == rec["version"]), None)
    if not pub:
        return say(False, "logOS doesn't list this release")
    theirs = {f["name"]: f.get("sha256") for f in pub.get("files") or []}
    ours = {f["name"]: f["sha256"] for f in rec["files"]}
    ok &= say(all(theirs.get(n) == h for n, h in ours.items()), "file hashes match logOS's record")
    files = tree(tag, folder)
    if "release.sig" in files:
        body, sig = files["release.json"], files["release.sig"]
        ok &= say(hashlib.sha256(body).hexdigest() == ours.get("release.json"), "release.json is the published one")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            Ed25519PublicKey.from_public_bytes(base64.b64decode(RELEASE_KEY)).verify(sig, body)
            ok &= say(True, "release.json is signed by the runtime key")
        except ImportError:
            print("  --  signature not checked: pip install cryptography")
        except Exception:                                            # noqa: BLE001 - InvalidSignature
            ok &= say(False, "release.sig does not verify")
        named = json.loads(body)["bundle"]["sha256"]
    bundle = next(n for n in ours if n.endswith((".tar.gz", ".zip")))
    data = fetch(f"/api/runtimes/{rec['runtime_id']}/releases/{rec['version']}/files/{bundle}")
    digest = hashlib.sha256(data).hexdigest()
    ok &= say(digest == ours[bundle], f"{bundle} from logOS has the published sha256")
    if "release.sig" in files:
        ok &= say(digest == named, "the signed release names this bundle")
    inside = unpacked(bundle, data)
    here = {n: b for n, b in files.items() if n not in ADDED}
    same = inside.keys() == here.keys() and all(inside[n] == here[n] for n in here)
    if not same:
        extra, missing = sorted(here.keys() - inside.keys())[:5], sorted(inside.keys() - here.keys())[:5]
        changed = sorted(n for n in here.keys() & inside.keys() if here[n] != inside[n])[:5]
        print(f"     differs: extra {extra} missing {missing} changed {changed}")
    ok &= say(same, f"the tag's tree is the bundle, file for file ({len(here)} files)")
    return ok


def main() -> None:
    tags = git("tag", "--list").decode().split() if sys.argv[1:] == ["--all"] else sys.argv[1:]
    if not tags:
        sys.exit(__doc__)
    results = [verify(t) for t in tags]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

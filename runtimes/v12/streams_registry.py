"""
streams_registry.py — discover available streams.

Scans the streams/ directory for modules that declare STREAM_NAME and
STREAM_DESCRIPTION. Returns a list of dicts with name, description, 
and path. Doesn't import or execute the modules — just parses the 
constants out of the source.

This way:
  - dropping a new file into streams/ makes it appear in the launcher
    and gui automatically
  - parsing the metadata is cheap and safe (no side effects from 
    importing third-party deps a stream might need)
"""

import ast
import os
from pathlib import Path


def _bundle_root():
    """Resolve the bundle directory regardless of where this module 
    is imported from."""
    return Path(__file__).resolve().parent


def _read_constants(path):
    """Parse a python file and return a dict of top-level string constants
    declared as `NAME = "value"`. Skips anything that isn't a literal
    string assignment to a single name.
    """
    try:
        src = path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return {}
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if not isinstance(node.value, ast.Constant):
            continue
        if isinstance(node.value.value, str):
            out[node.targets[0].id] = node.value.value
    return out


def _stream_dirs(extra_dirs=None):
    """Stream script search order.

    User/runtime streams come first so people can drop accessories into
    ~/Library/Application Support/soma/streams without editing the app.
    Bundled streams remain as built-in defaults.
    """
    dirs = []
    seen = set()

    soma_home = os.environ.get("SOMA_HOME")
    if soma_home:
        dirs.append(Path(soma_home).expanduser() / "streams")

    if extra_dirs:
        dirs.extend(Path(d).expanduser() for d in extra_dirs)

    dirs.append(_bundle_root() / "streams")

    out = []
    for d in dirs:
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def list_streams(extra_dirs=None):
    """Return a list of {name, description, path, module} for every
    stream module found. Sorted by name. Modules without both 
    STREAM_NAME and STREAM_DESCRIPTION are skipped.
    """
    found = []
    seen_names = set()
    for streams_dir in _stream_dirs(extra_dirs):
        if not streams_dir.is_dir():
            continue
        for f in sorted(streams_dir.glob("*.py")):
            if f.name.startswith("_"):
                continue
            const = _read_constants(f)
            name = const.get("STREAM_NAME")
            desc = const.get("STREAM_DESCRIPTION")
            if not name or not desc or name in seen_names:
                continue
            seen_names.add(name)
            found.append({
                "name": name,
                "description": desc,
                "path": str(f),
                "module": f"streams.{f.stem}",
            })
    return sorted(found, key=lambda s: s["name"])


def find_stream(name):
    """Return the registry entry matching `name`, or None."""
    for s in list_streams():
        if s["name"] == name:
            return s
    return None


if __name__ == "__main__":
    # `python streams_registry.py` lists discovered streams (handy 
    # for the launcher's `./soma streams` subcommand).
    streams = list_streams()
    if not streams:
        print("  no streams found in streams/")
    else:
        print("  available streams:\n")
        width = max(len(s["name"]) for s in streams)
        for s in streams:
            print(f"    {s['name']:<{width}}   {s['description']}")

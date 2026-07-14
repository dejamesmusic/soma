#!/usr/bin/env python3
"""Pack PNG iconset files into a modern PNG-backed .icns file."""

from __future__ import annotations

import struct
import sys
from pathlib import Path


ENTRIES = [
    ("icp4", "icon_16x16.png"),
    ("ic11", "icon_16x16@2x.png"),
    ("icp5", "icon_32x32.png"),
    ("ic12", "icon_32x32@2x.png"),
    ("ic07", "icon_128x128.png"),
    ("ic13", "icon_128x128@2x.png"),
    ("ic08", "icon_256x256.png"),
    ("ic14", "icon_256x256@2x.png"),
    ("ic09", "icon_512x512.png"),
    ("ic10", "icon_512x512@2x.png"),
]


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: pack_icns.py <iconset> <output.icns>", file=sys.stderr)
        return 2

    iconset = Path(sys.argv[1])
    output = Path(sys.argv[2])
    chunks = []

    for code, filename in ENTRIES:
        path = iconset / filename
        if not path.exists():
            continue
        data = path.read_bytes()
        chunks.append(code.encode("ascii") + struct.pack(">I", len(data) + 8) + data)

    body = b"".join(chunks)
    output.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""macOS app entry point for soma."""

import os
import runpy
import sys
from pathlib import Path


def _resource_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def main():
    root = _resource_root()
    os.environ.setdefault(
        "SOMA_HOME",
        str(Path.home() / "Library" / "Application Support" / "soma"),
    )
    sys.path.insert(0, str(root))

    if len(sys.argv) >= 3 and sys.argv[1] == "--run-script":
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        runpy.run_path(script, run_name="__main__")
        return

    import soma_gui

    soma_gui.main()


if __name__ == "__main__":
    main()

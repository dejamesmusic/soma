#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "$ROOT/.packaging-venv/bin/python" ]; then
    PY="$ROOT/.packaging-venv/bin/python"
else
    PY="$ROOT/venv/bin/python"
fi
DIST="$ROOT/dist"
PYI_CONFIG="$ROOT/build/pyinstaller-config"
SWIFT_CACHE="$ROOT/build/swift-module-cache"
ICONSET="$ROOT/macos/assets/soma.iconset"
ICON="$ROOT/macos/assets/soma.icns"
DMG="$DIST/soma.dmg"
ZIP="$DIST/soma.zip"
DMG_STAGING="$ROOT/build/dmg"
SCRIPT_BUNDLE="$DIST/soma-v12.2-script-bundle"
SCRIPT_ZIP="$DIST/soma-v12.2-script-bundle.zip"
SCRIPT_ZIP_ALIAS="$DIST/soma-script-bundle.zip"

if [ ! -x "$PY" ]; then
    echo "python not found at $PY" >&2
    exit 1
fi

rm -rf "$ICONSET" "$ICON" "$DMG" "$ZIP" "$SCRIPT_BUNDLE" "$SCRIPT_ZIP" "$SCRIPT_ZIP_ALIAS"
mkdir -p "$ROOT/macos/assets"
mkdir -p "$SWIFT_CACHE"

CLANG_MODULE_CACHE_PATH="$SWIFT_CACHE" \
    /usr/bin/swift "$ROOT/macos/render_icon.swift" "$ICONSET"
if ! /usr/bin/iconutil -c icns -o "$ICON" "$ICONSET" 2>/dev/null; then
    "$PY" "$ROOT/macos/pack_icns.py" "$ICONSET" "$ICON"
fi

if ! "$PY" -m PyInstaller --version >/dev/null 2>&1; then
    "$PY" -m pip install pyinstaller
fi

rm -rf "$ROOT/build" "$DIST/soma" "$DIST/soma.app"
mkdir -p "$PYI_CONFIG"
export PYINSTALLER_CONFIG_DIR="$PYI_CONFIG"
"$PY" -m PyInstaller --clean --noconfirm "$ROOT/macos/soma.spec"

/usr/bin/codesign --force --deep --sign - "$DIST/soma.app"

rm -rf "$DMG_STAGING"
mkdir -p "$DMG_STAGING"
/usr/bin/ditto "$DIST/soma.app" "$DMG_STAGING/soma.app"
ln -s /Applications "$DMG_STAGING/Applications"

if /usr/bin/hdiutil create \
        -volname "soma" \
        -srcfolder "$DMG_STAGING" \
        -ov \
        -format UDZO \
        "$DMG"; then
    echo "$DMG"
else
    echo "hdiutil could not create a DMG here; creating zip fallback." >&2
    (cd "$DIST" && /usr/bin/ditto -c -k --sequesterRsrc --keepParent "soma.app" "soma.zip")
    echo "$ZIP"
fi

mkdir -p "$SCRIPT_BUNDLE/data" "$SCRIPT_BUNDLE/checkpoints"
mkdir -p "$SCRIPT_BUNDLE/cloud" "$SCRIPT_BUNDLE/streams" "$SCRIPT_BUNDLE/experiments"
/usr/bin/ditto "$ROOT/soma" "$SCRIPT_BUNDLE/soma"
/usr/bin/ditto "$ROOT/README.md" "$SCRIPT_BUNDLE/README.md"
/usr/bin/ditto "$ROOT/soma_v12_spec.md" "$SCRIPT_BUNDLE/soma_v12_spec.md"
/usr/bin/ditto "$ROOT/logOS_v12_2_web_handover.md" "$SCRIPT_BUNDLE/logOS_v12_2_web_handover.md"
/usr/bin/ditto "$ROOT/requirements.txt" "$SCRIPT_BUNDLE/requirements.txt"
/usr/bin/ditto "$ROOT/soma_v12_2.py" "$SCRIPT_BUNDLE/soma_v12_2.py"
/usr/bin/ditto "$ROOT/soma_train_worker.py" "$SCRIPT_BUNDLE/soma_train_worker.py"
/usr/bin/ditto "$ROOT/soma_v12_1.py" "$SCRIPT_BUNDLE/soma_v12_1.py"
/usr/bin/ditto "$ROOT/soma_v12.py" "$SCRIPT_BUNDLE/soma_v12.py"
/usr/bin/ditto "$ROOT/soma_loop.py" "$SCRIPT_BUNDLE/soma_loop.py"
/usr/bin/ditto "$ROOT/soma_logos_bridge.py" "$SCRIPT_BUNDLE/soma_logos_bridge.py"
/usr/bin/ditto "$ROOT/streams_registry.py" "$SCRIPT_BUNDLE/streams_registry.py"
/usr/bin/ditto "$ROOT/build_fineweb_edu_txt.py" "$SCRIPT_BUNDLE/build_fineweb_edu_txt.py"
/usr/bin/ditto "$ROOT/concat_txt_corpus.py" "$SCRIPT_BUNDLE/concat_txt_corpus.py"
/usr/bin/ditto "$ROOT/cloud/train_cloud.py" "$SCRIPT_BUNDLE/cloud/train_cloud.py"
/usr/bin/ditto "$ROOT/cloud/README_cloud.md" "$SCRIPT_BUNDLE/cloud/README_cloud.md"
for stream_script in "$ROOT"/streams/*.py; do
    [ -e "$stream_script" ] || continue
    /usr/bin/ditto "$stream_script" "$SCRIPT_BUNDLE/streams/$(basename "$stream_script")"
done
/usr/bin/ditto "$ROOT/experiments/throughput_benchmark.py" "$SCRIPT_BUNDLE/experiments/throughput_benchmark.py"
touch "$SCRIPT_BUNDLE/data/.gitkeep" "$SCRIPT_BUNDLE/checkpoints/.gitkeep"
chmod +x "$SCRIPT_BUNDLE/soma"
(cd "$DIST" && /usr/bin/zip -qry -X \
    "$(basename "$SCRIPT_ZIP")" "$(basename "$SCRIPT_BUNDLE")")
/usr/bin/ditto "$SCRIPT_ZIP" "$SCRIPT_ZIP_ALIAS"
echo "$SCRIPT_ZIP"
echo "$SCRIPT_ZIP_ALIAS"

echo "$DIST/soma.app"

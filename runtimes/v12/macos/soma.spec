# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

root = Path.cwd()

datas = [
    (str(root / "README.md"), "."),
    (str(root / "soma_v12_spec.md"), "."),
    (str(root / "requirements.txt"), "."),
    (str(root / "soma_logos_bridge.py"), "."),
    (str(root / "soma_train_worker.py"), "."),
    (str(root / "soma_v12_2.py"), "."),
    (str(root / "streams"), "streams"),
]

hiddenimports = [
    "soma_v10",
    "soma_v11",
    "soma_v12",
    "soma_v12_1",
    "soma_v12_2",
    "soma_gui",
    "soma_loop",
    "soma_logos_bridge",
    "soma_train_worker",
    "streams_registry",
    "torch",
    "numpy",
    "certifi",
]

a = Analysis(
    [str(root / "macos" / "soma_app.py")],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "scipy",
        "pandas",
        "IPython",
        "pytest",
        "setuptools.tests",
        "transformers",
        "sklearn",
        "scipy",
        "nltk",
        "onnxruntime",
        "PIL",
        "lxml",
        "pydantic",
        "cryptography",
        "uvloop",
        "anyio",
        "rich",
        "pygments",
        "urllib3",
        "requests",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="soma",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(root / "macos" / "assets" / "soma.icns"),
    target_arch="arm64",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="soma",
)

app = BUNDLE(
    coll,
    name="soma.app",
    icon=str(root / "macos" / "assets" / "soma.icns"),
    bundle_identifier="com.logos.soma",
    info_plist={
        "CFBundleName": "soma",
        "CFBundleDisplayName": "soma",
        "CFBundleShortVersionString": "12.2",
        "CFBundleVersion": "12.2.0",
        "LSMinimumSystemVersion": "12.0",
        "NSHighResolutionCapable": True,
    },
)

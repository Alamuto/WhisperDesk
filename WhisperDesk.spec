# -*- mode: python ; coding: utf-8 -*-
# Build: tools\prepare_build.py (preenche build_cache\) e depois `pyinstaller WhisperDesk.spec --noconfirm`.
# onedir sem UPX: inicialização rápida, DLLs CUDA intactas e menos falsos positivos de antivírus.
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)
CACHE = ROOT / "build_cache"
for need in (CACHE / "bin" / "ffmpeg.exe", CACHE / "bin" / "ffprobe.exe", CACHE / "cuda" / "cublas64_12.dll"):
    if not need.is_file():
        sys.exit(f"Falta {need}. Rode: python tools\\prepare_build.py")

binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
binaries += [(str(p), "bin") for p in (CACHE / "bin").glob("*.exe")]
binaries += [(str(p), "cuda") for p in (CACHE / "cuda").glob("*.dll")]

datas = collect_data_files("faster_whisper")          # silero_vad_v6.onnx
datas += [(str(p), "bin") for p in (CACHE / "bin").glob("*.txt")]
datas += [(str(ROOT / "assets" / "icon.ico"), "assets"), (str(ROOT / "assets" / "icon.png"), "assets")]
datas += [(str(ROOT / "packaging" / "THIRD_PARTY_NOTICES.txt"), ".")]

excludes = [
    "torch", "torchaudio", "torchvision", "tensorflow", "matplotlib", "pandas", "scipy", "IPython", "jupyter",
    "tkinter", "_tkinter", "pydub", "ffmpeg", "hf_xet", "pytest", "PyInstaller",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtWebChannel",
    "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.Qt3DCore", "PySide6.QtCharts",
    "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtPdfWidgets", "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtDesigner", "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtLocation",
    "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtTextToSpeech", "PySide6.QtSpatialAudio",
    "PySide6.QtGraphs", "PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSvg", "PySide6.QtSvgWidgets",
]

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=["app.pipeline.worker", "app.pipeline.live_worker", "app.pipeline.diarize", "pynvml", "PySide6.QtMultimedia", "PySide6.QtNetwork"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
# A análise de dependências também puxa cuBLAS/cuDNN do venv para a raiz de _internal; a cópia válida é a de cuda\.
_CUDA_NAMES = {p.name.lower() for p in (CACHE / "cuda").glob("*.dll")}
a.binaries = [b for b in a.binaries
              if not (Path(b[0]).name.lower() in _CUDA_NAMES and Path(b[0]).parent.as_posix() != "cuda")]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="WhisperDesk",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "icon.ico"),
    version=str(ROOT / "packaging" / "version_info.py"),
    manifest=str(ROOT / "packaging" / "WhisperDesk.manifest"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="WhisperDesk",
)

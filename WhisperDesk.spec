# -*- mode: python ; coding: utf-8 -*-
# Build: `pyinstaller WhisperDesk.spec --noconfirm` (Windows, Linux e macOS).
# onedir sem UPX: inicialização rápida e menos falsos positivos de antivírus.
#
# O que NÃO vai no pacote (v2.2):
#   - ffmpeg/ffprobe: a mídia é decodificada pelo PyAV (bibliotecas do FFmpeg já na wheel `av`);
#   - cuBLAS/cuDNN: baixados sob demanda pelo app só em máquinas com GPU NVIDIA (app/gpu_pack.py);
#   - partes do Qt que um app de widgets não usa (Quick/QML, teclado virtual, OpenGL por software, traduções).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH)
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"
sys.path.insert(0, str(ROOT))
from app import __version__ as VERSION  # noqa: E402

binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime")
datas = collect_data_files("faster_whisper")          # silero_vad_v6.onnx
datas += [(str(ROOT / "assets" / n), "assets") for n in ("icon.ico", "icon.png", "icon.icns")]
datas += [(str(ROOT / "packaging" / "THIRD_PARTY_NOTICES.txt"), ".")]

excludes = [
    "torch", "torchaudio", "torchvision", "tensorflow", "matplotlib", "pandas", "scipy", "IPython", "jupyter",
    "tkinter", "_tkinter", "pydub", "ffmpeg", "hf_xet", "pytest", "PyInstaller", "PIL",
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
    hiddenimports=["app.pipeline.worker", "app.pipeline.live_worker", "app.pipeline.diarize", "pynvml",
                   "PySide6.QtMultimedia", "PySide6.QtNetwork", *collect_submodules("av")],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# ---- poda do que não é usado (os nomes variam entre .dll, .so e .framework)
_DROP_PARTS = (
    "cublas", "cudnn", "nvrtc", "cudart", "nvblas",           # CUDA: pacote GPU sob demanda
    "opengl32sw",                                            # OpenGL por software (só para Qt Quick)
    "qt6quick", "qt6qml", "qtquick", "qtqml",                # Qt Quick/QML
    "qt6virtualkeyboard", "qtvirtualkeyboard",               # teclado virtual (e seu plugin)
    "qtuiotouchplugin",
    "qtiff", "qwebp", "qtga", "qwbmp",                       # formatos de imagem não usados
)
_DROP_DIRS = ("translations",)                               # o app tem i18n próprio; QTranslator não é usado
if not IS_MAC:
    _DROP_PARTS += ("qicns",)


def _keep(entry) -> bool:
    dest = entry[0].replace("\\", "/").lower()
    name = dest.rsplit("/", 1)[-1]
    if any(f"/{d}/" in f"/{dest}" for d in _DROP_DIRS) and "pyside6" in dest:
        return False
    if any(p in name for p in _DROP_PARTS):
        return name == "cudnn64_9.dll" and "ctranslate2" in dest  # stub de 260 kB da própria wheel do CT2
    return True


before = len(a.binaries) + len(a.datas)
a.binaries = [b for b in a.binaries if _keep(b)]
a.datas = [d for d in a.datas if _keep(d)]
print(f"[WhisperDesk.spec] podados {before - len(a.binaries) - len(a.datas)} arquivos")

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
    icon=str(ROOT / "assets" / ("icon.ico" if IS_WIN else "icon.icns")) if (IS_WIN or IS_MAC) else None,
    version=str(ROOT / "packaging" / "version_info.py") if IS_WIN else None,
    manifest=str(ROOT / "packaging" / "WhisperDesk.manifest") if IS_WIN else None,
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

if IS_MAC:
    app = BUNDLE(
        coll,
        name="WhisperDesk.app",
        icon=str(ROOT / "assets" / "icon.icns"),
        bundle_identifier="io.github.alamuto.whisperdesk",
        version=VERSION,
        info_plist={
            "CFBundleName": "WhisperDesk",
            "CFBundleDisplayName": "WhisperDesk",
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "LSMinimumSystemVersion": "12.0",
            "NSMicrophoneUsageDescription": "WhisperDesk usa o microfone para transcrever gravações ao vivo, "
                                            "no próprio computador.",
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "Audio/Video",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Alternate",
                "LSItemContentTypes": ["public.audio", "public.movie"],
            }],
        },
    )

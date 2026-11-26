"""Preparação do runtime nativo. Deve rodar antes de importar ctranslate2/faster_whisper,
tanto no processo da UI quanto no processo worker (spawn).

Bibliotecas CUDA (cuBLAS/cuDNN) vêm, em ordem: do pacote GPU baixado sob demanda (`app.gpu_pack`),
da pasta `cuda` de um build que as embuta, das wheels nvidia-* do venv (desenvolvimento) ou do sistema."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_done = False
_dll_handles = []   # mantém os handles vivos (add_dll_directory é revogado no GC)
_preloaded = []     # Linux: bibliotecas carregadas com RTLD_GLOBAL
_added: set[str] = set()


def _candidate_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        from app import gpu_pack

        dirs.append(gpu_pack.pack_dir())
    except Exception:  # noqa: BLE001
        pass
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        dirs += [base / "cuda", base / "bin"]
    else:
        # Desenvolvimento: bibliotecas das wheels nvidia-* no venv.
        import site

        for root in map(Path, site.getsitepackages()):
            nv = root / "nvidia"
            if nv.is_dir():
                sub = "bin" if os.name == "nt" else "lib"
                dirs += [d for d in nv.glob(f"*/{sub}") if d.is_dir()]
        repo = Path(__file__).resolve().parents[1]
        dirs += [repo / "bin", repo / "build_cache" / "cuda"]
    return [d for d in dirs if d.is_dir()]


def _add_windows_dirs(dirs: list[Path]) -> None:
    new = [d for d in dirs if str(d) not in _added]
    for d in new:
        _added.add(str(d))
        try:
            _dll_handles.append(os.add_dll_directory(str(d)))
        except OSError:
            pass
    # CTranslate2 usa LoadLibrary sem LOAD_LIBRARY_SEARCH_USER_DIRS: o PATH também é necessário.
    if new:
        os.environ["PATH"] = os.pathsep.join([*map(str, new), os.environ.get("PATH", "")])


def _preload_linux(dirs: list[Path]) -> None:
    """O LD_LIBRARY_PATH não muda depois que o processo começa: carregar as .so com RTLD_GLOBAL faz o
    dlopen("libcublas.so.12") do CTranslate2 (e os do próprio cuDNN) encontrarem as já carregadas."""
    libs: list[Path] = []
    for d in dirs:
        if str(d) in _added:
            continue
        _added.add(str(d))
        libs += sorted(p for p in d.iterdir() if ".so" in p.name and p.name.startswith(("libcublas", "libcudnn")))
    pending = libs
    while pending:  # ordem de dependência desconhecida: repete até não progredir
        left = []
        for p in pending:
            try:
                _preloaded.append(ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL))
            except OSError:
                left.append(p)
        if len(left) == len(pending):
            break
        pending = left


def activate_cuda_dirs() -> None:
    """(Re)registra as pastas de bibliotecas CUDA — chamado também logo após instalar o pacote GPU."""
    dirs = _candidate_dll_dirs()
    if os.name == "nt":
        _add_windows_dirs(dirs)
    elif sys.platform.startswith("linux"):
        _preload_linux(dirs)


def cuda_libs_ready() -> bool:
    """As bibliotecas que o CTranslate2 carrega para CUDA estão resolvíveis neste processo?"""
    prepare()
    if os.name == "nt":
        names = ("cublas64_12.dll", "cublasLt64_12.dll")
        loader = ctypes.WinDLL
    elif sys.platform.startswith("linux"):
        names = ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9")
        loader = ctypes.CDLL
    else:
        return False
    for n in names:
        try:
            loader(n)
        except OSError:
            return False
    return True


def prepare() -> None:
    global _done
    if _done:
        return
    _done = True
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "4,3,12,419430400")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")       # progresso de download observável em disco
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    activate_cuda_dirs()

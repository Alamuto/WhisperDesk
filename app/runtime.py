"""Preparação do runtime nativo. Deve rodar antes de importar ctranslate2/faster_whisper,
tanto no processo da UI quanto no processo worker (spawn)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

_done = False
_dll_handles = []  # mantém os handles vivos (add_dll_directory é revogado no GC)


def _candidate_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        dirs += [base / "cuda", base / "bin"]
    else:
        # Desenvolvimento: DLLs das wheels nvidia-* no venv.
        import site

        roots = [Path(p) for p in site.getsitepackages()]
        for root in roots:
            nv = root / "nvidia"
            if nv.is_dir():
                dirs += [d for d in nv.glob("*/bin") if d.is_dir()]
        repo = Path(__file__).resolve().parents[1]
        dirs += [repo / "bin", repo / "build_cache" / "cuda"]
    return [d for d in dirs if d.is_dir()]


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
    if os.name != "nt":
        return
    dirs = _candidate_dll_dirs()
    for d in dirs:
        try:
            _dll_handles.append(os.add_dll_directory(str(d)))
        except OSError:
            pass
    # CTranslate2 usa LoadLibrary sem LOAD_LIBRARY_SEARCH_USER_DIRS: o PATH também é necessário.
    if dirs:
        os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])

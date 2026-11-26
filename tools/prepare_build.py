"""Prepara build_cache/ com os binários nativos que o PyInstaller empacota.

  build_cache/bin/   ffmpeg.exe, ffprobe.exe (+ licença e README do build)
  build_cache/cuda/  cuBLAS/cuBLASLt (as únicas DLLs CUDA que o CTranslate2 carrega para o Whisper;
                     medido com EnumProcessModules durante transcrição float16 e int8_float16)

Uso: python tools/prepare_build.py [--ffmpeg-dir DIR]   (padrão: o ffmpeg do PATH)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "build_cache"
CUDA_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")


def _copy(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
        return
    print(f"  {src} -> {dst}")
    shutil.copy2(src, dst)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_ffmpeg_dir(explicit: str | None) -> Path:
    if explicit:
        d = Path(explicit)
    else:
        exe = shutil.which("ffmpeg")
        if not exe:
            sys.exit("ffmpeg não encontrado no PATH; use --ffmpeg-dir")
        d = Path(exe).resolve().parent
    if not (d / "ffmpeg.exe").is_file() or not (d / "ffprobe.exe").is_file():
        sys.exit(f"ffmpeg.exe/ffprobe.exe não estão em {d}")
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffmpeg-dir")
    args = ap.parse_args()

    ff = find_ffmpeg_dir(args.ffmpeg_dir)
    print(f"ffmpeg: {ff}")
    bin_dir = CACHE / "bin"
    for name in ("ffmpeg.exe", "ffprobe.exe"):
        _copy(ff / name, bin_dir)
    for doc in ("LICENSE", "LICENSE.txt", "README.txt"):
        for cand in (ff / doc, ff.parent / doc):
            if cand.is_file():
                shutil.copy2(cand, bin_dir / f"FFMPEG-{doc if doc.endswith('.txt') else doc + '.txt'}")
                break

    found = {}
    for sp in map(Path, site.getsitepackages()):
        for dll in CUDA_DLLS:
            for cand in (sp / "nvidia").glob(f"*/bin/{dll}"):
                found.setdefault(dll, cand)
    missing = [d for d in CUDA_DLLS if d not in found]
    if missing:
        sys.exit(f"DLLs CUDA ausentes no venv (instale nvidia-cublas-cu12): {missing}")
    for dll in CUDA_DLLS:
        _copy(found[dll], CACHE / "cuda")

    manifest = CACHE / "MANIFEST.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        for p in sorted(CACHE.rglob("*")):
            if p.is_file() and p != manifest:
                f.write(f"{_sha(p)}  {p.relative_to(CACHE).as_posix()}  {p.stat().st_size}\n")
    print(f"ok: {manifest}")


if __name__ == "__main__":
    main()

"""Pacote de aceleração por GPU (bibliotecas NVIDIA cuBLAS/cuDNN), baixado sob demanda.

O instalador não traz as bibliotecas CUDA (≈ 0,7 GB no Windows): só quem tem placa NVIDIA precisa delas.
Elas vêm das wheels oficiais da NVIDIA no PyPI, com versão e SHA-256 fixados aqui; apenas os arquivos
necessários são extraídos para a pasta local do app. O download é retomável e cancelável.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import ssl
import sys
import threading
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PACK_ID = "cu12-1"


@dataclass(frozen=True)
class Wheel:
    url: str
    sha256: str
    size: int
    members: tuple[str, ...]      # prefixos dentro da wheel
    names: tuple[str, ...] = ()   # se definido, só estes nomes de arquivo (basename começa com)


_PYPI = "https://files.pythonhosted.org/packages/"
PACKS: dict[str, tuple[Wheel, ...]] = {
    "win": (
        Wheel(_PYPI + "20/e2/fc9a0e985249d873150276d5afb02e39a66817fedbf1a385724393e505ed/"
                      "nvidia_cublas_cu12-12.9.2.10-py3-none-win_amd64.whl",
              "623f43027d40d44ceadf0043f002bd25cf353e8f13ce90b9a87057019f560661", 553_162_896,
              ("nvidia/cublas/bin/",), ("cublas64_12.dll", "cublasLt64_12.dll")),
    ),
    "linux": (
        Wheel(_PYPI + "cb/c0/0a517bfe63ccd3b92eb254d264e28fca3c7cab75d07daea315250fb1bf73/"
                      "nvidia_cublas_cu12-12.9.2.10-py3-none-manylinux_2_27_x86_64.whl",
              "e4f53a8ca8c5d6e8c492d0d0a3d565ecb59a751b19cfdaa4f6da0ab2104c1702", 581_240_110,
              ("nvidia/cublas/lib/",), ("libcublas.so", "libcublasLt.so")),
        Wheel(_PYPI + "2a/16/be895437624ae0400bda2b4bb69336cdb43346545e54edb15389d193a3b3/"
                      "nvidia_cudnn_cu12-9.28.1.4-py3-none-manylinux_2_27_x86_64.whl",
              "cbcf6f33c77fef3aa61d3fec36f2e89b07eb27d65b9f369c1adf3cfee08c2c21", 716_579_350,
              ("nvidia/cudnn/lib/",), ("libcudnn",)),
    ),
}

# Bibliotecas que o CTranslate2 carrega para rodar o Whisper em CUDA (por plataforma).
REQUIRED = {
    "win": ("cublas64_12.dll", "cublasLt64_12.dll"),
    "linux": ("libcublas.so.12", "libcublasLt.so.12", "libcudnn.so.9"),
}


class PackError(Exception):
    pass


class Cancelled(Exception):
    pass


def platform_key() -> str | None:
    if os.name == "nt" and platform.machine().lower() in ("amd64", "x86_64"):
        return "win"
    if sys.platform.startswith("linux") and platform.machine().lower() in ("x86_64", "amd64"):
        return "linux"
    return None  # macOS: sem CUDA (o CTranslate2 usa a CPU/Accelerate)


def pack_dir() -> Path:
    from app import paths

    return paths.local_dir() / "gpu" / PACK_ID


def _marker() -> Path:
    return pack_dir() / "pack.json"


def is_installed() -> bool:
    key = platform_key()
    if key is None:
        return False
    try:
        data = json.loads(_marker().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    files = data.get("files") or []
    return bool(files) and all((pack_dir() / f).is_file() for f in files)


def download_size_mb() -> int:
    key = platform_key()
    return sum(w.size for w in PACKS.get(key, ())) >> 20 if key else 0


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


def _download(w: Wheel, dst: Path, on_bytes: Callable[[int], None], cancel: threading.Event | None) -> None:
    """Baixa com retomada (Range) e confere o SHA-256 fixado."""
    have = dst.stat().st_size if dst.exists() else 0
    if have > w.size:
        dst.unlink()
        have = 0
    if have < w.size:
        req = urllib.request.Request(w.url, headers={"User-Agent": "WhisperDesk-gpu-pack",
                                                     **({"Range": f"bytes={have}-"} if have else {})})
        with urllib.request.urlopen(req, timeout=60, context=_ssl_context()) as r:
            if have and r.status != 206:
                have = 0  # servidor ignorou o Range: recomeça
            on_bytes(have)
            with open(dst, "ab" if have else "wb") as f:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    on_bytes(have)
    h = hashlib.sha256()
    with open(dst, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            h.update(block)
    if h.hexdigest() != w.sha256:
        dst.unlink(missing_ok=True)
        raise PackError("checksum")


def _extract(w: Wheel, wheel_path: Path, out: Path) -> list[str]:
    got: list[str] = []
    with zipfile.ZipFile(wheel_path) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or not name.startswith(w.members):
                continue
            base = name.rsplit("/", 1)[-1]
            if w.names and not base.startswith(w.names):
                continue
            if "/" in base or base in ("", ".", ".."):
                continue
            with z.open(info) as src, open(out / base, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            if os.name != "nt":
                os.chmod(out / base, 0o755)
            got.append(base)
    return got


def install(on_progress: Callable[[int, int], None], cancel: threading.Event | None = None) -> Path:
    """Baixa, verifica e extrai o pacote. `on_progress(bytes_baixados, bytes_totais)`."""
    key = platform_key()
    if key is None:
        raise PackError("unsupported platform")
    wheels = PACKS[key]
    total = sum(w.size for w in wheels)
    out = pack_dir()
    out.mkdir(parents=True, exist_ok=True)
    dl = out / ".download"
    dl.mkdir(exist_ok=True)
    need = total * 2 + (256 << 20)  # wheel baixada + arquivos extraídos
    free = shutil.disk_usage(out).free
    if free < need:
        raise PackError(f"disk: {need >> 20} MB / {free >> 20} MB")
    done_before = 0
    files: list[str] = []
    for w in wheels:
        part = dl / w.url.rsplit("/", 1)[-1]
        _download(w, part, lambda n, base=done_before: on_progress(base + n, total), cancel)
        files += _extract(w, part, out)
        part.unlink(missing_ok=True)
        done_before += w.size
    missing = [f for f in REQUIRED[key] if f not in files]
    if missing:
        raise PackError(f"missing {missing}")
    shutil.rmtree(dl, ignore_errors=True)
    _marker().write_text(json.dumps({"id": PACK_ID, "files": sorted(set(files))}), encoding="utf-8")
    on_progress(total, total)
    return out


def remove() -> None:
    shutil.rmtree(pack_dir(), ignore_errors=True)

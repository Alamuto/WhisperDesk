"""Catálogo de modelos, estimativas de VRAM e download com progresso/retomada."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

# Ordem exibida na UI
MODEL_LIST = ["tiny", "base", "small", "medium", "large-v3-turbo", "large-v2", "large-v3"]

# Tamanho aproximado do download (MB) — usado só para a barra de progresso.
MODEL_DOWNLOAD_MB = {
    "tiny": 75, "base": 145, "small": 484, "medium": 1530,
    "large-v2": 3090, "large-v3": 3090, "large-v3-turbo": 1620,
}

# Pesos em fp16 (MB) e custo incremental de ativação por item do batch (MB), medidos no CTranslate2.
_WEIGHTS_FP16_MB = {
    "tiny": 80, "base": 150, "small": 500, "medium": 1550,
    "large-v2": 3100, "large-v3": 3100, "large-v3-turbo": 1650,
}
_PER_BATCH_ITEM_MB = {
    "tiny": 20, "base": 30, "small": 60, "medium": 120,
    "large-v2": 180, "large-v3": 180, "large-v3-turbo": 110,
}
_OVERHEAD_MB = 600  # contexto CUDA + workspace cuBLAS/cuDNN

# Velocidade típica (x tempo real) numa GPU de 12 GB, fp16 batched — só para orientação na UI.
MODEL_SPEED_ESTIMATE = {
    "tiny": 120.0, "base": 90.0, "small": 50.0, "medium": 25.0,
    "large-v2": 14.0, "large-v3": 14.0, "large-v3-turbo": 40.0,
}

# Modelos que não suportam a tarefa "translate".
NO_TRANSLATE = {"large-v3-turbo"}

_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")


def estimate_vram_mb(model: str, compute_type: str, batch: int) -> int:
    w = _WEIGHTS_FP16_MB.get(model, 3100)
    if compute_type.startswith("int8"):
        w = w // 2 + 100
    per = _PER_BATCH_ITEM_MB.get(model, 180)
    return int(w + max(1, batch) * per + _OVERHEAD_MB)


def model_dir(models_root: Path, size: str) -> Path:
    return Path(models_root) / size


def is_model_ready(models_root: Path, size: str) -> bool:
    d = model_dir(models_root, size)
    return (d / ".complete").is_file() and all((d / f).is_file() for f in _REQUIRED_FILES)


def _legacy_cached(size: str) -> str | None:
    """Modelo já baixado pela versão antiga no cache padrão do Hugging Face."""
    try:
        from faster_whisper.utils import download_model

        p = download_model(size, local_files_only=True)
        if p and all(os.path.isfile(os.path.join(p, f)) for f in _REQUIRED_FILES):
            return p
    except Exception:
        return None
    return None


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def ensure_model(size: str, models_root: Path, on_progress: Callable[[int, int], None] | None = None,
                 cancel_event=None) -> str:
    """Retorna o caminho local do modelo, baixando (com retomada) se necessário."""
    if is_model_ready(models_root, size):
        return str(model_dir(models_root, size))
    legacy = _legacy_cached(size)
    if legacy:
        return legacy

    from faster_whisper.utils import download_model

    target = model_dir(models_root, size)
    target.mkdir(parents=True, exist_ok=True)
    expected = MODEL_DOWNLOAD_MB.get(size, 1000) << 20
    stop = threading.Event()

    def _monitor():
        while not stop.wait(0.5):
            if on_progress:
                on_progress(min(_dir_size(target), expected), expected)

    mon = threading.Thread(target=_monitor, name="model-download-monitor", daemon=True)
    mon.start()
    err: list[BaseException] = []
    result: list[str] = []

    def _dl():
        try:
            result.append(download_model(size, output_dir=str(target)))
        except BaseException as e:  # noqa: BLE001 — repassado ao chamador
            err.append(e)

    t = threading.Thread(target=_dl, name="model-download", daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.25)
            if cancel_event is not None and cancel_event.is_set():
                # O download é retomável (arquivos .incomplete); o processo será encerrado em seguida.
                raise InterruptedError("download cancelado")
    finally:
        stop.set()
    if err:
        e = err[0]
        raise RuntimeError(
            f"Falha ao baixar o modelo '{size}'. Verifique a conexão com a internet "
            f"(huggingface.co). Detalhe: {type(e).__name__}: {e}"
        ) from e
    if not all((target / f).is_file() for f in _REQUIRED_FILES):
        raise RuntimeError(f"Download do modelo '{size}' incompleto em {target}.")
    (target / ".complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    if on_progress:
        on_progress(expected, expected)
    return str(target)

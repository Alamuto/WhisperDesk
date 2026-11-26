"""
Configuração persistente (JSON em %APPDATA%\\WhisperDesk\\config.json), com escrita atômica.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any

from app import paths

_DEFAULTS = {
    "model_size": "large-v3-turbo",
    "language": None,  # None = detecção automática
    "task": "transcribe",
    "device": "cuda",
    "compute_type": "float16",
    "vad_filter": True,
    "beam_size": 5,
    "temperature": 0.0,
    "theme": "dark",
    "ui_language": None,
    "window_width": 1280,
    "window_height": 800,
}

_lock = threading.RLock()
_cache: dict | None = None
_migrated = False


def _path():
    global _migrated
    if not _migrated:
        _migrated = True
        paths.migrate_legacy()
    return paths.config_path()


def _write(data: dict) -> None:
    p = _path()
    fd, tmp = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def load() -> dict:
    global _cache
    with _lock:
        if _cache is not None:
            return dict(_cache)
        saved = {}
        p = _path()
        if p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                if not isinstance(saved, dict):
                    saved = {}
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                saved = {}
        _cache = {**_DEFAULTS, **saved}
        return dict(_cache)


def save(config: dict) -> None:
    global _cache
    with _lock:
        merged = {**_DEFAULTS, **config}
        _write(merged)
        _cache = merged


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set_value(key: str, value: Any) -> None:
    with _lock:
        cfg = load()
        cfg[key] = value
        save(cfg)


def reset() -> dict:
    global _cache
    with _lock:
        _write(dict(_DEFAULTS))
        _cache = dict(_DEFAULTS)
        return dict(_DEFAULTS)

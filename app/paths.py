"""Diretórios de dados por usuário (sem exigir privilégios de administrador).

- Configuração (pequena, itinerante):   %APPDATA%\\WhisperDesk\\config.json
- Dados volumosos (máquina local):       %LOCALAPPDATA%\\WhisperDesk\\{jobs,models,logs,gpu}
- macOS: ~/Library/Application Support/WhisperDesk · Linux: $XDG_CONFIG_HOME e $XDG_DATA_HOME

`WHISPERDESK_HOME` sobrescreve ambos (testes / modo portátil).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

APP_NAME = "WhisperDesk"
LEGACY_DIR = Path.home() / ".whisperdesk"


def _override() -> Path | None:
    v = os.environ.get("WHISPERDESK_HOME")
    return Path(v) if v else None


def roaming_dir() -> Path:
    o = _override()
    if o:
        return o
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_NAME


def local_dir() -> Path:
    o = _override()
    if o:
        return o
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_NAME


def _mk(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_path() -> Path:
    return _mk(roaming_dir()) / "config.json"


def jobs_dir() -> Path:
    return _mk(local_dir() / "jobs")


def journal_path() -> Path:
    return jobs_dir() / "journal.db"


def models_dir() -> Path:
    return _mk(local_dir() / "models")


def logs_dir() -> Path:
    return _mk(local_dir() / "logs")


def resource_dir() -> Path:
    """Raiz dos recursos empacotados (assets, bin, cuda)."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def migrate_legacy() -> list[str]:
    """Copia a configuração de ~/.whisperdesk (versões antigas). Idempotente; nunca apaga o legado."""
    notes: list[str] = []
    if not LEGACY_DIR.is_dir():
        return notes
    old_cfg = LEGACY_DIR / "config.json"
    new_cfg = roaming_dir() / "config.json"
    if old_cfg.is_file() and not new_cfg.exists():
        try:
            _mk(new_cfg.parent)
            shutil.copy2(old_cfg, new_cfg)
            notes.append(f"config migrada de {old_cfg}")
        except OSError as e:
            notes.append(f"falha ao migrar config: {e}")
    return notes

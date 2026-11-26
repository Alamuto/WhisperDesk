"""Integração com o sistema: abrir pastas, revelar arquivos e fontes padrão — Windows, macOS e Linux."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"


def _popen(args: list[str]) -> bool:
    try:
        subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True)
        return True
    except OSError:
        return False


def open_folder(path: str | os.PathLike) -> bool:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if IS_WIN:
        os.startfile(str(p))  # noqa: S606 — pasta do próprio app / do usuário
        return True
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    return QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


def reveal_file(path: str | os.PathLike) -> bool:
    """Abre o gerenciador de arquivos com o arquivo selecionado (ou a pasta, se não houver como selecionar)."""
    p = Path(path)
    if IS_WIN:
        return _popen(["explorer", "/select,", os.path.normpath(str(p))])
    if IS_MAC:
        return _popen(["open", "-R", str(p)])
    # Linux: o FileManager1 (Nautilus, Dolphin, Nemo, Caja…) seleciona o arquivo; senão abre a pasta.
    if shutil.which("dbus-send"):
        uri = p.resolve().as_uri()
        ok = subprocess.call(["dbus-send", "--session", "--print-reply", "--dest=org.freedesktop.FileManager1",
                              "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1.ShowItems",
                              f"array:string:{uri}", "string:"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5) == 0
        if ok:
            return True
    return open_folder(p.parent)


def ui_font_family() -> str:
    if IS_WIN:
        return "Segoe UI"
    if IS_MAC:
        return ".AppleSystemUIFont"
    return "Sans Serif"


def mono_families() -> list[str]:
    if IS_WIN:
        return ["Cascadia Mono", "Consolas", "Courier New"]
    if IS_MAC:
        return ["SF Mono", "Menlo", "Monaco", "Courier New"]
    return ["JetBrains Mono", "DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono", "Monospace"]

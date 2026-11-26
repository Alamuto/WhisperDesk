"""Compatibilidade: a folha de estilos agora é gerada a partir dos tokens em app.gui.theme."""
from app.gui.theme import delegate_colors, stylesheet


def get_theme(name: str) -> str:
    return stylesheet(name)


def theme_colors(name: str) -> dict:
    return delegate_colors(name)

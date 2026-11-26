#!/usr/bin/env bash
# Adiciona o WhisperDesk ao menu de aplicativos do usuário (sem root). Rode de dentro da pasta extraída.
set -euo pipefail
APPDIR="$(cd "$(dirname "$0")" && pwd)"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$DEST"
sed "s|@APPDIR@|$APPDIR|g" "$APPDIR/whisperdesk.desktop" > "$DEST/whisperdesk.desktop"
chmod +x "$DEST/whisperdesk.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$DEST" || true
echo "WhisperDesk adicionado ao menu ($DEST/whisperdesk.desktop)."

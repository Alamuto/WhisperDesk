#!/usr/bin/env bash
# Build de distribuição para Linux e macOS: testes -> PyInstaller -> selftest -> pacote.
#   Linux: dist/WhisperDesk-<versão>-linux-x86_64.tar.gz  (pasta onedir + atalho .desktop)
#   macOS: dist/WhisperDesk-<versão>-macos-<arch>.dmg      (WhisperDesk.app, assinatura ad-hoc)
# Uso: ./build.sh [--skip-tests]       (Python 3.12 em .venv; veja o README)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || { echo "[ERRO] $PY não encontrado. Crie o venv: python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt"; exit 1; }
VERSION="$("$PY" -c 'import app; print(app.__version__)')"
ARCH="$(uname -m)"

if [ "${1:-}" != "--skip-tests" ]; then
  echo "=== [1/3] Testes ==="
  QT_QPA_PLATFORM=offscreen "$PY" -m pytest -q tests
fi

echo "=== [2/3] PyInstaller ==="
rm -rf dist/WhisperDesk dist/WhisperDesk.app
"$PY" -m PyInstaller --noconfirm --clean WhisperDesk.spec

echo "=== Selftest do executável ==="
mkdir -p build
if [ "$(uname -s)" = "Darwin" ]; then
  BIN="dist/WhisperDesk.app/Contents/MacOS/WhisperDesk"
else
  BIN="dist/WhisperDesk/WhisperDesk"
fi
QT_QPA_PLATFORM=offscreen "$BIN" --selftest --out build/selftest.json || { cat build/selftest.json; exit 1; }
cat build/selftest.json

echo "=== [3/3] Pacote ==="
if [ "$(uname -s)" = "Darwin" ]; then
  # Assinatura ad-hoc: o Gatekeeper ainda pede "Abrir" na primeira vez (app não notarizado; veja o README).
  codesign --force --deep --sign - dist/WhisperDesk.app
  OUT="dist/WhisperDesk-${VERSION}-macos-${ARCH}.dmg"
  rm -f "$OUT"
  STAGE="$(mktemp -d)"
  cp -R dist/WhisperDesk.app "$STAGE/"
  ln -s /Applications "$STAGE/Applications"
  hdiutil create -volname "WhisperDesk ${VERSION}" -srcfolder "$STAGE" -ov -format UDZO "$OUT"
  rm -rf "$STAGE"
else
  cp packaging/linux/whisperdesk.desktop packaging/linux/install.sh dist/WhisperDesk/
  chmod +x dist/WhisperDesk/install.sh
  OUT="dist/WhisperDesk-${VERSION}-linux-${ARCH}.tar.gz"
  tar -C dist -czf "$OUT" WhisperDesk
fi
echo "Pacote: $OUT"

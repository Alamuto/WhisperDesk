@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo   WhisperDesk - ambiente de desenvolvimento
echo ============================================

set "PY=py -3.12"
%PY% --version >nul 2>&1
if errorlevel 1 (
    set "PY=python"
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERRO] Python 3.12 nao encontrado. Instale em https://www.python.org/downloads/
        exit /b 1
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Criando .venv ...
    %PY% -m venv .venv || exit /b 1
) else (
    echo [1/3] .venv ja existe.
)

echo [2/3] Instalando dependencias ...
".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt || (
    echo [ERRO] Falha ao instalar dependencias.
    exit /b 1
)

echo [3/3] Verificando ambiente ...
".venv\Scripts\python.exe" main.py --selftest
echo.
echo Pronto. Execute:  .venv\Scripts\python.exe main.py
endlocal

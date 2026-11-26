@echo off
REM Build de distribuicao: testes -> binarios nativos -> PyInstaller -> selftest -> instalador (se Inno Setup existir).
REM Uso: build.bat [--skip-tests] [--ffmpeg-dir C:\caminho\ffmpeg\bin]
setlocal EnableDelayedExpansion
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
set "SKIPTESTS="
set "FFARG="

:args
if "%~1"=="" goto args_done
if /i "%~1"=="--skip-tests" set "SKIPTESTS=1"
if /i "%~1"=="--ffmpeg-dir" (set "FFARG=--ffmpeg-dir "%~2"" & shift)
shift
goto args

:args_done
if not exist "%PY%" (
    echo [ERRO] .venv nao encontrado. Rode install.bat primeiro.
    exit /b 1
)

if not defined SKIPTESTS (
    echo === [1/5] Testes ===
    "%PY%" -m pytest -q tests || (echo [ERRO] Testes falharam. & exit /b 1)
)

echo === [2/5] Binarios nativos (ffmpeg, cuBLAS) ===
"%PY%" tools\prepare_build.py !FFARG! || exit /b 1

echo === [3/5] Icone ===
if not exist "assets\icon.ico" ("%PY%" tools\make_icon.py || exit /b 1)

echo === [4/5] PyInstaller ===
if exist "dist\WhisperDesk" rmdir /s /q "dist\WhisperDesk"
"%PY%" -m PyInstaller --noconfirm --clean WhisperDesk.spec || (echo [ERRO] PyInstaller falhou. & exit /b 1)

echo === Selftest do executavel ===
"dist\WhisperDesk\WhisperDesk.exe" --selftest --out "build\selftest.json"
if errorlevel 1 (
    echo [ERRO] Selftest falhou. Veja build\selftest.json
    exit /b 1
)
type "build\selftest.json"

echo === [5/5] Instalador ===
set "ISCC="
for %%P in ("%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" "%ProgramFiles%\Inno Setup 6\ISCC.exe" "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe") do (
    if exist %%P set "ISCC=%%~P"
)
if not defined ISCC (
    where ISCC.exe >nul 2>&1 && for /f "delims=" %%I in ('where ISCC.exe') do set "ISCC=%%I"
)
if defined ISCC (
    REM Compila fora do projeto e move depois: antivirus/indexadores podem travar o .exe
    REM recem-criado dentro da pasta do projeto ("arquivo em uso" no fim da compilacao).
    set "ISS_OUT=%TEMP%\WhisperDesk-iss-%RANDOM%"
    "!ISCC!" /Q /O"!ISS_OUT!" installer\WhisperDesk.iss || (echo [ERRO] Inno Setup falhou. & exit /b 1)
    if not exist "dist\installer" mkdir "dist\installer"
    move /Y "!ISS_OUT!\*.exe" "dist\installer\" >nul || (echo [ERRO] Nao foi possivel mover o instalador. & exit /b 1)
    rmdir /q "!ISS_OUT!" 2>nul
    echo Instalador: dist\installer\
) else (
    echo [AVISO] Inno Setup 6 nao encontrado - instalador nao gerado.
    echo         Instale com:  winget install JRSoftware.InnoSetup   e rode build.bat de novo.
    echo         A pasta dist\WhisperDesk\ ja e distribuivel como ZIP.
)

echo.
echo Build concluido: dist\WhisperDesk\WhisperDesk.exe
endlocal

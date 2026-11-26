"""WhisperDesk: transcrição local de áudio/vídeo com faster-whisper.

Ponto de entrada. Uso:
  WhisperDesk.exe [arquivo]                    abre a interface (instância única; o arquivo é repassado)
  WhisperDesk.exe --selftest [--out f.json]    verifica ffmpeg, CTranslate2 e CUDA; código de saída 0/1
  WhisperDesk.exe --selftest-transcribe ARQ [--model tiny] [--device cpu|cuda] [--spawn] [--out f.json]
                                               executa o pipeline completo sem interface (smoke test)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys

# Precisa vir antes de qualquer outra coisa: no executável congelado, o processo worker (spawn)
# reentra por aqui e deve ser desviado para o código do multiprocessing.
if __name__ == "__main__":
    mp.freeze_support()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

APP_ID = "WhisperDesk"


def _setup_logging():
    import logging
    from logging.handlers import RotatingFileHandler

    from app import paths

    handlers: list[logging.Handler] = [
        RotatingFileHandler(paths.logs_dir() / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"),
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import platform

    from app import __version__

    logging.getLogger("whisperdesk").info(
        "WhisperDesk %s | Python %s | %s | frozen=%s | pid=%d", __version__, platform.python_version(),
        platform.platform(), bool(getattr(sys, "frozen", False)), os.getpid())


def _install_excepthook():
    import datetime
    import logging
    import traceback

    from app import paths

    def hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(paths.logs_dir() / "crash.log", "a", encoding="utf-8") as f:
                f.write(f"\n{'=' * 60}\n{datetime.datetime.now().isoformat()}\n{text}")
        except OSError:
            pass
        logging.getLogger("crash").error("Exceção não tratada:\n%s", text)
        # Slots Qt que levantam exceção não derrubam o app; o erro fica registrado.

    sys.excepthook = hook


def _arg_value(argv: list[str], name: str, default: str | None = None) -> str | None:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return argv[i + 1]
    return default


def _emit_report(report: dict, out: str | None) -> None:
    import json

    from app import paths

    text = json.dumps(report, ensure_ascii=False, indent=2)
    target = out or str(paths.logs_dir() / "selftest.json")
    with open(target, "w", encoding="utf-8") as f:
        f.write(text)
    if sys.stdout is not None:
        try:
            sys.stdout.write(text + "\n")
            sys.stdout.flush()
        except (OSError, ValueError):
            pass


def selftest(argv: list[str]) -> int:
    import platform
    import subprocess

    from app import paths, runtime

    runtime.prepare()
    report: dict = {"ok": False, "frozen": bool(getattr(sys, "frozen", False)), "python": platform.python_version(),
                    "data_dir": str(paths.local_dir())}
    ok = True
    try:
        from app.pipeline import ffmpeg_tools

        for name in ("ffmpeg", "ffprobe"):
            exe = ffmpeg_tools.tool(name)
            out = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            report[name] = {"path": exe, "version": (out.stdout.splitlines() or ["?"])[0]}
    except Exception as e:  # noqa: BLE001
        report["ffmpeg_error"] = repr(e)
        ok = False
    try:
        import ctranslate2

        report["ctranslate2"] = ctranslate2.__version__
        report["cpu_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cpu"))
        n = ctranslate2.get_cuda_device_count()
        report["cuda_devices"] = n
        if n:
            report["cuda_compute_types"] = sorted(ctranslate2.get_supported_compute_types("cuda"))
    except Exception as e:  # noqa: BLE001
        report["ctranslate2_error"] = repr(e)
        ok = False
    try:
        import faster_whisper

        report["faster_whisper"] = faster_whisper.__version__
        from faster_whisper.vad import get_vad_model  # carrega o ONNX do Silero

        get_vad_model()
        report["vad"] = "ok"
    except Exception as e:  # noqa: BLE001
        report["faster_whisper_error"] = repr(e)
        ok = False
    try:
        from app.core.gpu_monitor import get_gpu_info

        info = get_gpu_info()
        report["gpu"] = {"name": info.name, "vram": info.display_str} if info else None
    except Exception as e:  # noqa: BLE001
        report["gpu_error"] = repr(e)
    if report.get("cuda_devices"):
        # cuBLAS/cuDNN são carregados sob demanda pelo CTranslate2: verificar que estão resolvíveis.
        import ctypes

        libs = {}
        # Só estas são carregadas pelo CTranslate2 no Whisper (medido); as sub-libs do cuDNN não são empacotadas.
        for dll in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
            try:
                ctypes.WinDLL(dll)
                libs[dll] = "ok"
            except OSError as e:
                libs[dll] = f"FALHOU: {e}"
        report["cuda_libs"] = libs
        report["cuda_runtime"] = "ok" if all(v == "ok" for v in libs.values()) else "incompleto"
    report["ok"] = ok
    _emit_report(report, _arg_value(argv, "--out"))
    return 0 if ok else 1


def selftest_transcribe(argv: list[str]) -> int:
    """Executa o pipeline real no próprio processo (sem GUI) e exporta SRT/JSON ao lado do --out."""
    import queue
    import threading
    import time

    from app import paths, runtime
    from app.pipeline import ffmpeg_tools, service
    from app.pipeline.worker import run_job

    runtime.prepare()
    src = _arg_value(argv, "--selftest-transcribe")
    report: dict = {"ok": False, "src": src}
    if not src or not os.path.isfile(src):
        report["error"] = "arquivo inexistente"
        _emit_report(report, _arg_value(argv, "--out"))
        return 1
    cfg = {"model_size": _arg_value(argv, "--model", "tiny"), "language": _arg_value(argv, "--language"),
           "task": "transcribe", "device": _arg_value(argv, "--device", "cpu"),
           "compute_type": _arg_value(argv, "--compute", "float16"), "vad_filter": True, "beam_size": 5,
           "temperature": 0.0}
    info = ffmpeg_tools.probe(src)
    jid, state = service.create_or_resume_job(src, cfg, info.default_stream.index, info.duration)
    if state == "DONE":
        service.journal().reset_for_rerun(jid)
    t0 = time.monotonic()
    job_args = (jid, str(paths.journal_path()), str(paths.models_dir()), str(paths.logs_dir()))
    collected: list = []
    if "--spawn" in argv:
        # Mesmo caminho da GUI: processo "spawn" separado (valida freeze_support no executável).
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        events = ctx.Queue()
        proc = ctx.Process(target=run_job, args=(*job_args, events, ctx.Event()), daemon=True)
        proc.start()
        while proc.is_alive() or not events.empty():
            try:
                collected.append(events.get(timeout=0.2))
            except queue.Empty:
                pass
        proc.join()
        report["worker_exitcode"] = proc.exitcode
    else:
        events = queue.Queue()
        run_job(*job_args, events, threading.Event())
        while not events.empty():
            collected.append(events.get_nowait())
    kinds: dict[str, int] = {}
    last = {}
    for kind, payload in collected:
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind in ("done", "error", "engine", "degraded", "language"):
            last[kind] = payload
    row = service.job(jid)
    report.update(job_id=jid, state=row["state"], elapsed_s=round(time.monotonic() - t0, 2), events=kinds,
                  **{k: v for k, v in last.items()})
    out = _arg_value(argv, "--out")
    if row["state"] == "DONE":
        segs = service.load_segments(jid)
        report["segments"] = len(segs)
        report["text_head"] = " ".join(s["text"] for s in segs)[:300]
        if out:
            base = os.path.splitext(out)[0]
            service.export(jid, "srt", base + ".srt")
            service.export(jid, "json", base + ".transcript.json")
        report["ok"] = True
    _emit_report(report, out)
    service.close()
    return 0 if report["ok"] else 1


def _forward_to_running_instance(server_name: str, path: str) -> bool:
    from PySide6.QtNetwork import QLocalSocket

    sock = QLocalSocket()
    sock.connectToServer(server_name)
    if not sock.waitForConnected(400):
        return False
    sock.write((path or "").encode("utf-8"))
    sock.flush()
    sock.waitForBytesWritten(1000)
    sock.disconnectFromServer()
    return True


def run_gui(argv: list[str]) -> int:
    import getpass
    import hashlib

    from app import runtime

    runtime.prepare()
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtNetwork import QLocalServer
    from PySide6.QtWidgets import QApplication

    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(argv)
    app.setApplicationName(APP_ID)
    app.setOrganizationName(APP_ID)

    file_arg = next((a for a in argv[1:] if not a.startswith("--") and os.path.isfile(a)), "")
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001
        user = "user"
    server_name = f"{APP_ID}-{hashlib.sha1(user.encode('utf-8')).hexdigest()[:12]}"
    if _forward_to_running_instance(server_name, os.path.abspath(file_arg) if file_arg else ""):
        return 0
    QLocalServer.removeServer(server_name)  # resíduo de uma instância que travou
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
    server.listen(server_name)

    from app.utils.i18n import init_i18n

    init_i18n()
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    from app import paths

    icon = paths.resource_dir() / "assets" / "icon.ico"
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))

    from app.gui.main_window import MainWindow

    window = MainWindow()

    def _on_new_connection():
        conn = server.nextPendingConnection()
        if conn is None:
            return

        def _read():
            data = bytes(conn.readAll()).decode("utf-8", "replace").strip()
            window.open_external_file(data)
            conn.disconnectFromServer()

        conn.readyRead.connect(_read)
        conn.disconnected.connect(conn.deleteLater)
        if conn.bytesAvailable():
            _read()
        else:
            # Conexão sem arquivo: só traz a janela para frente.
            window.open_external_file("")

    server.newConnection.connect(_on_new_connection)
    window.show()
    if file_arg:
        window.open_external_file(os.path.abspath(file_arg))
    code = app.exec()
    server.close()
    return code


def main() -> int:
    argv = list(sys.argv)
    _setup_logging()
    _install_excepthook()
    if "--selftest" in argv:
        return selftest(argv)
    if "--selftest-transcribe" in argv:
        return selftest_transcribe(argv)
    return run_gui(argv)


if __name__ == "__main__":
    sys.exit(main())

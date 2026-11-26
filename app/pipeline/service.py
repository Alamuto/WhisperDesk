"""Fachada usada pela UI: criação/retomada de jobs, histórico, carga de resultados e exportação."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

from app import paths
from app.pipeline import ENGINE_VERSION
from app.pipeline.exporters import export_segments
from app.pipeline.journal import ACTIVE_STATES, Journal, file_fingerprint, job_id_for

_lock = threading.Lock()
_journal: Journal | None = None


def journal() -> Journal:
    """Conexão do processo da UI (somente a thread principal a usa)."""
    global _journal
    with _lock:
        if _journal is None:
            _journal = Journal(paths.journal_path())
        return _journal


def close() -> None:
    global _journal
    with _lock:
        if _journal is not None:
            _journal.close()
            _journal = None


JOB_CONFIG_KEYS = ("model_size", "language", "task", "device", "compute_type", "vad_filter",
                   "beam_size", "temperature", "audio_stream")


def create_or_resume_job(src_path: str, cfg: dict, audio_stream: int, media_duration: float | None) -> tuple[str, str]:
    """Retorna (job_id, estado). Mesmo arquivo + mesma config relevante = mesmo job (retomável)."""
    j = journal()
    job_cfg = {k: cfg.get(k) for k in JOB_CONFIG_KEYS}
    job_cfg["audio_stream"] = audio_stream
    job_cfg["engine_version"] = ENGINE_VERSION
    fp = file_fingerprint(src_path)
    jid = job_id_for(fp, job_cfg)
    existing = j.job(jid)
    if existing is not None:
        # Parâmetros que não mudam a identidade (device/precisão) podem ser atualizados.
        merged = {**json.loads(existing["config_json"]), **job_cfg}
        j.update_fields(jid, config_json=json.dumps(merged), src_path=src_path,
                        display_name=os.path.basename(src_path))
        if existing["state"] == "FAILED":
            j.set_state(jid, "PAUSED" if j.chunks(jid) else "CREATED")
        return jid, j.job(jid)["state"]
    j.upsert_job(jid, src_path, fp, job_cfg, audio_stream, media_duration)
    return jid, "CREATED"


def resumable_jobs() -> list[sqlite3.Row]:
    rows = journal().list_jobs(ACTIVE_STATES)
    return [r for r in rows if os.path.isfile(r["src_path"])]


def history(limit: int = 200) -> list[sqlite3.Row]:
    return journal().list_jobs(("DONE",), limit=limit)


def job(job_id: str):
    return journal().job(job_id)


def job_progress(job_id: str) -> float:
    j = journal()
    row = j.job(job_id)
    if not row:
        return 0.0
    if row["state"] == "DONE":
        return 1.0
    if not row["total_samples"]:
        return 0.0
    return min(1.0, j.done_samples(job_id) / row["total_samples"])


def load_segments(job_id: str) -> list[dict]:
    return journal().segments_with_words(job_id)


def delete_job(job_id: str) -> None:
    journal().delete_job(job_id)


def clear_history() -> int:
    rows = history(limit=100000)
    for r in rows:
        journal().delete_job(r["id"])
    return len(rows)


def discard_job(job_id: str) -> None:
    """Abandona um job incompleto (remove WAV e estado)."""
    journal().delete_job(job_id)


def export(job_id: str, fmt: str, path: str | Path, segments: list[dict] | None = None) -> None:
    j = journal()
    row = j.job(job_id)
    if row is None:
        raise ValueError("Job inexistente.")
    segs = segments if segments is not None else j.segments_with_words(job_id)
    cfg = json.loads(row["config_json"])
    meta = {
        "source": row["display_name"],
        "duration_s": row["media_duration"],
        "model": cfg.get("model_size"),
        "language": cfg.get("language") or row["detected_language"],
        "task": cfg.get("task"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(row["created_at"])),
        "generator": "WhisperDesk",
        "warnings": j.warnings(job_id),
    }
    export_segments(segs, fmt, path, metadata=meta)


# ---------------------------------------------------------------------------
# Migração do histórico da versão antiga (~/.whisperdesk/history.db)
def import_legacy_history() -> int:
    from app.utils import config as app_config

    if app_config.get("legacy_history_imported"):
        return 0
    src = paths.LEGACY_DIR / "history.db"
    if not src.is_file():
        app_config.set_value("legacy_history_imported", True)
        return 0
    imported = 0
    j = journal()
    try:
        old = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=5)
        old.row_factory = sqlite3.Row
        rows = old.execute("SELECT * FROM transcriptions ORDER BY id").fetchall()
        old.close()
    except sqlite3.Error:
        return 0
    for r in rows:
        jid = f"legacy-{r['id']}"
        if j.job(jid) is not None:
            continue
        try:
            segs = json.loads(r["segments_json"] or "[]")
        except json.JSONDecodeError:
            continue
        try:
            ts = time.mktime(time.strptime(r["created_at"][:19], "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError):
            ts = time.time()
        cfg = {"model_size": r["model"], "language": r["language"] or None, "task": "transcribe"}
        with j.tx() as c:
            c.execute(
                """INSERT INTO job(id, src_path, display_name, fingerprint, config_json, audio_stream,
                                   media_duration, detected_language, state, created_at, updated_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,'DONE',?,?,?)""",
                (jid, r["filepath"], r["filename"], "legacy", json.dumps(cfg), -1, r["duration"],
                 r["language"] or None, ts, ts, ts))
            c.execute("INSERT INTO chunk(job_id, idx, start_sample, end_sample, pad_left, pad_right, state) "
                      "VALUES (?,0,0,?,0,0,'done')", (jid, int((r["duration"] or 0) * 16000)))
            seq = 0
            for s in segs:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                c.execute("INSERT INTO segment(job_id, chunk_idx, seq, start_ms, end_ms, text) VALUES (?,0,?,?,?,?)",
                          (jid, seq, int(float(s.get("start", 0)) * 1000), int(float(s.get("end", 0)) * 1000), text))
                seq += 1
        imported += 1
    app_config.set_value("legacy_history_imported", True)
    return imported

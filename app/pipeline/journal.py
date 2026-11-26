"""Journal SQLite: estado do job, plano de chunks e resultados commitados chunk a chunk.

Garantias:
- WAL + synchronous=NORMAL: queda de energia perde no máximo a última transação; o banco não corrompe.
- commit_chunk é uma transação idempotente (DELETE + INSERT + UPDATE): refazer um chunk nunca duplica.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.pipeline import ENGINE_VERSION

SCHEMA_VERSION = 1
MAX_CHUNK_ATTEMPTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS job (
    id                TEXT PRIMARY KEY,
    src_path          TEXT NOT NULL,
    display_name      TEXT NOT NULL,
    fingerprint       TEXT NOT NULL,
    config_json       TEXT NOT NULL,
    audio_stream      INTEGER NOT NULL,
    media_duration    REAL,
    wav_path          TEXT,
    total_samples     INTEGER,
    detected_language TEXT,
    state             TEXT NOT NULL DEFAULT 'CREATED',
    error             TEXT,
    warnings          TEXT,
    owner_pid         INTEGER,
    heartbeat         REAL,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    finished_at       REAL
);
CREATE INDEX IF NOT EXISTS job_state ON job(state, updated_at);

CREATE TABLE IF NOT EXISTS chunk (
    job_id        TEXT NOT NULL REFERENCES job(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    start_sample  INTEGER NOT NULL,
    end_sample    INTEGER NOT NULL,
    pad_left      INTEGER NOT NULL,
    pad_right     INTEGER NOT NULL,
    state         TEXT NOT NULL DEFAULT 'pending'
                  CHECK (state IN ('pending','running','done','failed')),
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    device        TEXT,
    compute_type  TEXT,
    batch_size    INTEGER,
    audio_s       REAL,
    elapsed_s     REAL,
    tail_text     TEXT,
    tail_logprob  REAL,
    PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS segment (
    job_id            TEXT NOT NULL,
    chunk_idx         INTEGER NOT NULL,
    seq               INTEGER NOT NULL,
    start_ms          INTEGER NOT NULL,
    end_ms            INTEGER NOT NULL,
    text              TEXT NOT NULL,
    avg_logprob       REAL,
    no_speech_prob    REAL,
    compression_ratio REAL,
    speaker           TEXT,
    PRIMARY KEY (job_id, chunk_idx, seq),
    FOREIGN KEY (job_id, chunk_idx) REFERENCES chunk(job_id, idx) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS segment_time ON segment(job_id, start_ms);

CREATE TABLE IF NOT EXISTS word (
    job_id     TEXT NOT NULL,
    chunk_idx  INTEGER NOT NULL,
    seg_seq    INTEGER NOT NULL,
    seq        INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    word       TEXT NOT NULL,
    prob       REAL NOT NULL,
    PRIMARY KEY (job_id, chunk_idx, seg_seq, seq),
    FOREIGN KEY (job_id, chunk_idx, seg_seq)
        REFERENCES segment(job_id, chunk_idx, seq) ON DELETE CASCADE
);
"""

RELEVANT_CONFIG_KEYS = (
    "model_size", "language", "task", "beam_size", "vad_filter", "audio_stream", "engine_version",
)

# Estados terminais / retomáveis
ACTIVE_STATES = ("CREATED", "EXTRACTING", "PLANNING", "TRANSCRIBING", "POSTPROCESSING", "PAUSED")
TERMINAL_STATES = ("DONE", "FAILED")


def file_fingerprint(path: str, probe_bytes: int = 4 << 20) -> str:
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(path, "rb") as f:
        h.update(f.read(probe_bytes))
        if st.st_size > probe_bytes:
            f.seek(max(0, st.st_size - probe_bytes))
            h.update(f.read(probe_bytes))
    return h.hexdigest()


def job_id_for(fingerprint: str, config: dict) -> str:
    relevant = {k: config.get(k) for k in RELEVANT_CONFIG_KEYS}
    relevant["engine_version"] = ENGINE_VERSION
    payload = fingerprint + json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


class Journal:
    def __init__(self, db_path: Path | str):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = db_path
        self.conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        ver = self.conn.execute("PRAGMA user_version").fetchone()[0]
        if ver < SCHEMA_VERSION:
            self.conn.executescript(SCHEMA)
            self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def workdir(self, job_id: str) -> Path:
        d = self.path.parent / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # ---------- job ----------
    def upsert_job(self, job_id, src_path, fingerprint, config, audio_stream, media_duration=None) -> sqlite3.Row:
        now = time.time()
        with self.tx() as c:
            c.execute(
                """INSERT INTO job(id, src_path, display_name, fingerprint, config_json, audio_stream,
                                   media_duration, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET src_path=excluded.src_path,
                       display_name=excluded.display_name, updated_at=excluded.updated_at""",
                (job_id, src_path, os.path.basename(src_path), fingerprint, json.dumps(config),
                 audio_stream, media_duration, now, now),
            )
        return self.job(job_id)

    def job(self, job_id) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()

    def config(self, job_id) -> dict:
        row = self.job(job_id)
        return json.loads(row["config_json"]) if row else {}

    def set_state(self, job_id, state, error=None, **fields):
        if state in TERMINAL_STATES:
            fields.setdefault("finished_at", time.time())
        cols = "".join(f", {k}=?" for k in fields)
        sql = f"UPDATE job SET state=?, error=?, updated_at=?{cols} WHERE id=?"
        with self.tx() as c:
            c.execute(sql, (state, error, time.time(), *fields.values(), job_id))

    def update_fields(self, job_id, **fields):
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        self.conn.execute(f"UPDATE job SET {cols}, updated_at=? WHERE id=?", (*fields.values(), time.time(), job_id))

    def add_warning(self, job_id, text: str):
        row = self.job(job_id)
        items = json.loads(row["warnings"]) if row and row["warnings"] else []
        if text not in items:
            items.append(text)
        self.conn.execute("UPDATE job SET warnings=? WHERE id=?", (json.dumps(items, ensure_ascii=False), job_id))

    def warnings(self, job_id) -> list[str]:
        row = self.job(job_id)
        return json.loads(row["warnings"]) if row and row["warnings"] else []

    def claim(self, job_id, pid: int, stale_after_s: float = 30.0) -> bool:
        """Lease: impede que dois processos vivos processem o mesmo job."""
        now = time.time()
        with self.tx() as c:
            row = c.execute("SELECT owner_pid, heartbeat FROM job WHERE id=?", (job_id,)).fetchone()
            if row is None:
                return False
            owner = row["owner_pid"]
            if (owner and owner != pid and pid_alive(owner)
                    and (now - (row["heartbeat"] or 0)) < stale_after_s):
                return False
            c.execute("UPDATE job SET owner_pid=?, heartbeat=? WHERE id=?", (pid, now, job_id))
            return True

    def release(self, job_id, pid: int):
        self.conn.execute("UPDATE job SET owner_pid=NULL WHERE id=? AND owner_pid=?", (job_id, pid))

    def heartbeat(self, job_id, pid: int):
        self.conn.execute("UPDATE job SET heartbeat=? WHERE id=? AND owner_pid=?", (time.time(), job_id, pid))

    def list_jobs(self, states: tuple[str, ...] | None = None, limit: int = 200) -> list[sqlite3.Row]:
        if states:
            q = ",".join("?" * len(states))
            return self.conn.execute(
                f"SELECT * FROM job WHERE state IN ({q}) ORDER BY updated_at DESC LIMIT ?", (*states, limit)
            ).fetchall()
        return self.conn.execute("SELECT * FROM job ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()

    def delete_job(self, job_id) -> None:
        with self.tx() as c:
            c.execute("DELETE FROM word WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM segment WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM chunk WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM job WHERE id=?", (job_id,))
        shutil.rmtree(self.path.parent / job_id, ignore_errors=True)

    def reset_for_rerun(self, job_id) -> None:
        """Reinicia um job FAILED/DONE do zero (mantém o WAV se válido)."""
        with self.tx() as c:
            c.execute("DELETE FROM word WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM segment WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM chunk WHERE job_id=?", (job_id,))
            c.execute("UPDATE job SET state='CREATED', error=NULL, warnings=NULL, finished_at=NULL, "
                      "updated_at=? WHERE id=?", (time.time(), job_id))

    # ---------- chunks ----------
    def insert_plan(self, job_id, plans):
        with self.tx() as c:
            c.execute("DELETE FROM word WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM segment WHERE job_id=?", (job_id,))
            c.execute("DELETE FROM chunk WHERE job_id=?", (job_id,))
            c.executemany(
                "INSERT INTO chunk(job_id, idx, start_sample, end_sample, pad_left, pad_right) VALUES (?,?,?,?,?,?)",
                [(job_id, p.idx, p.start, p.end, p.pad_left, p.pad_right) for p in plans],
            )

    def reset_running(self, job_id) -> int:
        """Chunks 'running' de um processo que morreu: voltam a 'pending', ou 'failed' se excederam tentativas."""
        with self.tx() as c:
            c.execute(
                "UPDATE chunk SET state='failed', error=COALESCE(error, 'o processo worker morreu repetidamente neste trecho') "
                "WHERE job_id=? AND state='running' AND attempts>=?", (job_id, MAX_CHUNK_ATTEMPTS))
            cur = c.execute("UPDATE chunk SET state='pending' WHERE job_id=? AND state='running'", (job_id,))
            return cur.rowcount

    def pending_chunks(self, job_id) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunk WHERE job_id=? AND (state='pending' OR (state='failed' AND attempts<?)) ORDER BY idx",
            (job_id, MAX_CHUNK_ATTEMPTS),
        ).fetchall()

    def chunks(self, job_id) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chunk WHERE job_id=? ORDER BY idx", (job_id,)).fetchall()

    def failed_chunks(self, job_id) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunk WHERE job_id=? AND state='failed' ORDER BY idx", (job_id,)).fetchall()

    def previous_tail(self, job_id, idx) -> tuple[str | None, float | None]:
        row = self.conn.execute(
            "SELECT tail_text, tail_logprob FROM chunk WHERE job_id=? AND idx=? AND state='done'", (job_id, idx - 1)
        ).fetchone()
        return (row["tail_text"], row["tail_logprob"]) if row else (None, None)

    def mark_running(self, job_id, idx):
        self.conn.execute(
            "UPDATE chunk SET state='running', attempts=attempts+1 WHERE job_id=? AND idx=?", (job_id, idx)
        )

    def mark_pending(self, job_id, idx):
        self.conn.execute("UPDATE chunk SET state='pending', attempts=MAX(0, attempts-1) WHERE job_id=? AND idx=?",
                          (job_id, idx))

    def mark_failed(self, job_id, idx, error: str):
        self.conn.execute("UPDATE chunk SET state='failed', error=? WHERE job_id=? AND idx=?",
                          (error[:4000], job_id, idx))

    def commit_chunk(self, job_id, idx, segments: list[dict], meta: dict):
        """Transação única e idempotente: o chunk inteiro entra ou nada entra."""
        with self.tx() as c:
            c.execute("DELETE FROM word WHERE job_id=? AND chunk_idx=?", (job_id, idx))
            c.execute("DELETE FROM segment WHERE job_id=? AND chunk_idx=?", (job_id, idx))
            for s_seq, s in enumerate(segments):
                c.execute(
                    """INSERT INTO segment(job_id, chunk_idx, seq, start_ms, end_ms, text,
                                           avg_logprob, no_speech_prob, compression_ratio, speaker)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (job_id, idx, s_seq, s["start_ms"], s["end_ms"], s["text"],
                     s.get("avg_logprob"), s.get("no_speech_prob"), s.get("compression_ratio"), s.get("speaker")),
                )
                c.executemany(
                    "INSERT INTO word(job_id, chunk_idx, seg_seq, seq, start_ms, end_ms, word, prob) VALUES (?,?,?,?,?,?,?,?)",
                    [(job_id, idx, s_seq, w_seq, w["start_ms"], w["end_ms"], w["word"], w["prob"])
                     for w_seq, w in enumerate(s.get("words", []))],
                )
            c.execute(
                """UPDATE chunk SET state='done', error=NULL, device=?, compute_type=?, batch_size=?,
                       audio_s=?, elapsed_s=?, tail_text=?, tail_logprob=?
                   WHERE job_id=? AND idx=?""",
                (meta["device"], meta["compute_type"], meta["batch_size"], meta["audio_s"], meta["elapsed_s"],
                 meta["tail_text"], meta["tail_logprob"], job_id, idx),
            )
            c.execute("UPDATE job SET updated_at=? WHERE id=?", (time.time(), job_id))

    def progress(self, job_id) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT SUM(state='done') AS done, COUNT(*) AS total FROM chunk WHERE job_id=?", (job_id,)
        ).fetchone()
        return int(row["done"] or 0), int(row["total"] or 0)

    def done_samples(self, job_id) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(end_sample-start_sample),0) FROM chunk WHERE job_id=? AND state IN ('done','failed')",
            (job_id,)).fetchone()
        return int(row[0])

    def stats(self, job_id) -> dict:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(audio_s),0) a, COALESCE(SUM(elapsed_s),0) e FROM chunk WHERE job_id=? AND state='done'",
            (job_id,)).fetchone()
        segs = self.conn.execute("SELECT COUNT(*) FROM segment WHERE job_id=?", (job_id,)).fetchone()[0]
        words = self.conn.execute("SELECT COUNT(*) FROM word WHERE job_id=?", (job_id,)).fetchone()[0]
        return {"audio_s": row["a"], "elapsed_s": row["e"], "segments": segs, "words": words}

    # ---------- leitura de resultados ----------
    def iter_segments(self, job_id) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT * FROM segment WHERE job_id=? ORDER BY start_ms, chunk_idx, seq", (job_id,)
        )

    def iter_words(self, job_id) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT w.*, s.speaker FROM word w JOIN segment s "
            "ON s.job_id=w.job_id AND s.chunk_idx=w.chunk_idx AND s.seq=w.seg_seq "
            "WHERE w.job_id=? ORDER BY w.chunk_idx, w.seg_seq, w.seq", (job_id,)
        )

    def segments_with_words(self, job_id) -> list[dict]:
        """Carrega segmentos com palavras (tuplas compactas). Duas varreduras sequenciais, O(n)."""
        words: dict[tuple[int, int], list[tuple]] = {}
        for r in self.conn.execute(
                "SELECT chunk_idx, seg_seq, start_ms, end_ms, word, prob FROM word WHERE job_id=? "
                "ORDER BY chunk_idx, seg_seq, seq", (job_id,)):
            words.setdefault((r[0], r[1]), []).append((r[2], r[3], r[4], r[5]))
        out = []
        for r in self.conn.execute(
                "SELECT chunk_idx, seq, start_ms, end_ms, text, avg_logprob, speaker FROM segment "
                "WHERE job_id=? ORDER BY chunk_idx, seq", (job_id,)):
            out.append({"start_ms": r[2], "end_ms": r[3], "text": r[4], "avg_logprob": r[5],
                        "speaker": r[6], "words": words.get((r[0], r[1]), [])})
        out.sort(key=lambda s: s["start_ms"])
        return out


class Heartbeat:
    """Thread que renova o lease do job a cada `interval` s, com conexão SQLite própria."""

    def __init__(self, db_path: Path | str, job_id: str, pid: int, interval: float = 5.0):
        import threading

        self._stop = threading.Event()
        self._args = (str(db_path), job_id, pid, interval)
        self._t = threading.Thread(target=self._run, name="journal-heartbeat", daemon=True)

    def _run(self):
        db_path, job_id, pid, interval = self._args
        conn = sqlite3.connect(db_path, timeout=30, isolation_level=None)
        try:
            while not self._stop.wait(interval):
                try:
                    conn.execute("UPDATE job SET heartbeat=? WHERE id=? AND owner_pid=?", (time.time(), job_id, pid))
                except sqlite3.Error:
                    pass
        finally:
            conn.close()

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=2)

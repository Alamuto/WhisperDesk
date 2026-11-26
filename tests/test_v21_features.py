"""Testes da v2.1: edição de trechos, falantes, migração do journal, modo ao vivo e fila."""
import os
import sqlite3

import numpy as np
import pytest

from conftest import seg, word
from app.pipeline import diarize, planner
from app.pipeline.journal import Journal, file_fingerprint, job_id_for, realign_words
from app.pipeline.live_worker import VoiceGate, live_filter


def _meta():
    return {"device": "cpu", "compute_type": "int8", "batch_size": 1, "audio_s": 10.0, "elapsed_s": 1.0,
            "tail_text": "x", "tail_logprob": -0.1}


def _job_with_text(j: Journal, tmp_path):
    src = tmp_path / "media.bin"
    src.write_bytes(b"\x01" * 2000)
    fp = file_fingerprint(str(src))
    cfg = {"model_size": "tiny"}
    jid = job_id_for(fp, cfg)
    j.upsert_job(jid, str(src), fp, cfg, 0, 30.0)
    j.insert_plan(jid, [planner.ChunkPlan(0, 0, 100, 0, 0)])
    j.mark_running(jid, 0)
    j.commit_chunk(jid, 0, [
        seg([word(0, 400, " olá"), word(400, 900, " mundo")], speaker="Falante 1"),
        seg([word(1000, 1500, " tudo"), word(1500, 2000, " bem")], speaker="Falante 2"),
    ], _meta())
    return jid


# ------------------------------------------------------------------ realinhamento
def test_realign_same_count_keeps_timing():
    old = [(0, 400, " olá", 0.4), (400, 900, " mundo", 0.9)]
    out = realign_words(old, "olá Mundo", 0, 900)
    assert [(w[0], w[1]) for w in out] == [(0, 400), (400, 900)]
    assert out[0][3] == 0.4          # palavra igual: mantém a confiança
    assert out[1][3] == 1.0          # corrigida pelo usuário: confiança total


def test_realign_different_count_spreads_inside_speech():
    old = [(100, 400, " a", 0.9), (400, 900, " b", 0.9)]
    out = realign_words(old, "um dois três", 0, 2000)
    assert out[0][0] == 100 and out[-1][1] == 900
    assert all(a[1] <= b[0] for a, b in zip(out, out[1:]))
    assert realign_words(old, "   ", 0, 900) == []


# ------------------------------------------------------------------ journal
def test_edit_and_restore_segment(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job_with_text(j, tmp_path)
    new_words = realign_words([(0, 400, " olá", 0.9), (400, 900, " mundo", 0.9)], "Olá, gente", 0, 900)
    j.edit_segment(jid, (0, 0), "Olá, gente", new_words)
    j.edit_segment(jid, (0, 0), "Olá, pessoal", new_words)   # 2ª edição não perde o original
    segs = j.segments_with_words(jid)
    assert segs[0]["text"] == "Olá, pessoal" and segs[0]["edited"]
    assert not segs[1]["edited"]
    assert j.restore_segment(jid, (0, 0))
    segs = j.segments_with_words(jid)
    assert segs[0]["text"] == "olá mundo" and not segs[0]["edited"]
    assert [w[2] for w in segs[0]["words"]] == [" olá", " mundo"]
    assert not j.restore_segment(jid, (0, 0))
    with pytest.raises(KeyError):
        j.edit_segment(jid, (0, 9), "x", [])
    j.close()


def test_speakers_rename_and_reassign(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job_with_text(j, tmp_path)
    assert j.speakers(jid) == ["Falante 1", "Falante 2"]
    assert j.rename_speaker(jid, "Falante 1", "Ana") == 1
    j.set_segment_speaker(jid, (0, 1), "Ana")
    assert j.speakers(jid) == ["Ana"]
    j.close()


def test_schema_v1_migrates(tmp_path):
    db = tmp_path / "j.db"
    j = Journal(db)
    jid = _job_with_text(j, tmp_path)
    j.close()
    # Simula um banco da v2.0 (sem as colunas novas).
    c = sqlite3.connect(db)
    for sql in ("ALTER TABLE segment DROP COLUMN orig_text", "ALTER TABLE segment DROP COLUMN orig_words",
                "ALTER TABLE job DROP COLUMN diarized", "PRAGMA user_version=1"):
        c.execute(sql)
    c.commit()
    c.close()
    j = Journal(db)
    assert j.conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert j.job(jid)["diarized"] == 0
    assert j.segments_with_words(jid)[0]["text"] == "olá mundo"
    j.close()


def test_service_edit_segment(tmp_path):
    from app.pipeline import service

    jid = _job_with_text(service.journal(), tmp_path)
    d = service.edit_segment(jid, (0, 1), "  tudo   ótimo ")
    assert d["text"] == "tudo ótimo" and d["edited"] and d["key"] == (0, 1)
    assert [w[2].strip() for w in d["words"]] == ["tudo", "ótimo"]
    assert service.restore_segment(jid, (0, 1))["text"] == "tudo bem"
    assert service.restore_segment(jid, (0, 1)) is None


# ------------------------------------------------------------------ diarização
def _voices(rng, centers, per):
    out, truth = [], []
    for k, c in enumerate(centers):
        for _ in range(per):
            v = c + rng.normal(0, 0.07, c.shape)
            out.append(v / np.linalg.norm(v))
            truth.append(k)
    return np.array(out, np.float32), np.array(truth)


def _same_partition(a, b):
    return all((a[i] == a[j]) == (b[i] == b[j]) for i in range(len(a)) for j in range(len(a)))


def test_cluster_auto_finds_voices():
    rng = np.random.default_rng(3)
    centers = [rng.normal(size=64) for _ in range(3)]
    centers = [c / np.linalg.norm(c) for c in centers]
    emb, truth = _voices(rng, centers, 8)
    labels = diarize.cluster(emb)
    assert _same_partition(labels, truth)
    assert labels[0] == 0  # rótulos pela ordem de aparição


def test_cluster_fixed_respects_k():
    rng = np.random.default_rng(5)
    centers = [rng.normal(size=64) for _ in range(2)]
    centers = [c / np.linalg.norm(c) for c in centers]
    emb, truth = _voices(rng, centers, 10)
    labels = diarize.cluster_fixed(emb, 2, np.ones(len(emb)))
    assert len(set(labels.tolist())) == 2
    assert _same_partition(labels, truth)


def test_cluster_degenerate_sizes():
    assert diarize.cluster(np.zeros((0, 8), np.float32)).tolist() == []
    assert diarize.cluster(np.ones((1, 8), np.float32)).tolist() == [0]


# ------------------------------------------------------------------ ao vivo
def test_voice_gate_adapts_to_noise():
    g = VoiceGate()
    rng = np.random.default_rng(1)
    noise = lambda: rng.normal(0, 0.003, 480).astype(np.float32)  # noqa: E731
    for _ in range(200):
        g.frame(noise())
    assert not g.frame(noise())[0]
    t = np.arange(480) / 16000
    assert g.frame((0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))[0]


def test_live_filter_drops_low_confidence_and_echo():
    weak = seg([word(0, 300, " é", 0.2), word(300, 500, " ah", 0.3)])
    echo = seg([word(0, 300, " bom", 0.9), word(300, 600, " dia", 0.9), word(600, 900, " pessoal", 0.9)])
    good = seg([word(0, 300, " vamos", 0.9), word(300, 600, " começar", 0.8)])
    out = live_filter([weak, echo, good], "Bom dia, pessoal.")
    assert out == [good]


def test_to_mono_16k_resamples_continuously():
    pytest.importorskip("PySide6.QtMultimedia")
    from PySide6.QtMultimedia import QAudioFormat
    from app.gui.live import _to_mono_16k

    fmt = QAudioFormat()
    fmt.setSampleRate(48000)
    fmt.setChannelCount(2)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Float)
    t = np.arange(48000) / 48000
    mono = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    stereo = np.repeat(mono, 2)
    state: dict = {}
    parts = [_to_mono_16k(stereo[i:i + 1922].tobytes(), fmt, state) for i in range(0, len(stereo), 1922)]
    out = np.concatenate(parts)
    assert abs(len(out) - 16000) <= 2
    ref = np.sin(2 * np.pi * 300 * np.arange(len(out)) / 16000)
    assert np.max(np.abs(out - ref)) < 0.02          # sem saltos nas emendas dos blocos

    fmt16 = QAudioFormat()
    fmt16.setSampleRate(16000)
    fmt16.setChannelCount(1)
    fmt16.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    x = _to_mono_16k(np.array([0, 16384, -32768], np.int16).tobytes(), fmt16, {})
    assert np.allclose(x, [0, 0.5, -1.0])


# ------------------------------------------------------------------ interface (offscreen)
@pytest.fixture
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_queue_list_add_remove_persist(qapp, tmp_path):
    from app.gui.queue_list import QueueList
    from app.utils import config as app_config

    files = []
    for n in ("a.mp3", "b.wav", "c.mp4"):
        f = tmp_path / n
        f.write_bytes(b"x")
        files.append(str(f))
    q = QueueList()
    counts = []
    q.changed.connect(counts.append)
    assert q.add(files) == 3
    assert q.add([files[0].upper()]) == 0          # sem duplicatas (caminho sem diferenciar maiúsculas)
    assert len(q) == 3 and app_config.get("queue") == [os.path.normpath(f) for f in files]
    q.remove(files[1])
    assert q.items() == [os.path.normpath(files[0]), os.path.normpath(files[2])]
    os.remove(files[0])
    assert q.pop_next() == os.path.normpath(files[2])  # pula o que sumiu do disco
    assert len(q) == 0 and counts[-1] == 0

    q2 = QueueList()
    app_config.set_value("queue", [files[1], str(tmp_path / "nao-existe.mp3")])
    q2.restore()
    assert q2.items() == [os.path.normpath(files[1])]


def test_segment_body_strips_only_speaker_prefix(qapp):
    from app.gui.result_view import _Seg

    s = _Seg({"start_ms": 0, "end_ms": 900, "speaker": "Carla", "key": (0, 0),
              "words": [(0, 400, " Antes", 0.9), (400, 900, " disso", 0.9)]})
    assert s.text == "Carla: Antes disso"
    assert s.body == "Antes disso"
    assert _Seg({"start_ms": 0, "end_ms": 1, "text": " Oi "}).body == "Oi"

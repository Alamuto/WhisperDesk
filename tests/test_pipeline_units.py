"""Testes unitários do pipeline: planejamento, anti-alucinação, re-segmentação, exportação, journal."""
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline import exporters, hallucination, planner, segmenter
from app.pipeline.journal import Journal, file_fingerprint, job_id_for, pid_alive
from tests.conftest import seg, word

SR = planner.SR


# ------------------------------------------------------------------ planner
def _tone_with_gaps(path: Path, seconds: int, gaps_at: list[float], gap_s: float = 0.8):
    t = np.arange(seconds * SR) / SR
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for g in gaps_at:
        a = int(g * SR)
        audio[a:a + int(gap_s * SR)] = 0.0
    sf.write(str(path), audio, SR, subtype="PCM_16")
    return len(audio)


def _energy_gaps(audio):
    """Substituto determinístico do VAD (o Silero não considera tom puro como fala)."""
    frame = SR // 100
    n = len(audio) // frame
    quiet = np.abs(audio[: n * frame].reshape(n, frame)).max(axis=1) < 1e-3
    gaps, start = [], None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            gaps.append((start * frame, i * frame))
            start = None
    if start is not None:
        gaps.append((start * frame, n * frame))
    return gaps


def test_planner_cuts_inside_silence(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "_gaps", _energy_gaps)
    wav = tmp_path / "a.wav"
    n = _tone_with_gaps(wav, 100, gaps_at=[27.0, 55.0, 81.0])
    plans = planner.plan_chunks(str(wav), n, target_s=30, search_s=8)
    assert plans[0].start == 0 and plans[-1].end == n
    for a, b in zip(plans, plans[1:]):
        assert a.end == b.start                       # cobertura contígua, sem buracos nem sobreposição
    for p in plans[1:]:
        cut_s = p.start / SR
        assert any(g <= cut_s <= g + 0.8 for g in (27.0, 55.0, 81.0)), cut_s
        assert p.pad_left == 0 and not p.forced


def test_planner_forced_cut_gets_padding(tmp_path, monkeypatch):
    monkeypatch.setattr(planner, "_gaps", _energy_gaps)
    wav = tmp_path / "b.wav"
    n = _tone_with_gaps(wav, 80, gaps_at=[])          # sem silêncio algum
    plans = planner.plan_chunks(str(wav), n, target_s=30, search_s=5)
    assert len(plans) >= 2
    assert all(p.forced for p in plans)
    assert plans[1].pad_left > 0 and plans[0].pad_right > 0
    assert plans[0].start == 0 and plans[-1].end == n


SPEECH = Path(__file__).parent / "data" / "speech_en.wav"


@pytest.mark.skipif(not SPEECH.is_file(), reason="amostra de fala ausente")
def test_planner_real_vad_cuts_in_low_energy(tmp_path):
    """Com o Silero real, sobre fala de verdade: cada corte cai num trecho de energia baixa."""
    data, sr = sf.read(str(SPEECH), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SR:
        idx = (np.arange(int(len(data) * SR / sr)) * sr / SR).astype(int)
        data = data[idx]
    wav = tmp_path / "speech16k.wav"
    sf.write(str(wav), data, SR, subtype="PCM_16")
    plans = planner.plan_chunks(str(wav), len(data), target_s=20, search_s=6)
    assert len(plans) >= 2
    speech_rms = float(np.sqrt(np.mean(data ** 2)))
    for p in plans[1:]:
        if p.forced:
            continue
        win = data[max(0, p.start - SR // 20): p.start + SR // 20]
        assert float(np.sqrt(np.mean(win ** 2))) < speech_rms * 0.5


def test_planner_short_audio_single_chunk(tmp_path):
    wav = tmp_path / "c.wav"
    n = _tone_with_gaps(wav, 10, gaps_at=[])
    plans = planner.plan_chunks(str(wav), n)
    assert len(plans) == 1 and plans[0].start == 0 and plans[0].end == n


# ------------------------------------------------------------------ alucinação
def test_known_phrase_dropped_only_in_nonspeech():
    w = [word(0, 1500, " Obrigado"), word(1500, 2500, " por"), word(2500, 3500, " assistir")]
    assert hallucination.clean_segments([seg(w, no_speech_prob=0.6)], "pt") == []
    assert len(hallucination.clean_segments([seg(w, no_speech_prob=0.01, avg_logprob=-0.1)], "pt")) == 1


def test_repeated_segments_capped():
    segs = [seg([word(i * 2000, i * 2000 + 1500, " Sim.")]) for i in range(6)]
    out = hallucination.clean_segments(segs, "pt")
    assert len(out) == 2


def test_word_loop_collapsed():
    ws = [word(i * 300, i * 300 + 250, " ok") for i in range(20)]
    out = hallucination.clean_segments([seg(ws)], "en")
    assert len(out) == 1 and len(out[0]["words"]) <= 3


def test_impossible_speech_rate_dropped():
    s = seg([word(0, 100, " " + "palavra" * 10)])
    assert hallucination.clean_segments([s], "pt") == []


def test_normal_speech_untouched():
    ws = [word(0, 400, " Bom"), word(400, 800, " dia"), word(800, 1300, " a"), word(1300, 1900, " todos.")]
    out = hallucination.clean_segments([seg(ws)], "pt")
    assert out and out[0]["text"] == "Bom dia a todos."


# ------------------------------------------------------------------ segmenter
def test_resegment_splits_sentences_keeping_timestamps():
    ws = [word(0, 500, " Primeira"), word(500, 1500, " frase."), word(1600, 2200, " Segunda"),
          word(2200, 3000, " frase.")]
    out = segmenter.resegment([seg(ws)])
    assert [s["text"] for s in out] == ["Primeira frase.", "Segunda frase."]
    assert out[0]["start_ms"] == 0 and out[1]["start_ms"] == 1600 and out[1]["end_ms"] == 3000
    assert sum(len(s["words"]) for s in out) == 4


def test_resegment_splits_on_long_pause():
    ws = [word(0, 500, " antes"), word(3000, 3500, " depois")]
    out = segmenter.resegment([seg(ws)])
    assert len(out) == 2


# ------------------------------------------------------------------ exporters
def _segments_for_export():
    words = []
    t = 0
    for i, w in enumerate("Esta é uma frase de teste bastante longa para quebrar em linhas.".split()):
        words.append((t, t + 350, " " + w, 0.9))
        t += 400
    words2 = [(t + 2000, t + 2600, " Fim.", 0.8)]
    return [{"start_ms": words[0][0], "end_ms": words[-1][1], "text": "".join(w[2] for w in words).strip(),
             "avg_logprob": -0.2, "speaker": None, "words": words},
            {"start_ms": words2[0][0], "end_ms": words2[0][1], "text": "Fim.", "avg_logprob": -0.3,
             "speaker": None, "words": words2}]


def test_fmt_ts():
    assert exporters.fmt_ts(3_723_456, ",") == "01:02:03,456"
    assert exporters.fmt_ts(-5) == "00:00:00.000"
    assert exporters.fmt_ts(13 * 3_600_000) == "13:00:00.000"   # > 12 h sem estouro


@pytest.mark.parametrize("fmt", list(exporters.EXPORT_FORMATS))
def test_export_all_formats(tmp_path, fmt):
    out = tmp_path / f"out{exporters.EXPORT_FORMATS[fmt][1]}"
    exporters.export_segments(_segments_for_export(), fmt, out, {"source": "x"})
    text = out.read_text(encoding="utf-8-sig")
    assert "Fim." in text
    assert not list(tmp_path.glob("*.tmp"))
    if fmt == "json":
        data = json.loads(text)
        assert data["segments"][1]["text"] == "Fim."
    if fmt == "srt":
        assert text.startswith("1\n00:00:00,000 --> ")


def test_srt_rules():
    cues = exporters.build_cues(exporters._words_flat(_segments_for_export()))
    for c in cues:
        assert 1 <= len(c.lines) <= exporters.MAX_LINES
        assert all(len(line) <= exporters.MAX_LINE for line in c.lines)
        assert c.end_ms - c.start_ms <= exporters.MAX_DUR_MS
    for a, b in zip(cues, cues[1:]):
        assert a.end_ms <= b.start_ms                      # nunca sobrepõe


def test_atomic_write_keeps_old_file_on_error(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("antigo", encoding="utf-8")

    def boom(f):
        f.write("parcial")
        raise RuntimeError("falha no meio")

    with pytest.raises(RuntimeError):
        exporters.atomic_write(p, boom)
    assert p.read_text(encoding="utf-8") == "antigo"
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ journal
def _meta():
    return {"device": "cpu", "compute_type": "int8", "batch_size": 1, "audio_s": 10.0, "elapsed_s": 1.0,
            "tail_text": "x", "tail_logprob": -0.1}


def _job(j: Journal, tmp_path):
    src = tmp_path / "media.bin"
    src.write_bytes(b"\x00" * 1000)
    fp = file_fingerprint(str(src))
    cfg = {"model_size": "tiny"}
    jid = job_id_for(fp, cfg)
    j.upsert_job(jid, str(src), fp, cfg, 0, 30.0)
    j.insert_plan(jid, [planner.ChunkPlan(i, i * 100, (i + 1) * 100, 0, 0) for i in range(3)])
    return jid


def test_job_id_stable_and_config_sensitive(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc" * 1000)
    fp = file_fingerprint(str(f))
    a = {"model_size": "tiny", "language": "pt", "device": "cuda"}
    assert job_id_for(fp, a) == job_id_for(fp, dict(reversed(list(a.items()))))
    assert job_id_for(fp, a) == job_id_for(fp, {**a, "device": "cpu"})       # device não muda a identidade
    assert job_id_for(fp, a) != job_id_for(fp, {**a, "model_size": "small"})
    assert job_id_for(fp, a) != job_id_for(fp, {**a, "language": None})


def test_commit_chunk_idempotent(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path)
    segs = [seg([word(0, 500, " olá"), word(500, 900, " mundo")])]
    for _ in range(3):                                   # retentativas não duplicam
        j.mark_running(jid, 0)
        j.commit_chunk(jid, 0, segs, _meta())
    st = j.stats(jid)
    assert st["segments"] == 1 and st["words"] == 2
    assert j.progress(jid) == (1, 3)
    assert [c["idx"] for c in j.pending_chunks(jid)] == [1, 2]
    j.close()


def test_running_chunks_recovered_after_crash(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path)
    j.mark_running(jid, 1)
    assert j.reset_running(jid) == 1
    assert [c["idx"] for c in j.pending_chunks(jid)] == [0, 1, 2]
    # chunk que derruba o processo repetidamente acaba marcado como failed (não trava o job)
    for _ in range(3):
        j.mark_running(jid, 2)
        j.reset_running(jid)
    assert [c["idx"] for c in j.failed_chunks(jid)] == [2]
    j.close()


def test_claim_lease(tmp_path):
    import os

    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path)
    assert j.claim(jid, os.getpid())
    assert not j.claim(jid, os.getpid() + 1)           # dono vivo e com heartbeat recente
    j.release(jid, os.getpid())
    assert j.claim(jid, os.getpid() + 1)
    j.close()


def test_pid_alive():
    import os

    assert pid_alive(os.getpid())
    assert not pid_alive(None)
    assert not pid_alive(2 ** 30)


def test_concurrent_commits_do_not_corrupt(tmp_path):
    db = tmp_path / "j.db"
    j = Journal(db)
    jid = _job(j, tmp_path)
    errors = []

    def worker(idx):
        try:
            jj = Journal(db)
            for _ in range(20):
                jj.commit_chunk(jid, idx, [seg([word(idx * 100, idx * 100 + 50, f" w{idx}")])], _meta())
            jj.close()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors
    assert j.stats(jid)["segments"] == 3
    assert j.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    j.close()


def test_reset_for_rerun_clears_results(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path)
    j.commit_chunk(jid, 0, [seg([word(0, 500, " a")])], _meta())
    j.set_state(jid, "DONE")
    j.reset_for_rerun(jid)
    assert j.job(jid)["state"] == "CREATED"
    assert j.stats(jid)["segments"] == 0
    j.close()

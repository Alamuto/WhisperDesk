"""Testes da v2.2: revisão contextual (sugestões para trechos de baixa confiança)."""
import os
import sqlite3
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from conftest import seg, word
from app.pipeline import planner, refine
from app.pipeline.journal import Journal, file_fingerprint, job_id_for


def _meta():
    return {"device": "cpu", "compute_type": "int8", "batch_size": 1, "audio_s": 10.0, "elapsed_s": 1.0,
            "tail_text": "x", "tail_logprob": -0.1}


def _t(words):
    """Palavras no formato do journal: (início, fim, palavra, prob)."""
    return [(w["start_ms"], w["end_ms"], w["word"], w["prob"]) for w in words]


# ------------------------------------------------------------------ regras puras
def test_is_suspect_and_edited_are_final():
    ok = seg([word(0, 400, " tudo", 0.9), word(400, 800, " bem", 0.8)])
    ok["words"] = _t(ok["words"])
    bad = seg([word(0, 400, " cuber", 0.2), word(400, 800, " nets", 0.4)])
    bad["words"] = _t(bad["words"])
    assert not refine.is_suspect(ok)
    assert refine.is_suspect(bad)
    assert not refine.is_suspect({**bad, "edited": True})
    assert refine.is_suspect({**ok, "avg_logprob": -1.5})


def test_context_prompt_drops_doubtful_words():
    segs = [{"words": [(0, 1, " Falamos", 0.9), (1, 2, " de", 0.9), (2, 3, " xyz", 0.1)]},
            {"words": [(3, 4, " Kubernetes", 0.95), (4, 5, " hoje.", 0.9)]},
            {"words": [(5, 6, " alvo", 0.2)]}]
    p = refine.context_prompt(segs, 2)
    assert "Kubernetes" in p and "Falamos" in p
    assert "xyz" not in p
    assert refine.context_prompt(segs, 0) is None


def test_glossary_keeps_confident_proper_nouns_and_repeated_terms():
    words = [(0, 1, " Hoje", 0.99), (1, 2, " o", 0.9), (2, 3, " Kubernetes", 0.95), (3, 4, " orquestra", 0.9),
             (4, 5, " contêineres", 0.95), (5, 6, " e", 0.9), (6, 7, " contêineres.", 0.92),
             (7, 8, " Talvez", 0.4), (8, 9, " Docker", 0.3)]
    g = refine.glossary([{"words": words}])
    assert "Kubernetes" in g.split()
    assert "contêineres" in g.split()
    assert "Hoje" not in g.split()  # início de frase não conta como nome próprio
    assert "Docker" not in g.split()  # baixa confiança não entra no vocabulário


def test_choose_requires_change_and_clear_gain():
    orig = [(0, 400, " cuber", 0.21), (400, 800, " nets.", 0.33)]
    good = [(0, 800, " Kubernetes.", 0.88)]
    same = [(0, 400, " cuber", 0.6), (400, 800, " nets.", 0.7)]
    weak = [(0, 800, " Cubanets.", 0.35)]
    assert refine.choose(orig, [same, weak, good], None) == good
    assert refine.choose(orig, [same], None) is None
    assert refine.choose(orig, [weak], None) is None
    assert refine.choose(orig, [[]], None) is None


def test_choose_rejects_prompt_echo_and_loops():
    orig = [(0, 300, " a", 0.3), (300, 600, " b", 0.3), (600, 900, " c", 0.3), (900, 1200, " d", 0.3)]
    echo = [(0, 300, " vamos", 0.9), (300, 600, " falar", 0.9), (600, 900, " de", 0.9), (900, 1200, " dados", 0.9)]
    assert refine.choose(orig, [echo], "Hoje vamos falar de dados com calma") is None
    orig6 = [(i * 100, i * 100 + 90, f" w{i}", 0.3) for i in range(6)]
    loop = [(i * 100, i * 100 + 90, " sim", 0.95) for i in range(6)]
    assert refine.choose(orig6, [loop], None) is None


def test_changed_tokens_marks_only_differences():
    assert refine.changed_tokens("Hoje vamos falar do cuber nets.", "Hoje vamos falar do Kubernetes.") == [4]
    assert refine.changed_tokens("Olá mundo", "olá, mundo!") == []


def test_owned_words_keeps_only_the_segment_window():
    w = lambda s, e, t, p: SimpleNamespace(start=s, end=e, word=t, probability=p)  # noqa: E731
    raw = [SimpleNamespace(words=[w(0.0, 0.5, " antes", 0.9), w(1.5, 2.0, " dentro", 0.8), w(3.6, 4.0, " depois", 0.9)])]
    got = refine.owned_words(raw, 1000, 2400, 3000)  # janela começa em 1 s; trecho 2,4–3,0 s
    assert [x[2] for x in got] == [" dentro"]
    assert got[0][:2] == (2500, 3000)


# ------------------------------------------------------------------ journal + refine_job
def _job(j: Journal, tmp_path, segments):
    src = tmp_path / "media.bin"
    src.write_bytes(b"\x02" * 3000)
    fp = file_fingerprint(str(src))
    cfg = {"model_size": "tiny"}
    jid = job_id_for(fp, cfg)
    j.upsert_job(jid, str(src), fp, cfg, 0, 10.0)
    j.insert_plan(jid, [planner.ChunkPlan(0, 0, 160000, 0, 0)])
    j.mark_running(jid, 0)
    j.commit_chunk(jid, 0, segments, _meta())
    return jid


def _wav(tmp_path, seconds=6.0):
    p = tmp_path / "a.wav"
    rng = np.random.default_rng(0)
    sf.write(p, (rng.standard_normal(int(16000 * seconds)) * 0.01).astype("float32"), 16000)
    return str(p)


def _fake_decoder(text_by_prompt):
    """Decodificador falso: devolve um segmento cobrindo 2,0–3,0 s da janela, conforme o contexto."""
    calls = []

    def decode(audio, prompt, hotwords):
        calls.append((prompt, hotwords))
        word_text, prob = text_by_prompt(prompt, hotwords)
        w = SimpleNamespace(start=1.5, end=2.4, word=word_text, probability=prob)
        return [SimpleNamespace(words=[w])]

    return decode, calls


def test_refine_job_creates_and_resolves_suggestions(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path, [
        seg([word(0, 500, " Falamos", 0.95), word(500, 1000, " do", 0.9), word(1000, 1500, " Kubernetes.", 0.95)]),
        seg([word(2000, 2400, " cuber", 0.2), word(2400, 2900, " nets.", 0.3)]),
    ])
    wav = _wav(tmp_path)
    # janela do trecho 2 começa em 500 ms (2000 − 1500): 1,5–2,4 s → 2000–2900 ms globais
    decode, calls = _fake_decoder(lambda p, h: (" Kubernetes.", 0.9) if h and "Kubernetes" in h else (" cubernetes.", 0.5))
    n = refine.refine_job(j, jid, wav, decode, "pt", on_progress=lambda f: None, cancel_event=threading.Event())
    assert n == 1
    assert calls and "Kubernetes" in (calls[0][0] or "")  # o contexto anterior foi usado como prompt
    segs = j.segments_with_words(jid)
    assert segs[1]["suggestion"]["text"] == "Kubernetes."
    assert j.job(jid)["refined"] == 1

    key = segs[1]["key"]
    assert j.resolve_suggestion(jid, key, accept=True)
    after = j.segments_with_words(jid)[1]
    assert after["text"] == "Kubernetes." and after["edited"] and "suggestion" not in after
    assert not j.resolve_suggestion(jid, key, accept=True)  # já resolvida
    # Restaurar volta ao original; uma nova revisão não reabre o que o usuário já decidiu.
    assert j.restore_segment(jid, key)
    assert refine.refine_job(j, jid, wav, decode, "pt", lambda f: None, threading.Event()) == 0
    j.close()


def test_reject_and_manual_edit_close_the_suggestion(tmp_path):
    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path, [seg([word(2000, 2400, " cuber", 0.2), word(2400, 2900, " nets.", 0.3)])])
    j.add_suggestions(jid, [((0, 0), "Kubernetes.", [(2000, 2900, " Kubernetes.", 0.9)], 0.25, 0.9)])
    assert j.suggestions(jid)
    j.edit_segment(jid, (0, 0), "Kubernetes!", [(2000, 2900, " Kubernetes!", 1.0)])
    assert not j.suggestions(jid)  # editar à mão encerra a sugestão pendente
    j.add_suggestions(jid, [((0, 0), "Cubernetes", [(2000, 2900, " Cubernetes", 0.9)], 0.25, 0.9)])
    assert j.resolve_suggestion(jid, (0, 0), accept=False)
    assert j.segments_with_words(jid)[0]["text"] == "Kubernetes!"
    assert (0, 0) in j.resolved_suggestion_keys(jid)
    j.close()


def test_refine_job_honours_cancel(tmp_path):
    from app.pipeline.ffmpeg_tools import Cancelled

    j = Journal(tmp_path / "j.db")
    jid = _job(j, tmp_path, [seg([word(2000, 2400, " cuber", 0.2), word(2400, 2900, " nets.", 0.3)])])
    ev = threading.Event()
    ev.set()
    decode, _ = _fake_decoder(lambda p, h: (" x", 0.9))
    with pytest.raises(Cancelled):
        refine.refine_job(j, jid, _wav(tmp_path), decode, "pt", lambda f: None, ev)
    assert j.job(jid)["refined"] == 0
    j.close()


def test_schema_v2_migrates_to_v3(tmp_path):
    db = tmp_path / "j.db"
    j = Journal(db)
    jid = _job(j, tmp_path, [seg([word(0, 400, " oi", 0.9)])])
    j.close()
    c = sqlite3.connect(db)
    for sql in ("DROP TABLE suggestion", "ALTER TABLE job DROP COLUMN refined", "PRAGMA user_version=2"):
        c.execute(sql)
    c.commit()
    c.close()
    j = Journal(db)
    assert j.conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert j.job(jid)["refined"] == 0
    assert j.suggestions(jid) == {}
    j.close()


# ------------------------------------------------------------------ interface (offscreen)
def test_result_view_shows_and_applies_suggestion(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from app.gui.result_view import ResultView
    from app.pipeline import service

    j = service.journal()
    jid = _job(j, tmp_path, [seg([word(2000, 2400, " cuber", 0.2), word(2400, 2900, " nets.", 0.3)])])
    j.set_state(jid, "DONE")
    j.add_suggestions(jid, [((0, 0), "Kubernetes.", [(2000, 2900, " Kubernetes.", 0.9)], 0.25, 0.9)])
    v = ResultView()
    v.resize(700, 300)
    v.show_job(jid, service.load_segments(jid))
    app.processEvents()
    assert v.suggestion_total() == 1
    assert v._sugg_reveal.is_shown()
    seg0 = v._model.seg(0)
    assert seg0.sugg_text == "Kubernetes." and seg0.sugg_spans == [(0, len("Kubernetes."))]
    v._resolve_row(0, True)
    assert v.suggestion_total() == 0
    assert v._model.seg(0).body == "Kubernetes."
    assert not v._sugg_reveal.is_shown()
    v.close()
    service.close()

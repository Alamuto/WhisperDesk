"""Testes de integração de mídia (PyAV: probe, extração, picos) e da camada de serviço."""
import threading
from pathlib import Path

import pytest
import soundfile as sf

from tests.conftest import ffmpeg_available

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="PyAV indisponível")


def _make_media(path: Path, seconds: int = 12, two_tracks: bool = False):
    from tests.conftest import make_media

    return make_media(path, seconds, two_tracks)


def test_probe_reports_streams(tmp_path):
    from app.pipeline import ffmpeg_tools

    p = _make_media(tmp_path / "dois audios.mkv", two_tracks=True)
    info = ffmpeg_tools.probe(str(p))
    assert abs(info.duration - 12) < 0.5
    assert len(info.audio_streams) == 2
    assert info.audio_streams[1].language == "eng"
    assert info.default_stream.index == info.audio_streams[0].index
    assert not info.has_video


def test_probe_rejects_non_media(tmp_path):
    from app.pipeline import ffmpeg_tools

    p = tmp_path / "nada.mp4"
    p.write_bytes(b"isto nao e midia" * 100)
    with pytest.raises(ffmpeg_tools.AudioError):
        ffmpeg_tools.probe(str(p))


def test_extract_wav_mono_16k_selected_track(tmp_path):
    from app.pipeline import ffmpeg_tools

    p = _make_media(tmp_path / "unicode çãõ ü.mkv", seconds=8, two_tracks=True)
    info = ffmpeg_tools.probe(str(p))
    dst = tmp_path / "out" / "audio.wav"
    progress = []
    warn = ffmpeg_tools.extract_wav(str(p), dst, info.audio_streams[1], info.duration,
                                    cancel_event=threading.Event(), on_progress=progress.append,
                                    log_path=tmp_path / "ff.log")
    assert warn is None
    data, sr = sf.read(str(dst))
    assert sr == 16000 and data.ndim == 1
    assert abs(len(data) / sr - 8) < 0.2
    assert not list((tmp_path / "out").glob("*.partial*"))


def test_extract_cancel(tmp_path):
    from app.pipeline import ffmpeg_tools

    p = _make_media(tmp_path / "longo.wav", seconds=600)
    info = ffmpeg_tools.probe(str(p))
    ev = threading.Event()
    ev.set()
    with pytest.raises(ffmpeg_tools.Cancelled):
        ffmpeg_tools.extract_wav(str(p), tmp_path / "x.wav", info.default_stream, info.duration,
                                 cancel_event=ev, on_progress=lambda f: None, log_path=tmp_path / "ff.log")
    assert not (tmp_path / "x.wav").exists()


def test_decode_peaks(tmp_path):
    from app.pipeline import ffmpeg_tools

    p = _make_media(tmp_path / "a.wav", seconds=5)
    peaks = ffmpeg_tools.decode_peaks(str(p), None, 200, 5.0, threading.Event())
    assert len(peaks) == 200
    assert max(peaks) > 0.1


def test_service_job_identity_and_export(tmp_path):
    from app.pipeline import ffmpeg_tools, service

    p = _make_media(tmp_path / "s.wav", seconds=3)
    info = ffmpeg_tools.probe(str(p))
    cfg = {"model_size": "tiny", "language": None, "task": "transcribe", "device": "cpu", "compute_type": "int8",
           "vad_filter": True, "beam_size": 5, "temperature": 0.0}
    jid, st = service.create_or_resume_job(str(p), cfg, info.default_stream.index, info.duration)
    assert st == "CREATED"
    # trocar só device/precisão não cria outro job (retomada continua valendo)
    jid2, _ = service.create_or_resume_job(str(p), {**cfg, "device": "cuda", "compute_type": "float16"},
                                           info.default_stream.index, info.duration)
    assert jid2 == jid
    jid3, _ = service.create_or_resume_job(str(p), {**cfg, "model_size": "small"}, info.default_stream.index,
                                           info.duration)
    assert jid3 != jid
    j = service.journal()
    from app.pipeline import planner

    j.insert_plan(jid, [planner.ChunkPlan(0, 0, 48000, 0, 0)])
    j.commit_chunk(jid, 0, [{"start_ms": 100, "end_ms": 900, "text": "teste", "avg_logprob": -0.1,
                             "words": [{"start_ms": 100, "end_ms": 900, "word": " teste", "prob": 0.9}]}],
                   {"device": "cpu", "compute_type": "int8", "batch_size": 1, "audio_s": 3, "elapsed_s": 1,
                    "tail_text": "teste", "tail_logprob": -0.1})
    j.set_state(jid, "DONE")
    assert [r["id"] for r in service.history()] == [jid]
    assert service.job_progress(jid) == 1.0
    out = tmp_path / "r.srt"
    service.export(jid, "srt", out)
    assert "teste" in out.read_text(encoding="utf-8")
    service.delete_job(jid)
    assert service.job(jid) is None
    service.close()

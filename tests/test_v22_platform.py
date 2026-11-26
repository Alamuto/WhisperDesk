"""Testes da v2.2: mídia via PyAV (lacunas, 5.1), pacote GPU sob demanda e utilitários multiplataforma."""
import hashlib
import threading
import zipfile

import numpy as np
import pytest
import soundfile as sf

from app import gpu_pack


# ------------------------------------------------------------------ mídia (PyAV)
def _gap_51(path, sr=48000):
    """MKV 5.1 de 10 s com um buraco de 2 s nos carimbos de tempo e um bipe só no canal central em 8 s."""
    import av

    with av.open(str(path), "w") as out:
        s = out.add_stream("aac", rate=sr, layout="5.1")
        n = 1024
        for start in range(0, 10 * sr, n):
            if 4 * sr <= start < 6 * sr:
                continue
            t = np.arange(start, start + n) / sr
            x = np.zeros((6, n), np.float32)
            x[2] = (0.5 * np.sin(2 * np.pi * 1000 * t) * (np.abs(t - 8.0) < 0.1)).astype(np.float32)
            f = av.AudioFrame.from_ndarray(x, format="fltp", layout="5.1")
            f.sample_rate = sr
            f.pts = start
            for p in s.encode(f):
                out.mux(p)
        for p in s.encode(None):
            out.mux(p)
    return path


def test_extract_keeps_container_clock_and_prioritizes_center(tmp_path):
    from app.pipeline import ffmpeg_tools as ft

    src = _gap_51(tmp_path / "gap.mkv")
    info = ft.probe(str(src))
    assert info.default_stream.layout.startswith("5.1")
    dst = tmp_path / "a.wav"
    ft.extract_wav(str(src), dst, info.default_stream, info.duration, cancel_event=threading.Event(),
                   on_progress=lambda f: None, log_path=tmp_path / "l.log")
    a, sr = sf.read(dst)
    assert abs(len(a) / sr - 10.0) < 0.1          # o buraco virou silêncio, não sumiu
    onset = np.argmax(np.abs(a) > 0.05) / sr
    assert abs(onset - 7.9) < 0.03                 # bipe no lugar certo do relógio
    assert 0.25 < np.abs(a).max() < 0.35           # canal central com peso 0,6
    peaks = ft.decode_peaks(str(src), None, 100, info.duration)
    assert int(peaks.argmax()) in (79, 80)


def test_backend_version_reports_pyav():
    from app.pipeline import ffmpeg_tools as ft

    assert ft.backend_version().startswith("PyAV")


# ------------------------------------------------------------------ pacote GPU
def _fake_wheel(tmp_path, files):
    whl = tmp_path / "fake.whl"
    with zipfile.ZipFile(whl, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    data = whl.read_bytes()
    return whl, hashlib.sha256(data).hexdigest(), len(data)


@pytest.fixture
def fake_pack(tmp_path, monkeypatch):
    whl, sha, size = _fake_wheel(tmp_path, {
        "nvidia/cublas/bin/cublas64_12.dll": b"a" * 1000,
        "nvidia/cublas/bin/cublasLt64_12.dll": b"b" * 2000,
        "nvidia/cublas/bin/nvblas64_12.dll": b"c",
        "nvidia/cublas/include/cublas.h": b"h",
    })
    w = gpu_pack.Wheel(whl.as_uri(), sha, size, ("nvidia/cublas/bin/",), ("cublas64_12.dll", "cublasLt64_12.dll"))
    monkeypatch.setattr(gpu_pack, "platform_key", lambda: "win")
    monkeypatch.setattr(gpu_pack, "PACKS", {"win": (w,)})
    monkeypatch.setattr(gpu_pack, "pack_dir", lambda: tmp_path / "gpu")
    return w


def test_gpu_pack_installs_only_needed_files(fake_pack, tmp_path):
    seen = []
    assert not gpu_pack.is_installed()
    out = gpu_pack.install(lambda d, t: seen.append((d, t)))
    assert sorted(p.name for p in out.iterdir()) == ["cublas64_12.dll", "cublasLt64_12.dll", "pack.json"]
    assert gpu_pack.is_installed()
    assert seen[-1] == (fake_pack.size, fake_pack.size)
    (out / "cublasLt64_12.dll").unlink()
    assert not gpu_pack.is_installed()           # instalação incompleta é detectada


def test_gpu_pack_rejects_bad_checksum(fake_pack, monkeypatch):
    bad = gpu_pack.Wheel(fake_pack.url, "0" * 64, fake_pack.size, fake_pack.members, fake_pack.names)
    monkeypatch.setattr(gpu_pack, "PACKS", {"win": (bad,)})
    with pytest.raises(gpu_pack.PackError):
        gpu_pack.install(lambda d, t: None)
    assert not gpu_pack.is_installed()


def test_gpu_pack_cancel(fake_pack):
    ev = threading.Event()
    ev.set()
    with pytest.raises(gpu_pack.Cancelled):
        gpu_pack.install(lambda d, t: None, ev)
    assert not gpu_pack.is_installed()


def test_gpu_pack_pins_are_well_formed():
    for wheels in gpu_pack.PACKS.values():
        for w in wheels:
            assert w.url.startswith("https://files.pythonhosted.org/") and w.url.endswith(".whl")
            assert len(w.sha256) == 64 and int(w.sha256, 16) >= 0
            assert w.size > 100 << 20


# ------------------------------------------------------------------ utilitários de sistema
def test_desktop_fonts_and_paths(monkeypatch, tmp_path):
    from app import paths
    from app.utils import desktop

    assert desktop.mono_families() and desktop.ui_font_family()
    monkeypatch.setenv("WHISPERDESK_HOME", str(tmp_path))
    assert paths.local_dir() == tmp_path and paths.roaming_dir() == tmp_path

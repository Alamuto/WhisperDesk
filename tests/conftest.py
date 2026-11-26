import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Cada teste usa um diretório de dados próprio (nunca o %LOCALAPPDATA% real)."""
    monkeypatch.setenv("WHISPERDESK_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


def word(s, e, w, p=0.9):
    return {"start_ms": s, "end_ms": e, "word": w, "prob": p}


def seg(words, **kw):
    base = {"start_ms": words[0]["start_ms"], "end_ms": words[-1]["end_ms"],
            "text": "".join(w["word"] for w in words).strip(), "avg_logprob": -0.2,
            "no_speech_prob": 0.01, "compression_ratio": 1.4, "words": words}
    base.update(kw)
    return base


def ffmpeg_available() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:
        return False


def make_media(path, seconds: int = 12, two_tracks: bool = False):
    """Gera mídia de teste com PyAV: senoide 440 Hz (e uma 2ª trilha estéreo 880 Hz, idioma 'eng')."""
    import av
    import numpy as np

    path = str(path)
    aac = path.endswith((".mkv", ".m4a"))
    specs = [(440, 44100, "mono", None)]
    if two_tracks:
        specs.append((880, 48000, "stereo", "eng"))
    with av.open(path, "w") as out:
        streams = []
        for freq, sr, layout, lang in specs:
            s = out.add_stream("aac" if aac else "pcm_s16le", rate=sr, layout=layout)
            if lang:
                s.metadata["language"] = lang
            streams.append((s, freq, sr, layout))
        for s, freq, sr, layout in streams:
            fmt = "fltp" if aac else "s16"
            ch = 1 if layout == "mono" else 2
            n = 1024
            total = int(seconds * sr)
            for start in range(0, total, n):
                t = (np.arange(start, min(total, start + n)) / sr).astype(np.float64)
                x = 0.5 * np.sin(2 * np.pi * freq * t)
                if aac:
                    arr = np.tile(x.astype(np.float32), (ch, 1))
                else:
                    arr = np.repeat((x * 32767).astype(np.int16), ch).reshape(1, -1)
                f = av.AudioFrame.from_ndarray(arr, format=fmt, layout=layout)
                f.sample_rate = sr
                f.pts = start
                for pkt in s.encode(f):
                    out.mux(pkt)
            for pkt in s.encode(None):
                out.mux(pkt)
    from pathlib import Path

    return Path(path)


os.environ.setdefault("PYTHONIOENCODING", "utf-8")

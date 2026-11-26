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
        from app.pipeline import ffmpeg_tools

        ffmpeg_tools.tool("ffmpeg")
        ffmpeg_tools.tool("ffprobe")
        return True
    except Exception:
        return False


os.environ.setdefault("PYTHONIOENCODING", "utf-8")

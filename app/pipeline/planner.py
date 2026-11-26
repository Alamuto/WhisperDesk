"""Planejamento de chunks: cortes no meio de silêncios reais (VAD), nunca no meio de fonemas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf

SR = 16_000


@dataclass(frozen=True)
class ChunkPlan:
    idx: int
    start: int        # amostra inicial de posse (inclusiva)
    end: int          # amostra final de posse (exclusiva)
    pad_left: int     # amostras extras decodificadas antes de start (0 se o corte foi em silêncio)
    pad_right: int
    forced: bool = False


def _read(wav_path: str, start: int, stop: int) -> np.ndarray:
    data, sr = sf.read(wav_path, start=start, stop=stop, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"WAV de trabalho com sample rate inesperado: {sr}")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data


def _gaps(audio: np.ndarray) -> list[tuple[int, int]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = get_speech_timestamps(
        audio, VadOptions(threshold=0.5, min_silence_duration_ms=150, speech_pad_ms=30),
    )
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for s in speech:
        if s["start"] > cursor:
            gaps.append((cursor, s["start"]))
        cursor = max(cursor, s["end"])
    if cursor < len(audio):
        gaps.append((cursor, len(audio)))
    return gaps


def best_cut(audio: np.ndarray, min_gap: int) -> tuple[int, bool]:
    """Retorna (offset_de_corte_na_janela, forçado?)."""
    gaps = _gaps(audio)
    if gaps:
        g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
        if g1 - g0 >= min_gap:
            return (g0 + g1) // 2, False
    # Sem silêncio utilizável: quadro de 20 ms com menor energia.
    frame = SR // 50
    n = len(audio) // frame
    if n == 0:
        return len(audio), True
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    k = int(np.argmin(rms))
    return k * frame + frame // 2, True


def plan_chunks(
    wav_path: str,
    total_samples: int,
    target_s: int = 600,
    search_s: int = 60,
    min_gap_ms: int = 300,
    forced_pad_s: float = 2.0,
    on_progress=None,
) -> list[ChunkPlan]:
    target, search = target_s * SR, search_s * SR
    min_gap = min_gap_ms * SR // 1000
    pad = int(forced_pad_s * SR)

    cuts: list[tuple[int, bool]] = [(0, False)]
    pos = 0
    while total_samples - pos > target + search:
        win_start = pos + target - search
        off, forced = best_cut(_read(wav_path, win_start, pos + target), min_gap)
        cut = win_start + off
        cuts.append((cut, forced))
        pos = cut
        if on_progress:
            on_progress(min(1.0, pos / total_samples))
    cuts.append((total_samples, False))

    plans: list[ChunkPlan] = []
    for i in range(len(cuts) - 1):
        (a, forced_a), (b, forced_b) = cuts[i], cuts[i + 1]
        plans.append(
            ChunkPlan(
                idx=i,
                start=a,
                end=b,
                pad_left=min(pad, a) if forced_a else 0,
                pad_right=min(pad, total_samples - b) if forced_b else 0,
                forced=forced_a or forced_b,
            )
        )
    return plans

"""Re-segmentação por frases a partir dos timestamps de palavras.

O pipeline batched devolve segmentos de até 30 s; para leitura, navegação e TXT, dividimos em frases
(pontuação final ou pausa longa), sem alterar nenhum timestamp de palavra."""
from __future__ import annotations

SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
CLAUSE_END = (",", ";", ":", "—", "，", "、")
PAUSE_SPLIT_MS = 1000
MIN_SENTENCE_MS = 1200
SOFT_MAX_MS = 12_000
HARD_MAX_MS = 20_000


def _make(base: dict, words: list[dict]) -> dict:
    return {
        **{k: v for k, v in base.items() if k not in ("words", "text", "start_ms", "end_ms")},
        "start_ms": words[0]["start_ms"],
        "end_ms": max(words[-1]["end_ms"], words[0]["start_ms"] + 1),
        "text": "".join(w["word"] for w in words).strip(),
        "words": words,
    }


def resegment(segments: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in segments:
        words = s.get("words") or []
        if len(words) < 2:
            out.append(s)
            continue
        cur: list[dict] = [words[0]]
        for w in words[1:]:
            prev = cur[-1]
            dur = prev["end_ms"] - cur[0]["start_ms"]
            gap = w["start_ms"] - prev["end_ms"]
            p = prev["word"].rstrip()
            split = (
                gap >= PAUSE_SPLIT_MS
                or (p.endswith(SENTENCE_END) and dur >= MIN_SENTENCE_MS)
                or (p.endswith(CLAUSE_END) and dur >= SOFT_MAX_MS)
                or dur >= HARD_MAX_MS
            )
            if split:
                out.append(_make(s, cur))
                cur = []
            cur.append(w)
        if cur:
            out.append(_make(s, cur))
    return [x for x in out if x["text"]]

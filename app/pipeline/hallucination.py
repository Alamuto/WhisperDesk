"""Pós-filtro determinístico contra alucinações típicas do Whisper."""
from __future__ import annotations

import re
import unicodedata

# Frases clássicas que o Whisper gera em silêncio/música (vindas de dados de treino de legendas).
# Só são descartadas quando o segmento também tem sinal de não-fala ou baixa confiança.
KNOWN_PHRASES = {
    "legendas pela comunidade amara.org",
    "transcrição e legendas pela comunidade amara.org",
    "obrigado por assistir",
    "obrigada por assistir",
    "inscrevase no canal",
    "inscreva-se no canal",
    "se inscreva no canal",
    "tchau tchau",
    "amém",
    "não se esqueça de se inscrever no canal",
    "subtitles by the amara.org community",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "subtítulos realizados por la comunidad de amara.org",
    "subtitulado por la comunidad de amara.org",
    "gracias por ver el video",
    "sous-titres réalisés par la communauté d'amara.org",
    "sous-titrage st' 501",
    "untertitel im auftrag des zdf für funk 2017",
    "untertitel der amara.org-community",
    "продолжение следует",
    "субтитры сделал dimatorzok",
    "ご視聴ありがとうございました",
}

# Idiomas sem espaço entre palavras: o Whisper emite subpalavras e repetições legítimas são comuns.
_NO_SPACE_LANGS = {"ja", "zh", "yue", "th", "lo", "km", "my", "bo"}


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s.'-]", "", t)
    return re.sub(r"\s+", " ", t).strip(" .-")


_KNOWN = {_norm(p) for p in KNOWN_PHRASES}


def _collapse_word_repeats(words: list[dict], max_repeats: int = 3, max_n: int = 8) -> list[dict]:
    """Opera sobre PALAVRAS (preserva timestamps): 'a b a b a b a b a b' -> 'a b a b a b'."""
    for n in range(1, max_n + 1):
        if len(words) < n * (max_repeats + 1):
            break
        keys = [_norm(w["word"]) for w in words]
        out, i = [], 0
        while i < len(words):
            gram = keys[i:i + n]
            reps = 1
            if len(gram) == n and any(gram):
                while keys[i + reps * n: i + (reps + 1) * n] == gram:
                    reps += 1
            if reps > max_repeats:
                out.extend(words[i: i + n * max_repeats])
                i += reps * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return words


def clean_segments(segments: list[dict], language: str | None = None) -> list[dict]:
    no_space = (language or "") in _NO_SPACE_LANGS
    max_repeats = 6 if no_space else 3
    max_cps = 60 if no_space else 30
    cleaned: list[dict] = []
    run_key, run_len = None, 0
    for s in segments:
        key = _norm(s["text"])
        if not key:
            continue
        dur_ms = s["end_ms"] - s["start_ms"]
        nsp = s.get("no_speech_prob") or 0.0
        lp = s.get("avg_logprob")
        lp = 0.0 if lp is None else lp
        cr = s.get("compression_ratio") or 0.0

        # 1) Frase conhecida de alucinação em região provável de não-fala.
        if key in _KNOWN and (nsp > 0.3 or lp < -0.7):
            continue
        # 2) Alta compressão + baixa confiança = loop de decodificação.
        if cr > 2.4 and lp < -0.8:
            continue
        # 3) Duração degenerada ou taxa de fala fisicamente impossível.
        if dur_ms <= 0 or (len(key) > 8 and len(key) / (dur_ms / 1000) > max_cps):
            continue
        # 4) Mesmo texto repetido em segmentos consecutivos: mantém no máximo 2.
        if key == run_key:
            run_len += 1
            if run_len > 2:
                continue
        else:
            run_key, run_len = key, 1

        words = s.get("words") or []
        kept = _collapse_word_repeats(words, max_repeats=max_repeats) if words else words
        if len(kept) != len(words):
            if not kept:
                continue
            s = {**s, "words": kept,
                 "text": "".join(w["word"] for w in kept).strip(),
                 "start_ms": kept[0]["start_ms"],
                 "end_ms": kept[-1]["end_ms"]}
        cleaned.append(s)
    return cleaned

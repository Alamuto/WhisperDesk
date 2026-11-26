"""Revisão contextual dos trechos de baixa confiança.

Depois da transcrição, os trechos com palavras em vermelho (probabilidade < LOW_WORD) são
retranscritos isoladamente, com folga de áudio dos dois lados e com o contexto da conversa:

- `initial_prompt`: as frases anteriores (sem as palavras duvidosas, para o erro não se propagar);
- `hotwords`: termos que a própria conversa já fixou com confiança (nomes próprios, termos que se
  repetem), o "vocabulário" da gravação.

Cada candidato é comparado com o original pela confiança das palavras na mesma janela. Só vira
sugestão o que muda o texto e é claramente mais provável; nada é trocado sem o usuário aceitar.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from typing import Callable

import numpy as np

SR = 16_000
LOW_WORD = 0.50            # o mesmo limiar do vermelho na interface
SUSPECT_SEG_LOGPROB = -1.0
PAD_MS = 1500              # áudio extra de cada lado (frase completa para o modelo)
OWN_SLACK_MS = 200         # tolerância ao decidir quais palavras do candidato pertencem ao trecho
PROMPT_CHARS = 240
MAX_HOTWORDS = 24
MAX_SUSPECTS = 600         # teto por job (gravações muito ruidosas)
MIN_GAIN = 0.10            # ganho mínimo de confiança média para sugerir
MIN_NEW_MEAN = 0.60


def _norm_tok(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).casefold()
    return re.sub(r"[^\w'-]", "", t)


def norm_text(text: str) -> str:
    return " ".join(x for x in (_norm_tok(t) for t in text.split()) if x)


def is_suspect(seg: dict) -> bool:
    if seg.get("edited"):
        return False  # o que o usuário corrigiu à mão é definitivo
    words = seg.get("words") or []
    if not words:
        return False
    if any(w[3] < LOW_WORD for w in words):
        return True
    lp = seg.get("avg_logprob")
    return lp is not None and lp < SUSPECT_SEG_LOGPROB


def find_suspects(segs: list[dict]) -> list[int]:
    return [i for i, s in enumerate(segs) if is_suspect(s)][:MAX_SUSPECTS]


def context_prompt(segs: list[dict], i: int, limit: int = PROMPT_CHARS) -> str | None:
    """Frases anteriores ao trecho i, só com as palavras confiáveis, até `limit` caracteres."""
    parts: list[str] = []
    size = 0
    for s in reversed(segs[max(0, i - 8):i]):
        words = s.get("words") or []
        if words:
            text = "".join(w[2] for w in words if w[3] >= LOW_WORD).strip()
        else:
            text = (s.get("text") or "").strip()
        if not text:
            continue
        parts.append(text)
        size += len(text) + 1
        if size >= limit:
            break
    if not parts:
        return None
    prompt = " ".join(reversed(parts))
    return prompt[-limit:].lstrip()


def glossary(segs: list[dict], limit: int = MAX_HOTWORDS) -> str | None:
    """Vocabulário que a conversa já fixou com confiança: nomes próprios e termos longos recorrentes."""
    counts: Counter[str] = Counter()
    proper: Counter[str] = Counter()
    for s in segs:
        words = s.get("words") or []
        for k, w in enumerate(words):
            if w[3] < 0.85:
                continue
            raw = w[2].strip().strip(".,;:!?…\"'()[]«»“”")
            if len(raw) < 3 or not any(ch.isalpha() for ch in raw):
                continue
            prev = words[k - 1][2].strip() if k else ""
            sentence_start = k == 0 or prev.endswith((".", "!", "?", "…"))
            if raw[0].isupper() and not sentence_start and not raw.isupper():
                proper[raw] += 1
            elif len(raw) >= 7:
                counts[raw.lower()] += 1
    terms = [t for t, _ in proper.most_common()]
    terms += [t for t, n in counts.most_common() if n >= 2 and t not in {x.lower() for x in terms}]
    terms = terms[:limit]
    return " ".join(terms) if terms else None


def mean_prob(words) -> float:
    return float(np.mean([w[3] for w in words])) if words else 0.0


def low_count(words) -> int:
    return sum(1 for w in words if w[3] < LOW_WORD)


def owned_words(raw_segments, window_start_ms: int, seg_start_ms: int, seg_end_ms: int) -> list[tuple]:
    """Palavras do candidato cujo ponto médio cai dentro do trecho original (tempo global)."""
    out = []
    for s in raw_segments:
        for w in (s.words or []):
            a = window_start_ms + int(round(w.start * 1000))
            b = max(a, window_start_ms + int(round(w.end * 1000)))
            mid = (a + b) // 2
            if seg_start_ms - OWN_SLACK_MS <= mid <= seg_end_ms + OWN_SLACK_MS:
                out.append((a, b, w.word, float(w.probability)))
    return out


def choose(original: list[tuple], candidates: list[list[tuple]], prompt: str | None) -> list[tuple] | None:
    """Melhor candidato que muda o texto e é claramente mais confiável que o original, ou None."""
    orig_text = norm_text("".join(w[2] for w in original))
    orig_mean = mean_prob(original)
    orig_low = low_count(original)
    n_orig = max(1, len(original))
    norm_prompt = norm_text(prompt or "")
    best, best_mean = None, -1.0
    for cand in candidates:
        if not cand:
            continue
        text = norm_text("".join(w[2] for w in cand))
        if not text or text == orig_text:
            continue  # confirmou o original: nada a sugerir
        ratio = len(cand) / n_orig
        if not 0.5 <= ratio <= 2.0:
            continue
        # O contexto vira eco: o modelo repete a frase anterior em vez de ouvir o áudio.
        if len(text.split()) >= 3 and norm_prompt and text in norm_prompt:
            continue
        toks = text.split()
        if len(toks) >= 6 and len(set(toks)) <= len(toks) // 3:
            continue  # repetição degenerada
        m = mean_prob(cand)
        if m < MIN_NEW_MEAN or m < orig_mean + MIN_GAIN or low_count(cand) > orig_low:
            continue
        if m > best_mean:
            best, best_mean = cand, m
    return best


def changed_tokens(original_text: str, suggestion_text: str) -> list[int]:
    """Índices (nas palavras da sugestão) do que difere do original — para destacar na interface."""
    a = [_norm_tok(t) for t in original_text.split()]
    b = [_norm_tok(t) for t in suggestion_text.split()]
    out: list[int] = []
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag in ("replace", "insert"):
            out.extend(range(j1, j2))
    return out


def refine_job(j, job_id: str, wav_path: str, decode: Callable, language: str | None,
               on_progress: Callable[[float], None], cancel_event) -> int:
    """Gera as sugestões de um job. `decode(audio, prompt, hotwords) -> raw_segments`.

    Retorna quantas sugestões foram gravadas. Trechos já resolvidos (aceitos/recusados) e
    editados à mão não são revisitados."""
    import soundfile as sf

    from app.pipeline.ffmpeg_tools import Cancelled

    segs = j.segments_with_words(job_id)
    resolved = j.resolved_suggestion_keys(job_id)
    suspects = [i for i in find_suspects(segs) if segs[i]["key"] not in resolved]
    j.clear_pending_suggestions(job_id)
    hot = glossary(segs)
    info = sf.info(wav_path)
    total_ms = int(info.frames * 1000 // info.samplerate)
    found: list[tuple] = []
    for n, i in enumerate(suspects):
        if cancel_event is not None and cancel_event.is_set():
            raise Cancelled()
        s = segs[i]
        a_ms = max(0, s["start_ms"] - PAD_MS)
        b_ms = min(total_ms, s["end_ms"] + PAD_MS)
        audio, _ = sf.read(wav_path, start=a_ms * SR // 1000, stop=b_ms * SR // 1000, dtype="float32",
                           always_2d=False)
        if audio.size < SR // 4:
            continue
        prompt = context_prompt(segs, i)
        candidates = []
        tried = set()
        orig_norm = norm_text("".join(w[2] for w in s["words"]))
        for p, h in ((prompt, hot), (prompt, None), (None, hot)):
            if (p, h) in tried or (p is None and h is None):
                continue
            tried.add((p, h))
            raw = decode(audio, p, h)
            cand = owned_words(raw, a_ms, s["start_ms"], s["end_ms"])
            candidates.append(cand)
            # Com o contexto completo, o modelo confirmou o original: não vale gastar mais decodificações.
            if len(candidates) == 1 and norm_text("".join(w[2] for w in cand)) == orig_norm:
                break
            if choose(list(s["words"]), [cand], prompt) is not None and mean_prob(cand) >= 0.85:
                break  # já há uma alternativa clara
        best = choose(list(s["words"]), candidates, prompt)
        if best is not None:
            text = "".join(w[2] for w in best).strip()
            found.append((s["key"], text, best, mean_prob(s["words"]), mean_prob(best)))
        on_progress((n + 1) / max(1, len(suspects)))
    j.add_suggestions(job_id, found)
    j.update_fields(job_id, refined=1)
    return len(found)

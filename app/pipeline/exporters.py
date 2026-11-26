"""Exportação SRT/VTT/TXT/JSON/CSV com regras ergonômicas de legendagem e escrita atômica em streaming."""
from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

MAX_LINE = 42
MAX_LINES = 2
MAX_DUR_MS = 7000
MIN_DUR_MS = 833
MAX_CPS = 20.0
MIN_GAP_MS = 80
PAUSE_SPLIT_MS = 700
SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
CLAUSE_END = (",", ";", ":", "—", "，", "、")


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    lines: list[str]
    speaker: str | None = None


def fmt_ts(ms: int, sep: str = ".") -> str:
    ms = max(0, int(ms))
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


_fmt = fmt_ts


def fmt_short(ms: int) -> str:
    """HH:MM:SS (ou MM:SS abaixo de 1 h) para exibição."""
    s = max(0, int(ms)) // 1000
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _wrap(text: str) -> list[str] | None:
    """Quebra em até 2 linhas de ≤ MAX_LINE sem partir palavras; None se impossível."""
    if len(text) <= MAX_LINE:
        return [text]
    words = text.split()
    best, best_score = None, None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) > MAX_LINE or len(b) > MAX_LINE:
            continue
        score = abs(len(a) - len(b))
        if a.endswith(CLAUSE_END + SENTENCE_END):
            score -= 15
        if len(words[i - 1]) <= 2:            # não deixa "de", "a", "o" pendurado no fim da linha
            score += 20
        score += 0 if len(a) <= len(b) else 3  # leve preferência por pirâmide invertida
        if best_score is None or score < best_score:
            best, best_score = [a, b], score
    return best


def _hard_wrap(text: str) -> list[str]:
    """Último recurso (uma única "palavra" > 84 caracteres, ex.: URL): quebra por largura."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        while len(w) > MAX_LINE:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(w[:MAX_LINE])
            w = w[MAX_LINE:]
        if cur and len(cur) + 1 + len(w) > MAX_LINE:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [text]


def build_cues(words: Iterable[dict]) -> list[Cue]:
    """words: dicts com start_ms, end_ms, word (com espaço inicial como no Whisper), speaker opcional."""
    cues: list[Cue] = []
    cur: list[dict] = []

    def text_of(ws):
        return "".join(w["word"] for w in ws).strip()

    def flush():
        if cur:
            t = text_of(cur)
            cues.append(Cue(cur[0]["start_ms"], cur[-1]["end_ms"], _wrap(t) or _hard_wrap(t), cur[0].get("speaker")))
            cur.clear()

    for w in words:
        if not w["word"].strip():
            continue
        if cur:
            gap = w["start_ms"] - cur[-1]["end_ms"]
            cand = cur + [w]
            cand_text = text_of(cand)
            dur = cand[-1]["end_ms"] - cand[0]["start_ms"]
            too_long = dur > MAX_DUR_MS or _wrap(cand_text) is None
            too_fast = dur > 0 and len(cand_text) / (dur / 1000) > MAX_CPS * 1.5 and len(cand_text) > MAX_LINE
            speaker_change = w.get("speaker") != cur[-1].get("speaker")
            sentence_break = cur[-1]["word"].rstrip().endswith(SENTENCE_END) and len(text_of(cur)) >= MAX_LINE * 0.6
            if gap >= PAUSE_SPLIT_MS or too_long or too_fast or speaker_change or sentence_break:
                flush()
        cur.append(w)
    flush()

    # Ajuste temporal: duração mínima, CPS e intervalo sem sobreposição.
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start_ms if i + 1 < len(cues) else None
        chars = sum(len(line) for line in c.lines)
        wanted = max(MIN_DUR_MS, int(chars / MAX_CPS * 1000))
        end = min(max(c.end_ms, c.start_ms + wanted), c.start_ms + MAX_DUR_MS)
        end = max(end, c.end_ms)                 # nunca termina antes da última palavra...
        if nxt is not None:
            end = min(end, nxt - MIN_GAP_MS)     # ...exceto para não sobrepor a próxima
        c.end_ms = max(end, c.start_ms + 1)
    return cues


def _vtt_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("-->", "→")


def write_srt(f, cues: Iterable[Cue]) -> None:
    for i, c in enumerate(cues, 1):
        lines = list(c.lines)
        if c.speaker:
            lines[0] = f"[{c.speaker}] {lines[0]}"
        f.write(f"{i}\n{fmt_ts(c.start_ms, ',')} --> {fmt_ts(c.end_ms, ',')}\n" + "\n".join(lines) + "\n\n")


def write_vtt(f, cues: Iterable[Cue]) -> None:
    f.write("WEBVTT\n\n")
    for c in cues:
        body = "\n".join(_vtt_escape(line) for line in c.lines)
        if c.speaker:
            body = f"<v {_vtt_escape(c.speaker)}>{body}"
        f.write(f"{fmt_ts(c.start_ms, '.')} --> {fmt_ts(c.end_ms, '.')}\n{body}\n\n")


def to_srt(cues: list[Cue]) -> str:
    import io
    b = io.StringIO()
    write_srt(b, cues)
    return b.getvalue()


def to_vtt(cues: list[Cue]) -> str:
    import io
    b = io.StringIO()
    write_vtt(b, cues)
    return b.getvalue()


def write_txt(f, segments: Iterable[dict], timestamps: bool = False) -> None:
    for s in segments:
        prefix = f"[{fmt_ts(s['start_ms'], '.')[:-4]}] " if timestamps else ""
        speaker = f"{s['speaker']}: " if s.get("speaker") else ""
        f.write(f"{prefix}{speaker}{s['text']}\n")


def to_txt(segments: Iterable[dict], timestamps: bool = False) -> str:
    import io
    b = io.StringIO()
    write_txt(b, segments, timestamps)
    return b.getvalue()


def write_csv(f, segments: Iterable[dict]) -> None:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["start", "end", "start_ms", "end_ms", "speaker", "text", "avg_logprob"])
    for s in segments:
        w.writerow([fmt_ts(s["start_ms"]), fmt_ts(s["end_ms"]), s["start_ms"], s["end_ms"],
                    s.get("speaker") or "", s["text"],
                    "" if s.get("avg_logprob") is None else round(s["avg_logprob"], 4)])


def write_json(f, metadata: dict, segments: Iterable[dict], words_of: Callable[[dict], Iterable]) -> None:
    """Streaming: não monta o documento inteiro em RAM (12 h ≈ 100k palavras)."""
    f.write('{"schema_version": 1, "metadata": ')
    json.dump(metadata, f, ensure_ascii=False)
    f.write(', "segments": [\n')
    first = True
    for s in segments:
        if not first:
            f.write(",\n")
        first = False
        item = {k: s[k] for k in ("start_ms", "end_ms", "text", "speaker", "avg_logprob", "no_speech_prob")
                if k in s and s[k] is not None}
        item["words"] = [{"w": w[2], "s": w[0], "e": w[1], "p": round(float(w[3]), 4)} for w in words_of(s)]
        json.dump(item, f, ensure_ascii=False)
    f.write("\n]}\n")


def atomic_write(path: Path, writer, encoding: str = "utf-8") -> None:
    """Escreve num temporário no MESMO diretório, fsync, e troca atomicamente."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as f:
            if callable(writer):
                writer(f)
            else:
                f.write(writer)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Formatos públicos: chave -> (rótulo, extensão)
EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "srt": ("SubRip (.srt)", ".srt"),
    "vtt": ("WebVTT (.vtt)", ".vtt"),
    "txt": ("Texto (.txt)", ".txt"),
    "txt_ts": ("Texto com tempos (.txt)", ".txt"),
    "json": ("JSON com palavras (.json)", ".json"),
    "csv": ("CSV (.csv)", ".csv"),
}


def _words_flat(segments: list[dict]) -> Iterable[dict]:
    for s in segments:
        sp = s.get("speaker")
        words = s.get("words") or []
        if not words:   # segmento sem palavras (importado): uma pseudo-palavra por segmento
            yield {"start_ms": s["start_ms"], "end_ms": s["end_ms"], "word": " " + s["text"], "speaker": sp}
            continue
        for w in words:
            yield {"start_ms": w[0], "end_ms": w[1], "word": w[2], "speaker": sp}


def export_segments(segments: list[dict], fmt: str, path: Path | str, metadata: dict | None = None) -> None:
    """segments: [{start_ms, end_ms, text, avg_logprob, speaker, words: [(s, e, w, p), ...]}] em ordem temporal."""
    path = Path(path)
    if fmt == "srt":
        cues = build_cues(_words_flat(segments))
        atomic_write(path, lambda f: write_srt(f, cues))
    elif fmt == "vtt":
        cues = build_cues(_words_flat(segments))
        atomic_write(path, lambda f: write_vtt(f, cues))
    elif fmt == "txt":
        atomic_write(path, lambda f: write_txt(f, segments, False))
    elif fmt == "txt_ts":
        atomic_write(path, lambda f: write_txt(f, segments, True))
    elif fmt == "csv":
        atomic_write(path, lambda f: write_csv(f, segments), encoding="utf-8-sig")  # Excel reconhece UTF-8 com BOM
    elif fmt == "json":
        atomic_write(path, lambda f: write_json(f, metadata or {}, segments, lambda s: s.get("words") or []))
    else:
        raise ValueError(f"Formato desconhecido: {fmt}")

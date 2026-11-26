"""Diarização ("quem falou o quê") 100% local, sem dependências novas.

Pipeline:
1. Para cada segmento transcrito, extrai fbank Kaldi (80 bandas) do trecho de áudio.
2. Calcula o embedding de voz com o modelo WeSpeaker ResNet34 (ONNX, Apache-2.0) via onnxruntime,
   que o faster-whisper já usa para o VAD.
3. Agrupa os embeddings por similaridade de cosseno (aglomerativo, ligação média), com número de
   falantes automático ou fixo.
4. Trechos curtos demais para um embedding confiável herdam o falante do vizinho mais próximo.

O modelo (~26 MB) é baixado do Hugging Face na primeira vez, como os modelos do Whisper.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

SR = 16000
SPK_REPO = "Wespeaker/wespeaker-voxceleb-resnet34"
SPK_FILE = "voxceleb_resnet34.onnx"
SPK_DIR = "speaker-resnet34"
SPK_DOWNLOAD_MB = 26

MIN_EMBED_S = 0.9          # abaixo disso o embedding é ruidoso: herda o falante vizinho
MAX_EMBED_S = 12.0         # trechos longos: usa só o centro (custo limitado, voz estável)
AUTO_THRESHOLD = 0.62      # distância de cosseno para unir grupos no modo automático
MAX_CLUSTER_ITEMS = 1000  # acima disso agrupa uma amostra e atribui o resto ao centróide

# ----------------------------------------------------------------------------- fbank (Kaldi)
_N_FFT = 512
_WIN = 400      # 25 ms
_HOP = 160      # 10 ms
_N_MELS = 80
_mel_cache: dict = {}


def _mel_banks() -> np.ndarray:
    if "m" in _mel_cache:
        return _mel_cache["m"]

    def mel(f):
        return 1127.0 * np.log(1.0 + np.asarray(f, dtype=np.float64) / 700.0)

    low, high = 20.0, SR / 2
    mlow, mhigh = mel(low), mel(high)
    delta = (mhigh - mlow) / (_N_MELS + 1)
    n_bins = _N_FFT // 2
    freqs = np.arange(n_bins) * SR / _N_FFT
    mf = mel(freqs)
    banks = np.zeros((_N_MELS, n_bins + 1), dtype=np.float32)
    for b in range(_N_MELS):
        left, center, right = mlow + b * delta, mlow + (b + 1) * delta, mlow + (b + 2) * delta
        up = (mf - left) / (center - left)
        down = (right - mf) / (right - center)
        banks[b, :n_bins] = np.maximum(0.0, np.minimum(up, down))
    _mel_cache["m"] = banks
    return banks


_POVEY = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WIN) / (_WIN - 1))) ** 0.85


def fbank(audio: np.ndarray) -> np.ndarray:
    """Fbank log-mel compatível com torchaudio.compliance.kaldi.fbank (dither=0, janela povey),
    com normalização de média (CMN) — a entrada que os modelos WeSpeaker esperam."""
    x = np.asarray(audio, dtype=np.float32) * 32768.0
    if len(x) < _WIN:
        x = np.pad(x, (0, _WIN - len(x)))
    n = 1 + (len(x) - _WIN) // _HOP
    idx = np.arange(_WIN)[None, :] + _HOP * np.arange(n)[:, None]
    frames = x[idx].astype(np.float64)
    frames -= frames.mean(axis=1, keepdims=True)
    pre = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = frames - 0.97 * pre
    frames *= _POVEY
    spec = np.abs(np.fft.rfft(frames, n=_N_FFT)) ** 2
    feats = np.log(np.maximum(spec @ _mel_banks().T.astype(np.float64), np.finfo(np.float32).eps))
    feats -= feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


# ----------------------------------------------------------------------------- modelo
def model_path(models_root: Path) -> Path:
    return Path(models_root) / SPK_DIR / SPK_FILE


def is_model_ready(models_root: Path) -> bool:
    p = model_path(models_root)
    return p.is_file() and (p.parent / ".complete").is_file()


def ensure_model(models_root: Path, on_progress: Callable[[int, int], None] | None = None,
                 cancel_event=None) -> Path:
    if is_model_ready(models_root):
        return model_path(models_root)
    import threading

    from huggingface_hub import hf_hub_download

    target = Path(models_root) / SPK_DIR
    target.mkdir(parents=True, exist_ok=True)
    expected = SPK_DOWNLOAD_MB << 20
    err: list[BaseException] = []

    def _dl():
        try:
            hf_hub_download(SPK_REPO, SPK_FILE, local_dir=str(target))
        except BaseException as e:  # noqa: BLE001 — repassado ao chamador
            err.append(e)

    t = threading.Thread(target=_dl, name="speaker-model-download", daemon=True)
    t.start()
    while t.is_alive():
        t.join(0.4)
        if on_progress:
            size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            on_progress(min(size, expected), expected)
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("download cancelado")
    if err:
        raise RuntimeError(f"Falha ao baixar o modelo de falantes. Verifique a conexão com huggingface.co. "
                           f"Detalhe: {type(err[0]).__name__}: {err[0]}") from err[0]
    (target / ".complete").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return model_path(models_root)


class SpeakerEmbedder:
    def __init__(self, onnx_path: str | Path):
        import os

        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 2)
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, audio: np.ndarray) -> np.ndarray:
        feats = fbank(audio)[None, :, :]
        e = self.sess.run(None, {self.inp: feats})[0][0].astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


# ----------------------------------------------------------------------------- agrupamento
def cluster(emb: np.ndarray, num_speakers: int | None = None, threshold: float = AUTO_THRESHOLD) -> np.ndarray:
    """Aglomerativo com ligação média sobre distância de cosseno. Retorna rótulos 0..k-1,
    ordenados pela primeira aparição."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    sim = emb @ emb.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, np.inf)
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    size = np.ones(n)
    active = np.ones(n, dtype=bool)
    target = max(1, int(num_speakers)) if num_speakers else 1
    while active.sum() > target:
        sub = np.where(active)[0]
        d = dist[np.ix_(sub, sub)]
        k = int(np.argmin(d))
        a, b = sub[k // len(sub)], sub[k % len(sub)]
        dmin = dist[a, b]
        if not num_speakers and dmin > threshold:
            break
        # Ligação média: d(a∪b, c) = (|a| d(a,c) + |b| d(b,c)) / (|a|+|b|)
        na, nb = size[a], size[b]
        newd = (na * dist[a] + nb * dist[b]) / (na + nb)
        dist[a, :] = newd
        dist[:, a] = newd
        dist[a, a] = np.inf
        dist[b, :] = np.inf
        dist[:, b] = np.inf
        size[a] = na + nb
        active[b] = False
        members[a].extend(members.pop(b))
    labels = np.empty(n, dtype=int)
    for lab, idxs in enumerate(members.values()):
        labels[idxs] = lab
    # renumera pela primeira aparição
    order: dict[int, int] = {}
    for lab in labels:
        order.setdefault(int(lab), len(order))
    return np.array([order[int(x)] for x in labels], dtype=int)


def _weighted_cost(emb: np.ndarray, labels: np.ndarray, w: np.ndarray) -> float:
    cost = 0.0
    for c in set(labels.tolist()):
        m = labels == c
        cen = (emb[m] * w[m, None]).sum(axis=0)
        cen /= np.linalg.norm(cen) + 1e-9
        cost += float((w[m] * (1.0 - emb[m] @ cen)).sum())
    return cost


def cluster_fixed(emb: np.ndarray, k: int, weights: np.ndarray, restarts: int = 12, seed: int = 7) -> np.ndarray:
    """Número fixo de falantes: k-means esférico ponderado pela duração, comparado com o resultado
    aglomerativo. Fica a solução de menor custo, o que impede que trechos curtos e atípicos
    (risadas, alucinações) ocupem um grupo inteiro no lugar de uma voz real."""
    n = len(emb)
    k = max(1, min(int(k), n))
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum() or 1.0)
    cands = [cluster(emb, k)]
    rng = np.random.default_rng(seed)
    for _ in range(restarts):
        # k-means++ ponderado
        cents = [emb[rng.choice(n, p=w)]]
        for _ in range(1, k):
            d = np.min(1.0 - emb @ np.stack(cents).T, axis=1).clip(0) * w
            cents.append(emb[rng.choice(n, p=d / d.sum())] if d.sum() > 0 else emb[rng.integers(n)])
        C = np.stack(cents)
        lab = np.argmax(emb @ C.T, axis=1)
        for _ in range(30):
            C2 = []
            for c in range(k):
                m = lab == c
                v = (emb[m] * w[m, None]).sum(axis=0) if m.any() else C[c]
                C2.append(v / (np.linalg.norm(v) + 1e-9))
            C = np.stack(C2)
            new = np.argmax(emb @ C.T, axis=1)
            if np.array_equal(new, lab):
                break
            lab = new
        cands.append(lab)
    best = min(cands, key=lambda l: _weighted_cost(emb, l, w))
    order: dict[int, int] = {}
    for lab in best:
        order.setdefault(int(lab), len(order))
    return np.array([order[int(x)] for x in best], dtype=int)


def _merge_tiny_clusters(emb: np.ndarray, labels: np.ndarray, durations: np.ndarray, min_share: float = 0.04):
    """Grupos com fração ínfima da fala (ruído, risadas) são absorvidos pelo grupo mais próximo."""
    total = durations.sum() or 1.0
    labs = sorted(set(labels.tolist()))
    if len(labs) <= 1:
        return labels
    cents = {l: emb[labels == l].mean(axis=0) for l in labs}
    share = {l: durations[labels == l].sum() / total for l in labs}
    big = [l for l in labs if share[l] >= min_share]
    if not big:
        return labels
    out = labels.copy()
    for l in labs:
        if l in big:
            continue
        c = cents[l] / (np.linalg.norm(cents[l]) + 1e-9)
        best = max(big, key=lambda b: float(c @ (cents[b] / (np.linalg.norm(cents[b]) + 1e-9))))
        out[labels == l] = best
    order: dict[int, int] = {}
    for lab in out:
        order.setdefault(int(lab), len(order))
    return np.array([order[int(x)] for x in out], dtype=int)


def diarize_segments(wav_path: str, segments: list[dict], embedder: SpeakerEmbedder,
                     num_speakers: int | None = None, on_progress: Callable[[float], None] | None = None,
                     cancel_event=None) -> list[int]:
    """Retorna o índice do falante (0..k-1) de cada segmento (mesma ordem de `segments`)."""
    import soundfile as sf

    n = len(segments)
    if n == 0:
        return []
    total = sf.info(wav_path).frames
    usable: list[int] = []
    embs: list[np.ndarray] = []
    durs: list[float] = []
    last = 0.0
    with sf.SoundFile(wav_path) as f:
        for i, s in enumerate(segments):
            if cancel_event is not None and cancel_event.is_set():
                from app.pipeline.ffmpeg_tools import Cancelled

                raise Cancelled()
            a = int(s["start_ms"]) * SR // 1000
            b = min(total, int(s["end_ms"]) * SR // 1000)
            dur = (b - a) / SR
            if dur >= MIN_EMBED_S:
                if dur > MAX_EMBED_S:
                    mid = (a + b) // 2
                    half = int(MAX_EMBED_S * SR / 2)
                    a, b = mid - half, mid + half
                f.seek(max(0, a))
                audio = f.read(max(0, b - a), dtype="float32", always_2d=False)
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if len(audio) >= int(MIN_EMBED_S * SR) and float(np.abs(audio).max(initial=0)) > 1e-4:
                    embs.append(embedder.embed(audio))
                    usable.append(i)
                    durs.append(dur)
            now = time.monotonic()
            if on_progress and (now - last > 0.3 or i == n - 1):
                last = now
                on_progress((i + 1) / n * 0.9)

    labels_all = [-1] * n
    if not usable:
        return [0] * n
    E = np.stack(embs)
    D = np.array(durs)
    if len(usable) > MAX_CLUSTER_ITEMS:
        pick = np.argsort(-D)[:MAX_CLUSTER_ITEMS]
        if num_speakers:
            lab_s = cluster_fixed(E[pick], num_speakers, D[pick])
        else:
            lab_s = _merge_tiny_clusters(E[pick], cluster(E[pick]), D[pick])
        k = lab_s.max() + 1
        cents = np.stack([E[pick][lab_s == c].mean(axis=0) for c in range(k)])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
        labels = np.argmax(E @ cents.T, axis=1)
    else:
        if num_speakers:
            labels = cluster_fixed(E, num_speakers, D)
        else:
            labels = _merge_tiny_clusters(E, cluster(E), D)
    for i, lab in zip(usable, labels):
        labels_all[i] = int(lab)

    # Trechos sem embedding: falante do vizinho mais próximo no tempo
    known = [i for i in range(n) if labels_all[i] >= 0]
    for i in range(n):
        if labels_all[i] >= 0:
            continue
        mid = (segments[i]["start_ms"] + segments[i]["end_ms"]) / 2
        j = min(known, key=lambda k: abs((segments[k]["start_ms"] + segments[k]["end_ms"]) / 2 - mid))
        labels_all[i] = labels_all[j]
    # renumera pela primeira aparição na linha do tempo
    order: dict[int, int] = {}
    for idx in sorted(range(n), key=lambda k: segments[k]["start_ms"]):
        order.setdefault(labels_all[idx], len(order))
    if on_progress:
        on_progress(1.0)
    return [order[l] for l in labels_all]

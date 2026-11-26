# WhisperDesk — Auditoria de Produção e Arquitetura-Alvo

Escopo: código em `main.py`, `app/**`, `WhisperDesk.spec`, `build.bat`, `install.bat`, e o bundle em `dist/WhisperDesk` (4,9 GB, dos quais 4,0 GB são `torch`).
Alvo: vídeos de 2–12+ h, arquivos de dezenas de GB, retomada após crash, Windows 10/11 x64.

**Veredito:** a aplicação atual é um *wrapper* de uma única chamada `model.transcribe()` sobre o arquivo inteiro. Não há chunking, não há journaling, o worker roda em `QThread` dentro do processo da UI, e todo o estado vive em RAM até o fim. Para um vídeo de 10 h, qualquer falha aos 98% perde 100% do trabalho. Isso é um problema de arquitetura, não um bug isolado — a Fase 3 lista os defeitos pontuais, mas a correção real é a arquitetura da Fase 1.

> A Fase 3 vem primeiro neste documento porque é a base de evidência de tudo o que vem depois.

---

## FASE 3 — Auditoria forense (o que está quebrado hoje)

### 3.1 Bloqueadores para vídeos longos (P0)

| # | Local | Defeito | Consequência concreta |
|---|---|---|---|
| B1 | `app/core/transcriber.py:290` | `model.transcribe(wav_path)` sobre o arquivo inteiro. O faster-whisper chama `decode_audio()` e carrega **todo** o áudio como `float32`. | 12 h × 16 000 × 4 B = **2,76 GB de RAM** só para o array, com pico de ~4 GB durante a conversão int16→float32. Combinado com UI + QtWebEngine, máquinas de 8 GB entram em swap ou dão `MemoryError`. |
| B2 | `transcriber.py` (sem journal) | Os segmentos ficam apenas em `segments_list` (RAM). | Crash, OOM, queda de energia, Windows Update → **todo o progresso perdido**. |
| B3 | `app/gui/main_window.py:230-243` | O "draft" é um `json.dump` do resultado completo **na thread da UI** a cada 30 s, escrito sem atomicidade. | 12 h ≈ 15 mil segmentos + ~100 mil palavras = dezenas de MB serializados a cada 30 s → a UI congela. Crash durante a escrita = JSON truncado → `_restore_startup_state` cai no `except` e **apaga o draft** (`_clear_draft()`). Mesmo quando funciona, o draft só *exibe* o texto: **não retoma** a transcrição. |
| B4 | `transcriber.py:237-265` | O fallback de OOM só envolve o **carregamento** do modelo. | OOM de CUDA no meio da inferência (comum com `beam_size=5` + `word_timestamps`) cai em `except Exception` → "Unexpected error" → tudo perdido. |
| B5 | `app/utils/waveform.py:134` | Fallback do `pydub`: `AudioSegment.from_file()` decodifica o arquivo **inteiro** antes do corte de 30 min (`:139`). O `soundfile` não lê MP4/MKV, então todo vídeo cai nesse caminho. | MKV de 12 h, estéreo, 48 kHz, s16 = **8,3 GB de RAM** só para desenhar a forma de onda ao soltar o arquivo. Acontece **antes** de qualquer transcrição. |
| B6 | `app/core/audio_processor.py:242` | `timeout=3600` na extração do ffmpeg. | Um MKV de 50 GB em HDD USB ou em compartilhamento de rede pode passar de 1 h só de demux → aborta. |
| B7 | `main_window.py:703` | `closeEvent` faz `cancel()` e depois `wait(5000)`. O cancelamento só é verificado **entre** segmentos; dentro de `generate()` do CTranslate2 não há verificação. | Fechar a janela durante a inferência destrói um `QThread` ainda rodando → `QThread: Destroyed while thread is still running` / crash nativo, e o WAV temporário fica órfão em `%TEMP%`. |
| B8 | `transcriber.py:147` | Na troca de modelo, o novo `WhisperModel` é criado **antes** de liberar o antigo (`self._model = WhisperModel(...)`). | Por um momento os dois modelos ficam na VRAM → OOM artificial ao trocar de large-v2 para large-v3 numa GPU de 8 GB. |

### 3.2 Pontos únicos de falha na ingestão (SPOF)

| # | Local | Defeito |
|---|---|---|
| S1 | `audio_processor.py:62-87` | O ffmpeg/ffprobe são resolvidos por `where` no `PATH`. O build congelado **não inclui o ffmpeg** → em máquina limpa o `.exe` não funciona. Além disso, `subprocess` sem `CREATE_NO_WINDOW` num app `--windowed` faz **piscar uma janela de console** a cada chamada. |
| S2 | `audio_processor.py:232-245` | Não há `-map`: o ffmpeg escolhe a trilha "melhor" sozinho. Com várias trilhas (dublado/original/comentário), o usuário não controla qual é transcrita. |
| S3 | idem | `-ac 1` em 5.1/7.1 faz um downmix que inclui LFE e surrounds; o canal central (diálogo) fica diluído em trilha sonora/efeitos → WER pior e mais alucinação. |
| S4 | idem | Não há `aresample=async`. Arquivos com buracos de PTS (gravações de OBS, `.ts` de broadcast, streams cortados) geram um WAV *contíguo* e o buraco é colapsado → **drift cumulativo** entre legenda e vídeo após o ponto do buraco. Também não há compensação de `start_time` ≠ 0 (edit lists de MP4, `.ts`). |
| S5 | idem | `text=True` sem `encoding` → decodificação de stderr pela code page local (cp1252). Nomes de arquivo/metadados UTF-8 no stderr → `UnicodeDecodeError` mascarado como "Failed to run ffmpeg". `capture_output=True` acumula todo o stderr em RAM (ffmpeg com áudio corrompido pode emitir milhões de linhas de warning). |
| S6 | idem | Sem `-rf64 auto`: o WAV estoura 4 GB em ~37 h de áudio 16 kHz mono. Fora da meta de 12 h, mas está a um passo de distância. |
| S7 | `audio_processor.py:111` (`probe`) | Não verifica se existe **alguma** trilha de áudio. Vídeo sem áudio só falha depois, na conversão, com stderr cru do ffmpeg. |
| S8 | `audio_processor.py:224` | WAV temporário em `%TEMP%` via `NamedTemporaryFile(delete=False)`. Em crash nunca é removido (≈1,4 GB por vídeo de 12 h). Não há verificação de espaço livre em disco. |
| S9 | `gpu_monitor.py:79` | "CUDA disponível" = "NVML enxerga uma GPU". Isso não prova que o **CTranslate2** consegue carregar cuBLAS/cuDNN. O fallback `ctypes.CDLL("cudart.dll")` usa um nome que não existe (é `cudart64_12.dll`). Só a GPU de índice 0 é considerada. |
| S10 | `transcriber.py:23-30` | A tabela `MODEL_VRAM_MB` usa números do Whisper da OpenAI/PyTorch (large-v3 = 10 GB). No CTranslate2 o large-v3 em fp16 ocupa ~3–4,5 GB conforme beam/batch. Resultado: o pre-flight (`main_window.py:430-468`) **força downgrade desnecessário** de modelo em GPUs de 6–8 GB. |

### 3.3 Correção / qualidade

| # | Local | Defeito |
|---|---|---|
| Q1 | `transcriber.py:345` | Confiança da palavra = `(probability + 2) * 50`. `probability` é linear em [0,1], não log → o resultado fica sempre em [100,150], cortado em **100**. A métrica de confiança por palavra é inútil. `if w.probability` também transforma `0.0` em 80. |
| Q2 | `transcriber.py:280-287` | Não há nenhum controle de alucinação: `condition_on_previous_text` fica no padrão `True` (causa nº 1 dos loops de repetição em áudio longo), sem `hallucination_silence_threshold` e sem pós-filtro. Se `temperature > 0`, a tupla de fallback de temperatura do Whisper é **substituída** por um escalar, o que desliga o mecanismo de re-decodificação. |
| Q3 | `app/core/exporter.py:15,24` | `ms = int((seconds % 1) * 1000)` trunca em vez de arredondar e opera em float: `1.9995` vira `,999` em vez de `2,000`. Não há quebra de linha (máx. caracteres/linha), duração máxima por tela nem CPS. Segmentos de até 30 s viram uma única legenda. O VTT não escapa `<`, `&` nem `-->`. O JSON não exporta palavras. A escrita não é atômica. |
| Q4 | `app/db/history.py:88-89` | O resultado é salvo **três vezes** (`result_json` contém `segments` + `word_segments`; `segments_json` duplica). Um job de 12 h gera uma linha de ~50–100 MB, serializada na thread da UI. |
| Q5 | `app/gui/progress.py:35` | `QTextEdit` sem limite recebe 1 linha por segmento → 15 mil+ linhas, com re-layout progressivamente mais lento. Use `QPlainTextEdit.setMaximumBlockCount(2000)`. |
| Q6 | `app/gui/result_view.py:306-336` | Cada segmento é injetado por JS num DOM do QtWebEngine (~100 mil `<span>` de palavras em 12 h) → a UI degrada linearmente. Use `QListView` + `QAbstractListModel` virtualizado. |
| Q7 | `waveform.py:213` | `QPixmap` criado fora da thread GUI (o comentário afirma que é permitido; **não é** — só `QImage` é thread-safe). Pode causar crash intermitente. |
| Q8 | `transcriber.py:224-265` | A cadeia de fallback faz downgrade **silencioso** de modelo (large-v3 → … → tiny) e de dispositivo (CUDA → CPU large-v3). Em CPU, um vídeo de 12 h com large-v3 leva **dias**, e o usuário só descobre que a qualidade caiu para `tiny` lendo o log. |

### 3.4 Vazamentos de recursos

- **RAM:** B1, B5, Q4, Q6, além de três cópias da lista de segmentos (`segments_list` no worker, `_current_result` e `result_view._segments` na UI).
- **VRAM:** B8. `torch.cuda.empty_cache()` é **irrelevante** aqui: a inferência roda no CTranslate2, que tem alocador próprio, e o `torch` só é importado em `gpu_monitor.py:18` para um `is_available()`. A VRAM do CTranslate2 só é devolvida com `del model` (ou `model.model.unload_model()`) seguido de `gc.collect()`. O tamanho do cache do alocador é limitado por `CT2_CUDA_CACHING_ALLOCATOR_CONFIG`.
- **Disco:** WAV órfão (S8). O draft nunca é removido se a restauração falhar no meio.
- **Processos:** o ffmpeg via `subprocess.run(timeout=…)` é morto no timeout, mas um cancelamento durante a conversão **não** interrompe o ffmpeg (não existe caminho de cancelamento até a conversão terminar).
- **Handles:** `_PrewarmThread` em `main.py:73` carrega large-v3 em CUDA logo na abertura, mesmo que o usuário vá usar outro modelo. São ~1–4 GB de VRAM ocupados sem necessidade, e qualquer erro é engolido (`except: pass`).

### 3.5 Exceções não tratadas ou mal classificadas

| Cenário | Comportamento atual | Exigido |
|---|---|---|
| CUDA OOM durante a inferência | "Unexpected error", job perdido | Descer um degrau na escada de fallback e refazer **só o chunk atual** |
| `cublas64_12.dll`/`cudnn_ops64_9.dll` ausente | String sem "cuda"/"oom" → `raise` → erro genérico | Classificar como `GpuRuntimeMissing` e cair para CPU com aviso explícito |
| ffmpeg crasha (segfault, código negativo) | stderr cru (até 2 kB) mostrado ao usuário | Mapear para códigos de erro (`NO_AUDIO_STREAM`, `DECODE_ERROR`, `DISK_FULL`, `PERMISSION`) com o log completo em arquivo |
| Disco cheio durante a extração | WAV parcial → ou falha no ffmpeg, ou transcrição de áudio truncado **sem aviso** | Pré-checagem com `shutil.disk_usage` + validar a duração do WAV contra o `ffprobe` (≥ 99,5%) |
| Caminho > 260 caracteres / pasta somente leitura / OneDrive "somente online" | `OSError` em lugares variados | Manifest `longPathAware`, validar escrita na pasta de exportação **antes** do job, abrir o arquivo de origem para hidratar placeholders do OneDrive |
| Arquivo de origem movido ou alterado durante o job | Comportamento indefinido | Fingerprint (tamanho + mtime + hash da cabeça e da cauda) gravado no job; na retomada, comparar |
| `json.JSONDecodeError` no draft | Draft apagado silenciosamente | Fim do draft (substituído pelo journal SQLite) |
| Duas instâncias do app | Ambas escrevem em `history.db`/`draft.json` | Mutex nomeado (`QLockFile`) + *lease* de job no SQLite |
| Windows entra em suspensão às 3 h de um job de 10 h | Job "congela"; CUDA pode falhar no resume | `SetThreadExecutionState(ES_CONTINUOUS \| ES_SYSTEM_REQUIRED)` enquanto houver job ativo |

### 3.6 Build e ambiente

- `venv/pyvenv.cfg` aponta para `C:\Python314\python.exe`, que **não existe** nesta máquina (o venv foi criado em `C:\Users\Ronald\...`). O ambiente de build não é reproduzível. `requirements.txt` não fixa versões (`>=`), não inclui `torch` e o `install.bat` instala `torchvision` (não usado).
- `upx=True` (`WhisperDesk.spec:41,54`): o UPX em DLLs CUDA/cuDNN causa corrupção ou falha de carregamento, aumenta muito o tempo de inicialização e é gatilho clássico de falso positivo de antivírus. **Desligue.**
- `--collect-all torch` = 4,0 GB por um `torch.cuda.is_available()`. Detalhe crítico: hoje é o `torch/lib` que fornece `cublas64_12.dll`/`cublasLt64_12.dll` para o CTranslate2 (que traz apenas `cudnn64_9.dll`). Remover o torch **sem** empacotar o cuBLAS quebra a GPU. Ver a Fase 4.
- O modelo é baixado no primeiro uso para `~/.cache/huggingface` (exige internet, avisos de symlink no Windows sem Developer Mode, sem retomada de download nem verificação).

### 3.7 Métricas de performance

**Definição:** RTF = tempo_de_processamento / duração_do_áudio (menor é melhor; `1/RTF` é o "× tempo real" mostrado na UI).

Etapas e o gargalo de cada uma:

| Etapa | Gargalo | Custo típico (estimativa a validar) |
|---|---|---|
| Demux + decode (ffmpeg) | **I/O de disco**: o container precisa ser lido inteiro porque vídeo e áudio estão intercalados | MKV de 50 GB: ~6 min em HDD a 150 MB/s, ~1 min em NVMe |
| VAD (Silero ONNX, CPU) | CPU, 1 thread | ~1–2% da duração do áudio |
| ASR large-v3 fp16, sequencial (beam 5) | GPU (decoder autoregressivo) | Ordem de 15–30× tempo real numa RTX 4070 Super |
| ASR com `BatchedInferencePipeline` (batch 8–16) | GPU (encoder) | 2–4× mais rápido que o sequencial |
| `word_timestamps=True` | GPU + CPU (DTW) | +10–20% |
| Exportação / SQLite | Desprezível com journal por chunk | < 1 s |

Os números acima **não foram medidos neste hardware**; os de `README.md:99-107` (3,5× para large-v3) também não têm origem verificável. Implemente a coleta abaixo e decida com dados:

```sql
-- Em cada chunk concluído, grave na tabela chunk: audio_s, elapsed_s, device, compute_type, batch_size.
-- RTF do job:
SELECT SUM(elapsed_s) / SUM(audio_s) AS rtf,
       SUM(audio_s) / SUM(elapsed_s) AS x_realtime
FROM chunk WHERE job_id = ? AND state = 'done';
```

**Regra de diagnóstico:** se a GPU fica < 60% de uso (NVML `nvmlDeviceGetUtilizationRates`) durante o ASR, o gargalo é CPU/I-O (batch pequeno demais ou VAD/decodificação no caminho crítico). Se a extração leva > 10% do tempo total, faça *pipelining* da extração com o ASR (seção 2.1, otimização B).

---

## FASE 1 — Arquitetura-alvo

### 1.1 Pipeline de ponta a ponta

```
                          ┌──────────────────── PROCESSO UI (PySide6) ────────────────────┐
 arquivo ──► Probe ──────►│ JobController: cria/retoma job no journal, spawna worker,     │
 (ffprobe)   (streams,    │ lê eventos da mp.Queue via QTimer(100 ms), lê resultados do   │
              layout,     │ SQLite (fonte da verdade), nunca toca em modelo ou CUDA        │
              start_time) └──────────────▲─────────────────────────────┬──────────────────┘
                                         │ eventos (progress/chunk_done/error)   │ terminate()
                                         │                                       ▼
 ┌──────────────────────────── PROCESSO WORKER (spawn, isolado) ─────────────────────────────┐
 │ [EXTRACTING] ffmpeg ─► audio.wav (16 kHz mono s16, RF64, aresample async) ─► valida duração│
 │ [PLANNING]   Silero VAD em janelas ─► cortes em silêncio ─► tabela chunk (pending)        │
 │ [TRANSCRIBING] para cada chunk pending:                                                  │
 │      lê fatia (soundfile, ~38 MB) ─► faster-whisper (escada de fallback) ─► filtro       │
 │      anti-alucinação ─► dono-por-corte (dedupe) ─► COMMIT atômico (segment, word, done)  │
 │ [POSTPROCESSING] costura global, normalização ─► [EXPORTING] SRT/VTT/TXT/JSON atômicos   │
 │ [DONE]  (opcional: [DIARIZING] entre TRANSCRIBING e POSTPROCESSING)                       │
 └────────────────────────────────────────────────────────────────────────────────────────────┘
          journal: %LOCALAPPDATA%\WhisperDesk\jobs\<job_id>\{journal.db, audio.wav, ffmpeg.log}
```

Decisões não negociáveis:

1. **Worker em processo separado, não em `QThread`.** Um segfault no CTranslate2/cuDNN ou um OOM nativo mata só o worker; a UI detecta `exitcode != 0` e reinicia do último chunk commitado. O cancelamento vira `terminate()`, que é seguro porque cada chunk é uma transação.
2. **O SQLite é a fonte da verdade.** A UI não acumula a lista de segmentos; ela pagina o journal.
3. **Áudio nunca inteiro em RAM.** Só a fatia do chunk atual (10 min ≈ 38 MB em float32).
4. **Timestamps em amostras inteiras** (`int64`) até a exportação. Os segundos são derivados: `(chunk_start_sample + local_s * 16000) / 16000`.

### 1.2 Chunking com VAD — cortes sem quebrar fonemas e sem duplicar palavras

O faster-whisper já aplica VAD **dentro** de cada chunk (`vad_filter=True`) para pular silêncio. O problema do áudio longo é outro: **onde cortar os macro-chunks** (unidades de checkpoint). A estratégia:

1. **Tamanho-alvo** de 600 s por chunk. Esse é o compromisso entre a granularidade do checkpoint (perda máxima de 10 min em crash) e o overhead de reinicializar a decodificação.
2. **Janela de busca** `[alvo − 60 s, alvo]`. Rode o Silero VAD só nessa janela e escolha o **maior silêncio ≥ 300 ms**; corte no seu ponto médio. Um corte em silêncio não divide fonema nenhum, então **não precisa de overlap e não há o que deduplicar**.
3. **Corte forçado** (60 s de fala contínua sem pausa ≥ 300 ms: música, multidão, narração ininterrupta): corte no quadro de 20 ms de menor energia RMS e marque o corte como `forced`. Só nesse caso os dois chunks vizinhos decodificam com **2 s de padding** além do corte.
4. **Deduplicação por posse (ownership):** cada chunk é dono do intervalo `[start, end)`. Com `word_timestamps=True`, uma palavra pertence ao chunk que contém o seu **ponto médio**. As palavras do padding fora do intervalo de posse são descartadas e o texto do segmento é reconstruído a partir das palavras mantidas. Isso é determinístico: não usa *fuzzy matching* de texto, que falha com repetições legítimas ("não, não, não").
5. **Continuidade de contexto:** `initial_prompt` de cada chunk = últimos ~200 caracteres do chunk anterior, mas **só se** esse chunk terminou com `avg_logprob > -0.5` (evita propagar alucinação). `condition_on_previous_text=False` *dentro* do chunk.

Código (`app/pipeline/planner.py`):

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf
from faster_whisper.vad import VadOptions, get_speech_timestamps

SR = 16_000


@dataclass(frozen=True)
class ChunkPlan:
    idx: int
    start: int        # amostra inicial de posse (inclusiva)
    end: int          # amostra final de posse (exclusiva)
    pad_left: int     # amostras extras decodificadas antes de start (0 se o corte foi em silêncio)
    pad_right: int


def _read(wav_path: str, start: int, stop: int) -> np.ndarray:
    data, sr = sf.read(wav_path, start=start, stop=stop, dtype="float32", always_2d=False)
    if sr != SR:
        raise ValueError(f"WAV de trabalho com sample rate inesperado: {sr}")
    return data


def _best_cut(wav_path: str, win_start: int, win_end: int, min_gap: int) -> tuple[int, bool]:
    """Retorna (amostra_de_corte, forçado?)."""
    audio = _read(wav_path, win_start, win_end)
    speech = get_speech_timestamps(
        audio,
        VadOptions(threshold=0.5, min_silence_duration_ms=150, speech_pad_ms=30),
    )
    # Silêncios = complemento das regiões de fala dentro da janela.
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for s in speech:
        if s["start"] > cursor:
            gaps.append((cursor, s["start"]))
        cursor = max(cursor, s["end"])
    if cursor < len(audio):
        gaps.append((cursor, len(audio)))

    if gaps:
        g0, g1 = max(gaps, key=lambda g: g[1] - g[0])
        if g1 - g0 >= min_gap:
            return win_start + (g0 + g1) // 2, False

    # Sem silêncio utilizável: quadro de 20 ms com menor energia.
    frame = SR // 50
    n = len(audio) // frame
    if n == 0:
        return win_end, True
    rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1))
    k = int(np.argmin(rms))
    return win_start + k * frame + frame // 2, True


def plan_chunks(
    wav_path: str,
    total_samples: int,
    target_s: int = 600,
    search_s: int = 60,
    min_gap_ms: int = 300,
    forced_pad_s: float = 2.0,
) -> list[ChunkPlan]:
    target, search = target_s * SR, search_s * SR
    min_gap = min_gap_ms * SR // 1000
    pad = int(forced_pad_s * SR)

    cuts: list[tuple[int, bool]] = [(0, False)]
    pos = 0
    while total_samples - pos > target + search:
        cut, forced = _best_cut(wav_path, pos + target - search, pos + target, min_gap)
        cuts.append((cut, forced))
        pos = cut
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
            )
        )
    return plans
```

### 1.3 Escolha da engine ASR

| Engine | Veredito | Motivo |
|---|---|---|
| **faster-whisper (CTranslate2)** | **Escolhida** (já em uso) | 2–4× mais rápido que o Whisper vanilla com a mesma acurácia; int8/fp16; sem PyTorch em runtime (economiza ~3 GB no instalador); `BatchedInferencePipeline` para throughput; Silero VAD embutido em ONNX. |
| WhisperX | Não como engine | Usa faster-whisper por baixo e acrescenta alinhamento wav2vec2 + pyannote, o que **reintroduz o PyTorch** (+2–4 GB) e dependências de modelos fechados (HF token). Só vale como fonte de ideias para o alinhamento. |
| OpenAI Whisper (vanilla) | Descartada | PyTorch, mais lento, mais VRAM, sem VAD nativo, propenso a alucinação em áudio longo. |
| APIs externas | Descartadas como padrão | Limite de tamanho de upload (≈25 MB em várias delas) exigiria chunking + upload de horas de áudio; custo, privacidade e dependência de rede. No máximo um "modo nuvem" opcional e explícito. |

Configuração: `large-v3` fp16 em GPU com ≥ 6 GB; `large-v3-turbo` (ou `distil-large-v3` para inglês) quando o throughput importa mais que a acurácia máxima; `int8` em CPU com `medium`/`small`. **Nunca** faça downgrade de modelo sem consentimento explícito.

### 1.4 Diarização escalável

O pyannote 3.x é o estado da arte aberto, mas exige PyTorch e aceite de termos no Hugging Face. Para um desktop empacotado, a recomendação é:

- **Engine:** `sherpa-onnx` (segmentação pyannote exportada para ONNX + embeddings de locutor 3D-Speaker/WeSpeaker em ONNX) → sem PyTorch, roda em CPU e cabe no instalador. Use o pyannote apenas num "pacote avançado" opcional.
- **Escalabilidade para 12 h:** não diarize o arquivo de uma vez (a clusterização é O(n²) em memória sobre os embeddings).
  1. Rode a **segmentação** por chunk (os mesmos chunks do ASR) e grave os turnos locais `(start, end, local_speaker)` no journal.
  2. Extraia **um embedding por turno** (média sobre turnos ≥ 1,5 s; ignore os mais curtos) e grave-o (`BLOB` float32 de 192–512 dimensões).
  3. **Clusterização global** sobre todos os embeddings do job: aglomerativa com distância cosseno e limiar ~0,5–0,7. Com milhares de turnos isso é trivial em RAM (5 mil × 5 mil floats = 100 MB). Acima disso, use mini-batch k-means com centroides incrementais.
  4. Mapeie os rótulos globais → turnos → **palavras** (cada palavra recebe o locutor do turno que contém o seu ponto médio).
- A diarização é um **estado próprio** da state machine (`DIARIZING`) com checkpoint por chunk. Ela é retomável e opcional.

### 1.5 Persistência e state machine

```
CREATED ─► EXTRACTING ─► PLANNING ─► TRANSCRIBING ─► [DIARIZING] ─► POSTPROCESSING ─► EXPORTING ─► DONE
   │            │             │            │               │                │               │
   └────────────┴─────────────┴────────────┴───────────────┴────────────────┴───────────────┴──► FAILED
                                     (PAUSED = processo encerrado; retomável de qualquer estado)
```

Regras:

- O estado do job só avança depois que o artefato da etapa está **validado** (por exemplo, a duração do WAV bate com o probe).
- Um chunk vai de `pending` → `running` → `done`/`failed`. Na retomada, todo `running` volta para `pending` (o processo morreu no meio).
- O commit de um chunk é **uma transação**: `DELETE` dos segmentos/palavras do chunk + `INSERT` + `UPDATE chunk SET state='done'`. Isso é idempotente: refazer um chunk nunca duplica nada.
- `PRAGMA journal_mode=WAL; synchronous=NORMAL`: numa queda de energia, perde-se no máximo a última transação (um chunk), e o banco nunca corrompe.
- Identidade do job = `sha256(fingerprint_do_arquivo + config_relevante)`. Mudou o modelo, o idioma ou a trilha → é outro job.

`app/pipeline/journal.py`:

```python
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS job (
    id            TEXT PRIMARY KEY,
    src_path      TEXT NOT NULL,
    fingerprint   TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    audio_stream  INTEGER NOT NULL,
    wav_path      TEXT,
    total_samples INTEGER,
    state         TEXT NOT NULL DEFAULT 'CREATED',
    error         TEXT,
    owner_pid     INTEGER,
    heartbeat     REAL,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS chunk (
    job_id        TEXT NOT NULL REFERENCES job(id) ON DELETE CASCADE,
    idx           INTEGER NOT NULL,
    start_sample  INTEGER NOT NULL,
    end_sample    INTEGER NOT NULL,
    pad_left      INTEGER NOT NULL,
    pad_right     INTEGER NOT NULL,
    state         TEXT NOT NULL DEFAULT 'pending'
                  CHECK (state IN ('pending','running','done','failed')),
    attempts      INTEGER NOT NULL DEFAULT 0,
    error         TEXT,
    device        TEXT,
    compute_type  TEXT,
    batch_size    INTEGER,
    audio_s       REAL,
    elapsed_s     REAL,
    tail_text     TEXT,
    tail_logprob  REAL,
    PRIMARY KEY (job_id, idx)
);

CREATE TABLE IF NOT EXISTS segment (
    job_id            TEXT NOT NULL,
    chunk_idx         INTEGER NOT NULL,
    seq               INTEGER NOT NULL,
    start_ms          INTEGER NOT NULL,
    end_ms            INTEGER NOT NULL,
    text              TEXT NOT NULL,
    avg_logprob       REAL,
    no_speech_prob    REAL,
    compression_ratio REAL,
    speaker           TEXT,
    PRIMARY KEY (job_id, chunk_idx, seq),
    FOREIGN KEY (job_id, chunk_idx) REFERENCES chunk(job_id, idx) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS segment_time ON segment(job_id, start_ms);

CREATE TABLE IF NOT EXISTS word (
    job_id     TEXT NOT NULL,
    chunk_idx  INTEGER NOT NULL,
    seg_seq    INTEGER NOT NULL,
    seq        INTEGER NOT NULL,
    start_ms   INTEGER NOT NULL,
    end_ms     INTEGER NOT NULL,
    word       TEXT NOT NULL,
    prob       REAL NOT NULL,
    PRIMARY KEY (job_id, chunk_idx, seg_seq, seq),
    FOREIGN KEY (job_id, chunk_idx, seg_seq)
        REFERENCES segment(job_id, chunk_idx, seq) ON DELETE CASCADE
);
"""

RELEVANT_CONFIG_KEYS = (
    "model_size", "language", "task", "beam_size", "vad_filter", "audio_stream", "engine_version",
)


def file_fingerprint(path: str, probe_bytes: int = 4 << 20) -> str:
    st = os.stat(path)
    h = hashlib.sha256()
    h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    with open(path, "rb") as f:
        h.update(f.read(probe_bytes))
        if st.st_size > probe_bytes:
            f.seek(max(0, st.st_size - probe_bytes))
            h.update(f.read(probe_bytes))
    return h.hexdigest()


def job_id_for(fingerprint: str, config: dict) -> str:
    relevant = {k: config.get(k) for k in RELEVANT_CONFIG_KEYS}
    payload = fingerprint + json.dumps(relevant, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class Journal:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

    # ---------- job ----------
    def upsert_job(self, job_id, src_path, fingerprint, config, audio_stream) -> sqlite3.Row:
        now = time.time()
        with self.tx() as c:
            c.execute(
                """INSERT INTO job(id, src_path, fingerprint, config_json, audio_stream, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET src_path=excluded.src_path, updated_at=excluded.updated_at""",
                (job_id, src_path, fingerprint, json.dumps(config), audio_stream, now, now),
            )
        return self.job(job_id)

    def job(self, job_id) -> sqlite3.Row:
        return self.conn.execute("SELECT * FROM job WHERE id=?", (job_id,)).fetchone()

    def set_state(self, job_id, state, error=None, **fields):
        cols = ", ".join(f"{k}=?" for k in fields)
        sql = f"UPDATE job SET state=?, error=?, updated_at=?{', ' + cols if cols else ''} WHERE id=?"
        with self.tx() as c:
            c.execute(sql, (state, error, time.time(), *fields.values(), job_id))

    def claim(self, job_id, pid: int, stale_after_s: float = 30.0) -> bool:
        """Lease: impede que duas instâncias processem o mesmo job."""
        now = time.time()
        with self.tx() as c:
            row = c.execute("SELECT owner_pid, heartbeat FROM job WHERE id=?", (job_id,)).fetchone()
            if row["owner_pid"] and row["owner_pid"] != pid and (now - (row["heartbeat"] or 0)) < stale_after_s:
                return False
            c.execute("UPDATE job SET owner_pid=?, heartbeat=? WHERE id=?", (pid, now, job_id))
            return True

    def heartbeat(self, job_id, pid: int):
        self.conn.execute("UPDATE job SET heartbeat=? WHERE id=? AND owner_pid=?", (time.time(), job_id, pid))

    # ---------- chunks ----------
    def insert_plan(self, job_id, plans):
        with self.tx() as c:
            c.execute("DELETE FROM chunk WHERE job_id=?", (job_id,))
            c.executemany(
                "INSERT INTO chunk(job_id, idx, start_sample, end_sample, pad_left, pad_right) VALUES (?,?,?,?,?,?)",
                [(job_id, p.idx, p.start, p.end, p.pad_left, p.pad_right) for p in plans],
            )

    def reset_running(self, job_id):
        self.conn.execute("UPDATE chunk SET state='pending' WHERE job_id=? AND state='running'", (job_id,))

    def pending_chunks(self, job_id) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM chunk WHERE job_id=? AND state IN ('pending','failed') ORDER BY idx", (job_id,)
        ).fetchall()

    def previous_tail(self, job_id, idx) -> tuple[str | None, float | None]:
        row = self.conn.execute(
            "SELECT tail_text, tail_logprob FROM chunk WHERE job_id=? AND idx=? AND state='done'", (job_id, idx - 1)
        ).fetchone()
        return (row["tail_text"], row["tail_logprob"]) if row else (None, None)

    def mark_running(self, job_id, idx):
        self.conn.execute(
            "UPDATE chunk SET state='running', attempts=attempts+1 WHERE job_id=? AND idx=?", (job_id, idx)
        )

    def mark_failed(self, job_id, idx, error: str):
        self.conn.execute("UPDATE chunk SET state='failed', error=? WHERE job_id=? AND idx=?", (error[:4000], job_id, idx))

    def commit_chunk(self, job_id, idx, segments: list[dict], meta: dict):
        """Transação única e idempotente: o chunk inteiro entra ou nada entra."""
        with self.tx() as c:
            c.execute("DELETE FROM segment WHERE job_id=? AND chunk_idx=?", (job_id, idx))  # cascata em word
            for s_seq, s in enumerate(segments):
                c.execute(
                    """INSERT INTO segment(job_id, chunk_idx, seq, start_ms, end_ms, text,
                                           avg_logprob, no_speech_prob, compression_ratio)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (job_id, idx, s_seq, s["start_ms"], s["end_ms"], s["text"],
                     s.get("avg_logprob"), s.get("no_speech_prob"), s.get("compression_ratio")),
                )
                c.executemany(
                    "INSERT INTO word(job_id, chunk_idx, seg_seq, seq, start_ms, end_ms, word, prob) VALUES (?,?,?,?,?,?,?,?)",
                    [(job_id, idx, s_seq, w_seq, w["start_ms"], w["end_ms"], w["word"], w["prob"])
                     for w_seq, w in enumerate(s.get("words", []))],
                )
            c.execute(
                """UPDATE chunk SET state='done', error=NULL, device=?, compute_type=?, batch_size=?,
                       audio_s=?, elapsed_s=?, tail_text=?, tail_logprob=?
                   WHERE job_id=? AND idx=?""",
                (meta["device"], meta["compute_type"], meta["batch_size"], meta["audio_s"], meta["elapsed_s"],
                 meta["tail_text"], meta["tail_logprob"], job_id, idx),
            )

    def progress(self, job_id) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT SUM(state='done') AS done, COUNT(*) AS total FROM chunk WHERE job_id=?", (job_id,)
        ).fetchone()
        return int(row["done"] or 0), int(row["total"] or 0)

    def iter_segments(self, job_id) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT * FROM segment WHERE job_id=? ORDER BY start_ms, chunk_idx, seq", (job_id,)
        )

    def iter_words(self, job_id) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute(
            "SELECT * FROM word WHERE job_id=? ORDER BY start_ms, chunk_idx, seg_seq, seq", (job_id,)
        )
```

---

## FASE 2 — Implementação

### 2.1 Concorrência: UI ↔ worker

- **Processo worker** via `multiprocessing.get_context("spawn")`. A comunicação é uma `Queue` de eventos (worker → UI) e uma `Event` de cancelamento suave. Na UI, um `QTimer` de 100 ms drena a fila sem bloquear (`get_nowait`) e atualiza os widgets.
- **Cancelamento:** suave (`cancel_event.set()`, verificado entre chunks e por segmento) e, após um timeout de 10 s, `terminate()`. Os dois são seguros por causa da transação por chunk.
- **Supervisão:** se o worker morrer com `exitcode != 0` (segfault nativo, OOM do SO), a UI relança automaticamente até 3 vezes. Após 3 crashes **no mesmo chunk**, o chunk vai para `failed`, o job pula para o próximo e o problema é reportado no fim (melhor entregar 99,8% com uma lacuna marcada do que nada).
- **`main.py`:** `multiprocessing.freeze_support()` precisa ser a **primeira** linha dentro de `if __name__ == "__main__":`. Sem isso, o `.exe` do PyInstaller faz *fork bomb* no `spawn`.
- **Otimização B (pipelining):** a extração escreve o WAV progressivamente. O planner pode cortar o chunk *k* assim que `tamanho_do_wav ≥ fim_do_chunk_k + 60 s`, sobrepondo o I/O de demux com o ASR. Implemente isso **só** se a métrica da seção 3.7 mostrar a extração acima de 10% do tempo.

`app/pipeline/worker.py`:

```python
from __future__ import annotations

import gc
import os
import re
import time
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf

from app.pipeline import ffmpeg_tools
from app.pipeline.hallucination import clean_segments
from app.pipeline.journal import Journal
from app.pipeline.planner import SR, plan_chunks

OOM_PATTERNS = re.compile(r"out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|cudaErrorMemoryAllocation", re.I)
GPU_RUNTIME_PATTERNS = re.compile(r"cublas|cudnn|cudart|CUDA driver|no CUDA-capable device|Library .* is not found", re.I)

# Escada determinística: (device, compute_type, batch_size, beam_size). batch_size=0 => pipeline sequencial.
LADDER_GPU = [
    ("cuda", "float16", 16, 5),
    ("cuda", "float16", 8, 5),
    ("cuda", "int8_float16", 4, 5),
    ("cuda", "int8_float16", 0, 5),
    ("cuda", "int8_float16", 0, 1),
    ("cpu", "int8", 0, 5),
]
LADDER_CPU = [("cpu", "int8", 0, 5)]


class Engine:
    """Mantém exatamente um modelo carregado e desce a escada sob OOM."""

    def __init__(self, model_size: str, models_dir: Path, prefer_gpu: bool, emit):
        self.model_size = model_size
        self.models_dir = models_dir
        self.ladder = LADDER_GPU if prefer_gpu else LADDER_CPU
        self.rung = 0
        self.emit = emit
        self._model = None
        self._batched = None
        self._loaded_key = None

    @property
    def step(self):
        return self.ladder[self.rung]

    def _unload(self):
        self._batched = None
        if self._model is not None:
            try:
                self._model.model.unload_model()
            except Exception:
                pass
        self._model = None
        self._loaded_key = None
        gc.collect()

    def _ensure_loaded(self):
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        device, compute, batch, _beam = self.step
        key = (device, compute)
        if self._loaded_key == key:
            if batch and self._batched is None:
                self._batched = BatchedInferencePipeline(model=self._model)
            return
        self._unload()  # libera ANTES de alocar o próximo (corrige o B8)
        self.emit("log", f"Carregando {self.model_size} em {device}/{compute}")
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute,
            download_root=str(self.models_dir),
            cpu_threads=max(1, (os.cpu_count() or 4) - 2),
        )
        self._batched = BatchedInferencePipeline(model=self._model) if batch else None
        self._loaded_key = key

    def degrade(self, reason: str) -> bool:
        if self.rung + 1 >= len(self.ladder):
            return False
        self.rung += 1
        self.emit("degraded", {"reason": reason, "step": self.step})
        self._unload()
        return True

    def transcribe(self, audio: np.ndarray, language, task, initial_prompt):
        self._ensure_loaded()
        device, compute, batch, beam = self.step
        common = dict(
            language=language,
            task=task,
            beam_size=beam,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )
        if batch:
            segments, info = self._batched.transcribe(audio, batch_size=batch, **common)
        else:
            segments, info = self._model.transcribe(
                audio,
                condition_on_previous_text=False,
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                hallucination_silence_threshold=2.0,
                initial_prompt=initial_prompt,
                **common,
            )
        return list(segments), info  # o gerador é consumido aqui, dentro do try de OOM


def _owned_segments(raw_segments, chunk, decode_start: int) -> list[dict]:
    """Converte para ms globais (a partir de amostras inteiras) e aplica a posse por ponto médio."""
    out = []
    for s in raw_segments:
        words = []
        for w in (s.words or []):
            g_start = decode_start + int(round(w.start * SR))
            g_end = decode_start + int(round(w.end * SR))
            mid = (g_start + g_end) // 2
            if chunk["start_sample"] <= mid < chunk["end_sample"]:
                words.append({
                    "start_ms": g_start * 1000 // SR,
                    "end_ms": g_end * 1000 // SR,
                    "word": w.word,
                    "prob": float(w.probability),
                })
        if not words:
            continue
        out.append({
            "start_ms": words[0]["start_ms"],
            "end_ms": words[-1]["end_ms"],
            "text": "".join(w["word"] for w in words).strip(),
            "words": words,
            "avg_logprob": float(s.avg_logprob),
            "no_speech_prob": float(s.no_speech_prob),
            "compression_ratio": float(s.compression_ratio),
        })
    return out


def run_job(job_id: str, db_path: str, models_dir: str, prefer_gpu: bool, events, cancel_event):
    """Ponto de entrada do processo worker."""
    emit = lambda kind, payload=None: events.put((kind, payload))  # noqa: E731
    j = Journal(Path(db_path))
    pid = os.getpid()
    if not j.claim(job_id, pid):
        emit("error", {"code": "JOB_LOCKED", "message": "Job em uso por outra instância."})
        return

    with ffmpeg_tools.keep_system_awake():
        try:
            job = j.job(job_id)
            cfg = __import__("json").loads(job["config_json"])
            workdir = Path(db_path).parent

            # ---- EXTRACTING ----
            if job["state"] in ("CREATED", "EXTRACTING") or not (job["wav_path"] and Path(job["wav_path"]).is_file()):
                j.set_state(job_id, "EXTRACTING")
                info = ffmpeg_tools.probe(job["src_path"])
                stream = next(s for s in info.audio_streams if s.index == job["audio_stream"])
                wav = workdir / "audio.wav"
                ffmpeg_tools.extract_wav(
                    job["src_path"], wav, stream, info.duration,
                    cancel_event=cancel_event,
                    on_progress=lambda p: emit("extract_progress", p),
                    log_path=workdir / "ffmpeg.log",
                )
                total = sf.info(str(wav)).frames
                j.set_state(job_id, "PLANNING", wav_path=str(wav), total_samples=total)
                job = j.job(job_id)

            # ---- PLANNING ----
            if job["state"] == "PLANNING":
                plans = plan_chunks(job["wav_path"], job["total_samples"])
                j.insert_plan(job_id, plans)
                j.set_state(job_id, "TRANSCRIBING")

            # ---- TRANSCRIBING ----
            j.reset_running(job_id)
            engine = Engine(cfg["model_size"], Path(models_dir), prefer_gpu, emit)
            for chunk in j.pending_chunks(job_id):
                if cancel_event.is_set():
                    j.set_state(job_id, "PAUSED")
                    emit("paused")
                    return
                j.heartbeat(job_id, pid)
                j.mark_running(job_id, chunk["idx"])
                decode_start = chunk["start_sample"] - chunk["pad_left"]
                decode_end = chunk["end_sample"] + chunk["pad_right"]
                audio, _ = sf.read(job["wav_path"], start=decode_start, stop=decode_end, dtype="float32")

                tail_text, tail_lp = j.previous_tail(job_id, chunk["idx"])
                prompt = tail_text if (tail_text and tail_lp is not None and tail_lp > -0.5) else None

                t0 = time.monotonic()
                while True:
                    try:
                        raw, _info = engine.transcribe(audio, cfg.get("language"), cfg.get("task", "transcribe"), prompt)
                        break
                    except (RuntimeError, MemoryError) as e:
                        msg = str(e)
                        if OOM_PATTERNS.search(msg) or isinstance(e, MemoryError):
                            if engine.degrade(f"OOM: {msg[:200]}"):
                                continue
                        elif GPU_RUNTIME_PATTERNS.search(msg) and engine.step[0] == "cuda":
                            while engine.step[0] == "cuda" and engine.degrade(f"GPU runtime: {msg[:200]}"):
                                pass
                            continue
                        j.mark_failed(job_id, chunk["idx"], traceback.format_exc())
                        raise

                segments = clean_segments(_owned_segments(raw, chunk, decode_start))
                last = segments[-1] if segments else None
                device, compute, batch, _ = engine.step
                j.commit_chunk(job_id, chunk["idx"], segments, {
                    "device": device, "compute_type": compute, "batch_size": batch,
                    "audio_s": (chunk["end_sample"] - chunk["start_sample"]) / SR,
                    "elapsed_s": time.monotonic() - t0,
                    "tail_text": last["text"][-200:] if last else None,
                    "tail_logprob": last["avg_logprob"] if last else None,
                })
                done, total = j.progress(job_id)
                emit("chunk_done", {"idx": chunk["idx"], "done": done, "total": total})
                del audio, raw, segments

            j.set_state(job_id, "POSTPROCESSING")
            emit("transcribed")
        except ffmpeg_tools.Cancelled:
            j.set_state(job_id, "PAUSED")
            emit("paused")
        except Exception as e:
            j.set_state(job_id, "FAILED", error=traceback.format_exc()[-4000:])
            emit("error", {"code": type(e).__name__, "message": str(e)})
```

Lado da UI (`app/gui/job_controller.py`):

```python
import multiprocessing as mp
import queue

from PySide6.QtCore import QObject, QTimer, Signal

from app.pipeline.worker import run_job


class JobController(QObject):
    event = Signal(str, object)
    crashed = Signal(int)

    MAX_RESTARTS = 3

    def __init__(self, job_id, db_path, models_dir, prefer_gpu, parent=None):
        super().__init__(parent)
        self._ctx = mp.get_context("spawn")
        self._args = (job_id, str(db_path), str(models_dir), prefer_gpu)
        self._events = self._ctx.Queue()
        self._cancel = self._ctx.Event()
        self._proc = None
        self._restarts = 0
        self._finished = False
        self._timer = QTimer(self, interval=100, timeout=self._pump)

    def start(self):
        self._finished = False
        self._proc = self._ctx.Process(target=run_job, args=(*self._args, self._events, self._cancel), daemon=True)
        self._proc.start()
        self._timer.start()

    def cancel(self, hard_after_ms: int = 10_000):
        self._cancel.set()
        QTimer.singleShot(hard_after_ms, self._hard_kill)

    def _hard_kill(self):
        if self._proc and self._proc.is_alive():
            self._proc.terminate()  # seguro: o journal é transacional por chunk

    def _pump(self):
        for _ in range(200):  # limita o trabalho por tick para não congelar a UI
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                break
            if kind in ("transcribed", "paused", "error"):
                self._finished = True
            self.event.emit(kind, payload)

        if self._proc and not self._proc.is_alive():
            code = self._proc.exitcode
            self._proc.join(0)
            self._proc = None
            if not self._finished and not self._cancel.is_set() and code != 0 and self._restarts < self.MAX_RESTARTS:
                self._restarts += 1
                self.crashed.emit(code)
                self.start()  # retoma do último chunk commitado
                return
            self._timer.stop()
```

### 2.2 Hardware: VRAM, RAM, fallback

- **Detecção real da GPU:** use `ctranslate2.get_cuda_device_count()` e `ctranslate2.get_supported_compute_types("cuda")`. Isso testa o runtime que de fato vai rodar. Remova o `torch` de `gpu_monitor.py`. Mantenha o NVML **somente** para telemetria (uso, temperatura, VRAM livre).
- **Pre-flight:** substitua a tabela `MODEL_VRAM_MB` por valores medidos no CTranslate2 (rode cada modelo uma vez com batch 1 e batch 16, registre o pico via NVML e grave como constantes). Se `vram_livre < pico_estimado`, comece pelo degrau adequado da escada — **sem** trocar de modelo.
- **Limite do alocador do CTranslate2:** defina antes de importar o `ctranslate2` (no processo worker):
  `os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "4,3,12,419430400")` (cache máximo de 400 MB). Isso reduz a VRAM retida entre chunks.
- **"Limpeza entre chunks":** com CTranslate2 não existe `empty_cache`. O padrão correto é manter **um** modelo carregado (recarregar a cada chunk custa 5–20 s) e liberar com `unload_model()` + `gc.collect()` apenas ao descer de degrau ou ao terminar. `del audio, raw, segments` a cada chunk mantém a RAM do worker estável (~38 MB de áudio + modelo).
- **Escada determinística** (`LADDER_GPU` acima): fp16/batch16 → fp16/batch8 → int8_float16/batch4 → sequencial → beam 1 → CPU int8. Cada transição é registrada na coluna `chunk.device/compute_type` e mostrada na UI. O **modelo nunca muda** sem confirmação do usuário.
- **Afinidade de CPU:** `cpu_threads = núcleos − 2` deixa espaço para UI e ffmpeg.

### 2.3 Contenção de alucinações

Em camadas:

1. **Entrada:** o VAD por chunk (`vad_filter=True`, `min_silence_duration_ms=500`) remove silêncio e trilha sem voz — a causa raiz da maioria das alucinações.
2. **Decodificação:** `condition_on_previous_text=False` (quebra o loop de auto-reforço), `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`, `no_speech_threshold=0.6`, a tupla de temperatura completa (re-decodifica quando os limiares falham) e `hallucination_silence_threshold=2.0` (pula silêncios > 2 s em torno de segmentos suspeitos; exige `word_timestamps=True`). O pipeline batched não condiciona em texto anterior por construção.
3. **Contexto entre chunks:** o `initial_prompt` só é usado se o fim do chunk anterior tiver confiança alta (seção 1.2), para não "contaminar" o próximo chunk com lixo.
4. **Pós-filtro determinístico** (`app/pipeline/hallucination.py`):

```python
from __future__ import annotations

import re
import unicodedata

# Frases clássicas que o Whisper gera em silêncio/música (dados de treino de legendas).
KNOWN_PHRASES = {
    "legendas pela comunidade amara.org",
    "obrigado por assistir",
    "inscreva-se no canal",
    "subtitles by the amara.org community",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "sous-titres réalisés par la communauté d'amara.org",
}


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = re.sub(r"[^\w\s.']", "", t)
    return re.sub(r"\s+", " ", t).strip(" .")


def _collapse_word_repeats(words: list[dict], max_repeats: int = 3) -> list[dict]:
    """Opera sobre PALAVRAS (preserva timestamps): 'a b a b a b a b a b' -> 'a b a b a b'.
    Testa n-gramas de 1 a 8 palavras."""
    for n in range(1, 9):
        keys = [_norm(w["word"]) for w in words]
        out, i = [], 0
        while i < len(words):
            gram = keys[i:i + n]
            reps = 1
            while len(gram) == n and keys[i + reps * n: i + (reps + 1) * n] == gram:
                reps += 1
            if reps > max_repeats:
                out.extend(words[i: i + n * max_repeats])
                i += reps * n
            else:
                out.append(words[i])
                i += 1
        words = out
    return words


def clean_segments(segments: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    run_key, run_len = None, 0
    for s in segments:
        key = _norm(s["text"])
        if not key:
            continue
        dur_ms = s["end_ms"] - s["start_ms"]
        nsp = s.get("no_speech_prob") or 0.0
        lp = s.get("avg_logprob") or 0.0
        cr = s.get("compression_ratio") or 0.0

        # 1) Frase conhecida de alucinação em região provável de não-fala.
        if key in KNOWN_PHRASES and (nsp > 0.3 or lp < -0.7):
            continue
        # 2) Alta compressão + baixa confiança = loop.
        if cr > 2.4 and lp < -0.8:
            continue
        # 3) Taxa de fala fisicamente impossível (> 30 caracteres/s) ou duração degenerada.
        if dur_ms <= 0 or len(key) / (dur_ms / 1000) > 30:
            continue
        # 4) Mesmo texto repetido em segmentos consecutivos: mantém no máximo 2.
        if key == run_key:
            run_len += 1
            if run_len > 2:
                continue
        else:
            run_key, run_len = key, 1

        words = s.get("words") or []
        kept = _collapse_word_repeats(words) if words else words
        if len(kept) != len(words):
            s = {**s, "words": kept,
                 "text": "".join(w["word"] for w in kept).strip(),
                 "end_ms": kept[-1]["end_ms"]}
        cleaned.append(s)
    return cleaned
```

> Observação: em idiomas sem espaço (japonês/chinês), o Whisper emite "palavras" como caracteres ou subpalavras, e repetições legítimas são mais frequentes. Nesses idiomas, suba `max_repeats` ou desative `_collapse_word_repeats` pelo código do idioma.

### 2.4 Timestamps sem drift em 5–12 h

As fontes de drift, em ordem de impacto real:

1. **Buracos/descontinuidades de PTS na origem** (S4). Correção na extração: `-af aresample=async=1:min_hard_comp=0.100:first_pts=0`. O ffmpeg insere silêncio onde faltam amostras e descarta sobreposições, então a amostra *n* do WAV corresponde exatamente a `n/16000` s **do relógio do container**.
2. **`start_time` ≠ 0** (MP4 com edit list, `.ts` começando em 1,4 s, por exemplo). O player alinha a legenda ao início da *apresentação*. Com `first_pts=0` o WAV começa em zero na apresentação; registre `stream.start_time` no job e documente que os timestamps são relativos ao início da apresentação (o que os players esperam).
3. **Aritmética em float acumulada.** Nunca some durações de chunk em float. O offset global é sempre `chunk_start_sample` (int64), e a conversão para ms é `amostra * 1000 // 16000` em inteiro.
4. **Ressampling sem *sample-accuracy*.** `aresample` com `soxr` (`-af ...,aresample=16000:resampler=soxr`) é sample-accurate. A contagem de amostras é verificada: `|frames/16000 − duração_probe| < 0,5 s`; senão o job falha com `DURATION_MISMATCH`.

Validação automatizada: gere um arquivo de teste de 6 h com um *beep* de 1 kHz a cada 10 min (`ffmpeg -f lavfi -i "sine=f=1000:d=0.2,apad=pad_dur=599.8" -t 21600 ...`) muxado num MKV com um buraco artificial de 3 s. O VAD/planner deve localizar cada beep a ±20 ms da posição esperada.

`app/pipeline/ffmpeg_tools.py`:

```python
from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
SR = 16_000


class AudioError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    pass


def _bin_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "bin"          # PyInstaller onedir: _internal\bin
    return Path(__file__).resolve().parents[2] / "bin"


def tool(name: str) -> str:
    exe = f"{name}.exe" if os.name == "nt" else name
    bundled = _bin_dir() / exe
    if bundled.is_file():
        return str(bundled)
    found = shutil.which(name)               # só para desenvolvimento
    if found:
        return found
    raise AudioError("FFMPEG_MISSING", f"{exe} não encontrado em {_bin_dir()}")


@dataclass(frozen=True)
class AudioStream:
    index: int            # índice absoluto no container (para -map 0:<index>)
    codec: str
    channels: int
    layout: str
    sample_rate: int
    language: str | None
    title: str | None
    start_time: float


@dataclass(frozen=True)
class ProbeInfo:
    duration: float
    audio_streams: list[AudioStream]


def probe(path: str) -> ProbeInfo:
    try:
        r = subprocess.run(
            [tool("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
            capture_output=True, encoding="utf-8", errors="replace", timeout=300, creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        raise AudioError("PROBE_TIMEOUT", "ffprobe excedeu 5 min (arquivo em rede lenta ou corrompido).")
    if r.returncode != 0:
        raise AudioError("PROBE_FAILED", r.stderr.strip()[-1000:])
    data = json.loads(r.stdout or "{}")

    streams = []
    for s in data.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        tags = s.get("tags", {})
        try:
            streams.append(AudioStream(
                index=int(s["index"]),
                codec=s.get("codec_name", "?"),
                channels=int(s.get("channels") or 0),
                layout=s.get("channel_layout") or "",
                sample_rate=int(s.get("sample_rate") or 0),
                language=tags.get("language"),
                title=tags.get("title"),
                start_time=float(s.get("start_time") or 0.0),
            ))
        except (KeyError, ValueError):
            continue
    if not streams:
        raise AudioError("NO_AUDIO_STREAM", "O arquivo não contém trilha de áudio decodificável.")

    duration = 0.0
    for candidate in (data.get("format", {}).get("duration"),
                      *(s.get("duration") for s in data.get("streams", []))):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    if duration <= 0:
        raise AudioError("NO_DURATION", "Duração indeterminável (arquivo truncado?).")
    return ProbeInfo(duration=duration, audio_streams=streams)


def _downmix_filter(stream: AudioStream) -> str:
    """Prioriza o canal central (diálogo) em 5.1/7.1; downmix padrão nos demais."""
    if stream.channels >= 6 and stream.layout.split("(")[0] in {"5.1", "6.1", "7.1"}:
        return "pan=mono|c0=0.60*FC+0.20*FL+0.20*FR"
    return "aformat=channel_layouts=mono"


def extract_wav(src: str, dst: Path, stream: AudioStream, duration: float, *,
                cancel_event, on_progress, log_path: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    need = int(duration * SR * 2 * 1.05) + (64 << 20)
    free = shutil.disk_usage(dst.parent).free
    if free < need:
        raise AudioError("DISK_FULL", f"Espaço insuficiente: {need >> 20} MB necessários, {free >> 20} MB livres.")

    tmp = dst.with_suffix(".partial.wav")
    af = ",".join([
        "aresample=async=1:min_hard_comp=0.100:first_pts=0",
        _downmix_filter(stream),
        f"aresample={SR}:resampler=soxr",
    ])
    cmd = [
        tool("ffmpeg"), "-hide_banner", "-nostdin", "-y",
        "-fflags", "+discardcorrupt+genpts", "-err_detect", "ignore_err",
        "-i", src,
        "-map", f"0:{stream.index}", "-vn", "-sn", "-dn",
        "-af", af, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
        "-rf64", "auto", "-f", "wav",
        "-progress", "pipe:1", "-nostats",
        str(tmp),
    ]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW, encoding="utf-8", errors="replace", bufsize=1,
        )
        watchdog_stop = threading.Event()

        def _watch():
            while not watchdog_stop.wait(0.5):
                if cancel_event.is_set() and proc.poll() is None:
                    proc.kill()
                    return

        threading.Thread(target=_watch, daemon=True).start()
        try:
            for line in proc.stdout:
                if line.startswith("out_time_us="):
                    v = line.split("=", 1)[1].strip()
                    if v.isdigit() and duration > 0:
                        on_progress(min(1.0, int(v) / 1e6 / duration))
            rc = proc.wait()
        finally:
            watchdog_stop.set()
            if proc.poll() is None:
                proc.kill()

    if cancel_event.is_set():
        tmp.unlink(missing_ok=True)
        raise Cancelled()
    if rc != 0:
        tmp.unlink(missing_ok=True)
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise AudioError("DECODE_FAILED", f"ffmpeg saiu com código {rc}.\n{tail}")

    import soundfile as sf
    frames = sf.info(str(tmp)).frames
    if abs(frames / SR - duration) > max(0.5, duration * 0.005):
        raise AudioError(
            "DURATION_MISMATCH",
            f"Áudio extraído tem {frames / SR:.1f}s; o container declara {duration:.1f}s. "
            f"Possível corrupção — veja {log_path}.",
        )
    os.replace(tmp, dst)   # atômico no mesmo volume


@contextmanager
def keep_system_awake():
    """Impede a suspensão do Windows enquanto houver job ativo."""
    if os.name != "nt":
        yield
        return
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
```

> **Atenção ao `DURATION_MISMATCH`:** com `aresample=async=1` a contagem de amostras deve bater com a duração do container. A tolerância de 0,5% cobre containers cuja duração declarada inclui a trilha de vídeo mais longa. Se houver falsos positivos em produção, rebaixe para aviso e mantenha o log.

### 2.5 Exportação multi-formato

Regras ergonômicas (padrão de legendagem profissional):

| Parâmetro | Valor |
|---|---|
| Caracteres por linha | ≤ 42 |
| Linhas por legenda | ≤ 2 |
| Duração | 0,833 s ≤ d ≤ 7 s |
| Velocidade de leitura | ≤ 20 caracteres/s (17 para público infantil) |
| Intervalo mínimo entre legendas | 80 ms (2 quadros a 25 fps) |
| Quebra preferencial | Após pontuação; nunca separar artigo/preposição do substantivo (heurística: não quebrar depois de palavras de até 2 letras) |
| Pausa que força nova legenda | ≥ 700 ms entre palavras |

`app/pipeline/exporters.py`:

```python
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_LINE = 42
MAX_LINES = 2
MAX_DUR_MS = 7000
MIN_DUR_MS = 833
MAX_CPS = 20.0
MIN_GAP_MS = 80
PAUSE_SPLIT_MS = 700
SENTENCE_END = (".", "?", "!", "…", "。", "？", "！")
CLAUSE_END = (",", ";", ":", "—")


@dataclass
class Cue:
    start_ms: int
    end_ms: int
    lines: list[str]
    speaker: str | None = None


def _fmt(ms: int, sep: str) -> str:
    ms = max(0, int(ms))
    h, r = divmod(ms, 3_600_000)
    m, r = divmod(r, 60_000)
    s, ms = divmod(r, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


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


def build_cues(words: Iterable[dict]) -> list[Cue]:
    """words: dicts com start_ms, end_ms, word (com espaço inicial como no Whisper), speaker opcional."""
    cues: list[Cue] = []
    cur: list[dict] = []

    def text_of(ws):
        return "".join(w["word"] for w in ws).strip()

    def flush():
        if cur:
            t = text_of(cur)
            cues.append(Cue(cur[0]["start_ms"], cur[-1]["end_ms"], _wrap(t) or [t], cur[0].get("speaker")))
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
            too_fast = dur > 0 and len(cand_text) / (dur / 1000) > MAX_CPS * 1.5
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


def to_srt(cues: list[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        lines = list(c.lines)
        if c.speaker:
            lines[0] = f"[{c.speaker}] {lines[0]}"
        out.append(f"{i}\n{_fmt(c.start_ms, ',')} --> {_fmt(c.end_ms, ',')}\n" + "\n".join(lines) + "\n")
    return "\n".join(out)


def to_vtt(cues: list[Cue]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        body = "\n".join(_vtt_escape(line) for line in c.lines)
        if c.speaker:
            body = f"<v {_vtt_escape(c.speaker)}>{body}"
        out.append(f"{_fmt(c.start_ms, '.')} --> {_fmt(c.end_ms, '.')}\n{body}\n")
    return "\n".join(out)


def to_txt(segments: Iterable[dict], timestamps: bool = False) -> str:
    lines = []
    for s in segments:
        prefix = f"[{_fmt(s['start_ms'], '.')[:-4]}] " if timestamps else ""
        speaker = f"{s['speaker']}: " if s.get("speaker") else ""
        lines.append(f"{prefix}{speaker}{s['text']}")
    return "\n".join(lines) + "\n"


def write_json_streaming(path: Path, metadata: dict, segments: Iterable[dict], words_by_seg) -> None:
    """Escrita em streaming: não monta o documento inteiro em RAM (12 h ≈ 100k palavras)."""
    def body(f):
        f.write('{"schema_version": 1, "metadata": ')
        json.dump(metadata, f, ensure_ascii=False)
        f.write(', "segments": [\n')
        first = True
        for s in segments:
            if not first:
                f.write(",\n")
            first = False
            item = {k: s[k] for k in ("start_ms", "end_ms", "text", "speaker", "avg_logprob", "no_speech_prob") if k in s}
            item["words"] = [
                {"w": w["word"], "s": w["start_ms"], "e": w["end_ms"], "p": round(w["prob"], 4)}
                for w in words_by_seg(s)
            ]
            json.dump(item, f, ensure_ascii=False)
        f.write("\n]}\n")
    atomic_write(path, body)


def atomic_write(path: Path, writer) -> None:
    """Escreve num temporário no MESMO diretório, fsync, e troca atomicamente."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
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
```

> Observações: SRT com BOM UTF-8 (`encoding="utf-8-sig"`) aumenta a compatibilidade com players antigos no Windows; deixe como opção. O ajuste de `end_ms` em `build_cues` nunca encurta uma legenda abaixo do fim real da última palavra, exceto para evitar sobreposição com a próxima.

---

## FASE 4 — Distribuição Windows

### 4.1 Empacotamento

| Ferramenta | Veredito |
|---|---|
| **PyInstaller `--onedir`** | **Manter.** Já funciona com CTranslate2/PySide6, tem hooks maduros e o *hot path* é código nativo (o ganho de compilar o Python é nulo). `--onefile` é proibido: extrairia ~1–2 GB para `%TEMP%` a cada execução. |
| Nuitka | Viável, mas builds de 30–60 min, maior atrito com DLLs CUDA, e nenhum ganho de runtime relevante aqui. Considere só se for exigida ofuscação. |
| PyOxidizer | Descartado: projeto sem manutenção ativa. |
| Tauri/Electron | Só faria sentido com reescrita do frontend; a UI é PySide6. |

Spec de produção (`WhisperDesk.spec`), resumido:

```python
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "assets"), "assets"),
    *collect_data_files("faster_whisper"),          # silero_vad_v6.onnx etc.
]
binaries = [
    *collect_dynamic_libs("ctranslate2"),
    (str(ROOT / "vendor" / "ffmpeg" / "ffmpeg.exe"), "bin"),
    (str(ROOT / "vendor" / "ffmpeg" / "ffprobe.exe"), "bin"),
]
# GPU: DLLs vindas das wheels nvidia-* (ver 4.3). Elas vão para _internal\cuda.
for wheel_bin in (ROOT / "vendor" / "cuda").glob("*.dll"):
    binaries.append((str(wheel_bin), "cuda"))

a = Analysis(
    [str(ROOT / "main.py")],
    binaries=binaries,
    datas=datas,
    hiddenimports=["ctranslate2", "faster_whisper", "onnxruntime", "soundfile", "pynvml"],
    excludes=["torch", "torchaudio", "torchvision", "matplotlib.tests", "numpy.tests", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="WhisperDesk",
    console=False,
    upx=False,                                    # nunca UPX com CUDA/cuDNN
    icon=str(ROOT / "assets" / "app.ico"),
    version=str(ROOT / "packaging" / "version_info.txt"),
    manifest=str(ROOT / "packaging" / "app.manifest"),   # longPathAware + dpiAwareness PerMonitorV2
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="WhisperDesk")
```

`packaging/app.manifest` (trecho relevante):

```xml
<application xmlns="urn:schemas-microsoft-com:asm.v3">
  <windowsSettings>
    <longPathAware xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">true</longPathAware>
    <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
  </windowsSettings>
</application>
```

Ponto de entrada (`main.py`) — ordem obrigatória:

```python
import multiprocessing
import os
import sys
from pathlib import Path


def _register_dll_dirs():
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    for sub in ("cuda", "ctranslate2"):
        d = base / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
    # GPU Pack baixado depois da instalação (ver 4.3)
    gpu_pack = Path(os.environ.get("LOCALAPPDATA", "")) / "WhisperDesk" / "gpu"
    if gpu_pack.is_dir():
        os.add_dll_directory(str(gpu_pack))
        os.environ["PATH"] = str(gpu_pack) + os.pathsep + os.environ["PATH"]


if __name__ == "__main__":
    multiprocessing.freeze_support()      # PRIMEIRA instrução: o processo worker spawnado reentra aqui
    _register_dll_dirs()
    os.environ.setdefault("CT2_CUDA_CACHING_ALLOCATOR_CONFIG", "4,3,12,419430400")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from app.bootstrap import main
    main()
```

> O `PATH` é atualizado além de `add_dll_directory` porque o `ctranslate2.dll` carrega o cuBLAS/cuDNN com `LoadLibrary` simples em algumas versões, e esse caminho não consulta os diretórios registrados por `add_dll_directory`. Os processos spawnados herdam o `PATH`.

### 4.2 FFmpeg embutido

- Use builds estáticos x64 versionados (por exemplo, a série "release" do BtbN ou do gyan.dev), em `vendor/ffmpeg/`, com SHA-256 registrado no repositório e verificado pelo `build.bat`.
- **Licença:** builds "full/GPL" incluem x264/x265 (GPL). Distribuí-los como executáveis separados chamados por `subprocess` é agregação, mas você **precisa** distribuir o texto da licença e uma oferta de código-fonte. A rota mais limpa para um produto comercial é um build **LGPL** (você só precisa de decoders de áudio + demuxers; encoders GPL não são usados).
- Resolução: sempre `_internal\bin\ffmpeg.exe` (função `tool()` na seção 2.4). O `PATH` só é usado em desenvolvimento. Nunca use `where` nem `shell=True`.
- Toda chamada usa `creationflags=CREATE_NO_WINDOW`, `stdin=DEVNULL` e `encoding="utf-8", errors="replace"`. Os argumentos vão como lista (o `CreateProcessW` preserva o Unicode dos nomes de arquivo).

### 4.3 CUDA/cuDNN sem instalação manual

Hoje a GPU funciona "por acidente": o `torch/lib` traz o `cublas64_12.dll`/`cublasLt64_12.dll` e o CTranslate2 os encontra. O caminho correto:

1. **Descubra as dependências exatas** da sua versão do CTranslate2 (4.8 no bundle atual):
   ```bash
   dumpbin /dependents dist\WhisperDesk\_internal\ctranslate2\ctranslate2.dll
   ```
   Para CT2 4.x (CUDA 12 + cuDNN 9), o conjunto esperado é `cublas64_12.dll`, `cublasLt64_12.dll` e `cudnn64_9.dll` + `cudnn_ops64_9.dll`, `cudnn_cnn64_9.dll`, `cudnn_graph64_9.dll`, `cudnn_engines_*64_9.dll`, `cudnn_heuristic64_9.dll` (o cuDNN 9 é modular e carrega os sub-módulos dinamicamente). O `cudart` costuma ser linkado estaticamente — **confirme** com o `dumpbin`, não presuma.
2. **Origem das DLLs:** as wheels oficiais `nvidia-cublas-cu12` e `nvidia-cudnn-cu12` (versão 9.x), fixadas por versão. O `build.bat` extrai `nvidia/*/bin/*.dll` para `vendor/cuda/`. A redistribuição dessas DLLs é permitida pelo EULA da NVIDIA (anexo de redistribuíveis) — inclua o aviso de licença.
3. **Driver:** o usuário precisa só do **driver** NVIDIA ≥ R525 (CUDA 12.x) — nunca do Toolkit. Verifique no startup via NVML (`nvmlSystemGetCudaDriverVersion_v2`) e mostre uma mensagem acionável se for antigo.
4. **Tamanho:** cuBLAS + cuDNN ≈ 1,0–1,3 GB. Duas opções:
   - **Instalador único** (~1,5 GB comprimido com LZMA2): mais simples para o usuário.
   - **Recomendado:** **instalador base** (CPU, ~250 MB) + **"GPU Pack"** baixado sob demanda quando o NVML detectar uma GPU NVIDIA. O pack é um ZIP versionado com manifesto SHA-256, extraído para `%LOCALAPPDATA%\WhisperDesk\gpu\` e registrado em `_register_dll_dirs()`. O download é retomável (HTTP Range) e verificado **antes** de extrair.
5. **Auto-teste no primeiro uso:** no worker, `ctranslate2.get_cuda_device_count() > 0` + carregar `tiny` em `cuda/float16` e transcrever 1 s de silêncio. Se falhar com `GPU_RUNTIME_PATTERNS`, marque a GPU como indisponível (com o erro exato no log) e use a escada de CPU. Cache o resultado por versão do driver.
6. **Remova o `torch` por completo** (`gpu_monitor.py:18`, `install.bat:52`, `build.bat`, spec). Isso tira 4,0 GB do bundle e elimina o risco de duas cópias diferentes de `cudnn64_9.dll` no mesmo processo.

### 4.4 Instalador per-user (sem UAC)

Diretórios (substituem o `~/.whisperdesk`, com migração no primeiro start):

| Uso | Caminho |
|---|---|
| Programa | `%LOCALAPPDATA%\Programs\WhisperDesk\` |
| Config | `%APPDATA%\WhisperDesk\config.json` (roaming, pequeno) |
| Jobs / journal / WAV de trabalho | `%LOCALAPPDATA%\WhisperDesk\jobs\<job_id>\` (configurável para outro disco: o WAV de 12 h tem ~1,4 GB) |
| Modelos | `%LOCALAPPDATA%\WhisperDesk\models\` (`download_root` do `WhisperModel`) |
| GPU Pack | `%LOCALAPPDATA%\WhisperDesk\gpu\` |
| Logs | `%LOCALAPPDATA%\WhisperDesk\logs\` (rotacionados, 10 × 5 MB) |

Higiene dos jobs: o WAV de trabalho é apagado ao chegar em `DONE` (o journal fica). Jobs em `PAUSED`/`FAILED` com mais de 30 dias são oferecidos para limpeza na UI — **nunca** apagados silenciosamente.

`packaging/whisperdesk.iss` (Inno Setup 6):

```ini
#define AppName "WhisperDesk"
#define AppVersion GetVersionNumbersString("..\dist\WhisperDesk\WhisperDesk.exe")
#define AppExe "WhisperDesk.exe"

[Setup]
AppId={{7B1E6E52-4C1D-4D7A-9B7E-3C2A5F0D9A11}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=WhisperDesk
DefaultDirName={localappdata}\Programs\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
Compression=lzma2/ultra64
SolidCompression=yes
LZMANumBlockThreads=8
DiskSpanning=no
OutputDir=..\installer
OutputBaseFilename={#AppName}-{#AppVersion}-x64-setup
SetupIconFile=..\assets\app.ico
UninstallDisplayIcon={app}\{#AppExe}
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
; SignTool=signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 /a $f

[Languages]
Name: "ptbr"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\dist\WhisperDesk\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
Name: "{localappdata}\WhisperDesk\jobs"
Name: "{localappdata}\WhisperDesk\models"
Name: "{localappdata}\WhisperDesk\logs"

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Remove só o que é regenerável. Jobs, modelos e config ficam (o usuário decide).
Type: filesandordirs; Name: "{localappdata}\WhisperDesk\logs"
Type: filesandordirs; Name: "{localappdata}\WhisperDesk\gpu"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if CheckForMutexes('WhisperDesk.SingleInstance') then
  begin
    MsgBox('Feche o WhisperDesk antes de instalar/atualizar.', mbError, MB_OK);
    Result := False;
  end;
end;
```

O app cria o mutex `WhisperDesk.SingleInstance` no startup (via `ctypes.windll.kernel32.CreateMutexW`), o que também resolve o problema das duas instâncias da seção 3.5.

**Assinatura de código:** sem Authenticode, o SmartScreen bloqueia o instalador e os antivírus tendem a sinalizar o bootloader do PyInstaller. Assine o `WhisperDesk.exe`, o instalador e (opcionalmente) o `ffmpeg.exe` re-empacotado. Um certificado EV elimina o período de "reputação" do SmartScreen.

**Build reprodutível:**

- Python **3.12** x64 fixado (3.14 é recente demais para o ecossistema CUDA/ONNX; o venv atual aponta para um intérprete que não existe nesta máquina).
- `uv pip compile` / `pip-tools` → `requirements.lock` com hashes; `pip install --require-hashes`.
- O pipeline `build.bat` vira: venv limpo → instala o lock → baixa/verifica ffmpeg e as wheels nvidia (SHA-256) → PyInstaller → smoke test (`WhisperDesk.exe --selftest`, que transcreve um WAV de 10 s com `tiny` em CPU e GPU) → Inno Setup → assinatura.

---

## Checklist de implementação (ordem de execução)

**Sprint 1 — fazer o vídeo de 12 h terminar**
- [ ] `ffmpeg_tools.py`: ffmpeg embutido, `CREATE_NO_WINDOW`, `-map`, downmix com centro priorizado, `aresample async`, `-rf64`, progresso, cancelamento, checagem de disco, validação de duração.
- [ ] `journal.py` + `planner.py` + `worker.py` (processo separado, chunks de 600 s em silêncio, commit por chunk).
- [ ] `JobController` na UI; remover `TranscriptionWorker`, o draft JSON e o `_PrewarmThread`.
- [ ] Na abertura: listar jobs `PAUSED`/`FAILED`/`TRANSCRIBING` e oferecer "Retomar".
- [ ] `multiprocessing.freeze_support()` e mutex de instância única.
- [ ] `SetThreadExecutionState` durante os jobs.

**Sprint 2 — qualidade**
- [ ] Parâmetros anti-alucinação + `hallucination.py`.
- [ ] Escada de fallback por chunk; nunca trocar de modelo sem consentimento.
- [ ] `exporters.py` (cues a partir de palavras, escrita atômica, JSON em streaming com palavras).
- [ ] Corrigir a confiança de palavra (Q1): exibir `probability * 100`.
- [ ] Waveform: gerar a partir do WAV de trabalho com `soundfile` em blocos (nunca `pydub` sobre o original); `QImage` no worker, `QPixmap` só na thread GUI.
- [ ] Resultados: `QListView` virtualizado lendo do journal, no lugar do DOM no QtWebEngine; `QPlainTextEdit` com `setMaximumBlockCount`.
- [ ] Histórico = jobs `DONE` do journal; remover `result_json`/`segments_json` duplicados.

**Sprint 3 — distribuição**
- [ ] Remover o `torch`; detecção pelo `ctranslate2`; DLLs nvidia-* fixadas; `upx=False`.
- [ ] Diretórios em `%LOCALAPPDATA%`/`%APPDATA%` + migração do `~/.whisperdesk`.
- [ ] Spec + manifest + Inno Setup + assinatura + `--selftest`.
- [ ] GPU Pack sob demanda (opcional, se o tamanho do instalador importar).

**Sprint 4 — opcional**
- [ ] Diarização via sherpa-onnx (estado `DIARIZING`, clusterização global).
- [ ] Pipelining extração/ASR se as métricas justificarem.

**Testes de aceitação (obrigatórios antes do release)**
- [ ] Vídeo de 12 h: matar o processo (`taskkill /F`) em 10%, 50% e 98% → a retomada gera uma saída **byte a byte idêntica** à de uma execução sem interrupção (mesma configuração, com o `initial_prompt` determinístico).
- [ ] Arquivo de beeps de 6 h com buraco de PTS: erro de timestamp ≤ 20 ms no fim.
- [ ] MKV com 3 trilhas (5.1 dublado, estéreo original, comentário): a trilha escolhida é a transcrita.
- [ ] Arquivo com áudio corrompido no meio: o job termina, e o trecho afetado é reportado.
- [ ] Nome de arquivo `C:\Vídeos\日本語 — teste (1).mkv`, caminho > 260 caracteres e pasta de exportação somente leitura (o erro aparece **antes** do job começar).
- [ ] GPU de 4 GB com large-v3: desce a escada sem falhar e registra os degraus.
- [ ] Máquina limpa sem Python, ffmpeg nem CUDA Toolkit, só com o driver NVIDIA: instala sem UAC e transcreve em GPU.
- [ ] RAM do worker estável (±100 MB) do chunk 1 ao chunk 72.

---

## Status da implementação — build 2.0.0 (09/11/2026)

**Implementado e verificado nesta máquina (RTX 4070 SUPER, Windows 11):**
- Pipeline novo em `app/pipeline/`: worker em processo `spawn` isolado, diário SQLite com checkpoint por bloco, retomada após pausa ou queda, escada de fallback GPU → CPU, chunking por VAD, filtro de alucinações, exportadores atômicos. O código legado (`app/core/transcriber.py`, `exporter.py`, `audio_processor.py`, `app/db/`) foi removido.
- GUI reescrita (`main_window.py` + `job_controller.py`): oferta de retomada ao abrir, pausa ao fechar, instância única com "Abrir com", i18n pt/en completa, seguindo o idioma do sistema.
- 33 testes pytest (planner com VAD real, journal concorrente, ffmpeg com MKV de 2 trilhas e caminho unicode, exportadores, serviço).
- Distribuição: `WhisperDesk.spec` (onedir, sem UPX, manifesto longPathAware/PerMonitorV2, version info), `tools/prepare_build.py` (ffmpeg + cuBLAS com SHA-256), `build.bat` de ponta a ponta, `installer/WhisperDesk.iss`, `THIRD_PARTY_NOTICES.txt`, `requirements*.txt` com versões fixadas.
- Executável congelado testado: `--selftest` OK; `--selftest-transcribe --spawn` OK em CUDA float16 (52 s de áudio em 1,7 s) e em CPU int8 (vídeo MKV de 22 min em 41 s); download do modelo `tiny` do zero OK; GUI abre e registra o log. A pasta `dist\WhisperDesk` tem cerca de 1,5 GB (cuBLAS ≈ 736 MB, ffmpeg ≈ 435 MB).

**Ainda não verificado (fazer antes do release):**
- O instalador não foi compilado aqui porque o Inno Setup não está instalado. `build.bat` gera o instalador automaticamente quando o `ISCC.exe` existir.
- Assinatura de código (SmartScreen).
- Testes de aceitação da lista acima que exigem outro ambiente: máquina limpa, GPU de 4 GB, arquivos de 6–12 h com `taskkill` no meio da transcrição.
- Diarização (Sprint 4) não implementada.

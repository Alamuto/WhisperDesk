<div align="center">

<img src="assets/icon.png" width="96" alt="WhisperDesk">

# WhisperDesk

**Transcrição de áudio e vídeo 100% local para Windows, com Whisper e aceleração NVIDIA CUDA.**

Seu áudio nunca sai do computador.

![Windows 10/11 x64](https://img.shields.io/badge/Windows-10%20%7C%2011%20x64-0078D6?logo=windows)
![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![PySide6 / Qt 6](https://img.shields.io/badge/UI-PySide6%20%2F%20Qt%206-41CD52?logo=qt&logoColor=white)
![CUDA 12](https://img.shields.io/badge/GPU-CUDA%2012-76B900?logo=nvidia&logoColor=white)
![Versão 2.0.0](https://img.shields.io/badge/vers%C3%A3o-2.0.0-7C5CFF)
![Licença MIT](https://img.shields.io/badge/licen%C3%A7a-MIT-informational)

</div>

> **English summary** — WhisperDesk is a private, offline desktop transcriber for Windows built on
> [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) and Qt 6. Drop in any audio or video file,
> get an accurate, timestamped transcript with a synced player, search, and export to TXT/SRT/VTT/JSON/CSV.
> Long recordings (hours) are processed in silence-aligned chunks with an on-disk journal, so a crash, a reboot or a
> pause never loses finished work. NVIDIA GPUs are used automatically, with a transparent fallback to the CPU.
> The only network access is the one-time download of each model.

---

## Sumário

- [O que é](#o-que-é)
- [Escopo do projeto](#escopo-do-projeto)
- [Recursos](#recursos)
- [Interface](#interface)
- [Instalação](#instalação)
- [Modelos](#modelos)
- [Onde ficam os dados](#onde-ficam-os-dados)
- [Diagnóstico](#diagnóstico)
- [Desenvolvimento](#desenvolvimento)
- [Build de distribuição](#build-de-distribuição)
- [Arquitetura](#arquitetura)
- [Roadmap](#roadmap)
- [Licenças](#licenças)

## O que é

O WhisperDesk é um aplicativo de desktop para transformar gravações — reuniões, aulas, entrevistas, podcasts, vídeos —
em texto pesquisável e legendas, sem enviar nada para a nuvem. Ele empacota o modelo Whisper da OpenAI (via
faster-whisper / CTranslate2), o FFmpeg e o runtime CUDA num instalador único: o usuário final não precisa instalar
Python, CUDA Toolkit nem FFmpeg.

## Escopo do projeto

**Dentro do escopo**

- Transcrição **offline** de qualquer áudio ou vídeo que o FFmpeg leia, em ~100 idiomas, com detecção automática.
- Tradução de fala para **inglês** (modelos que suportam a tarefa).
- Gravações **longas** (de minutos a 12 h ou mais) e arquivos de dezenas de GB, com uso de memória constante.
- **Robustez de produção**: trabalhos retomáveis após pausa, fechamento do app, queda de energia ou travamento do worker.
- Uso automático de **GPU NVIDIA**, com redução de precisão e volta para a CPU quando falta VRAM.
- Revisão do resultado: player sincronizado, busca, cores por confiança e exportação para formatos de legenda e dados.
- Distribuição para **Windows 10/11 x64** como instalador por usuário (sem UAC) ou pasta portátil.

**Fora do escopo (por decisão)**

- Serviços em nuvem, contas, telemetria ou qualquer envio de áudio/texto para fora da máquina.
- Transcrição em tempo real do microfone e identificação de falantes (diarização) — candidatos a versões futuras.
- macOS e Linux (o código é quase todo portátil, mas o empacotamento e os testes cobrem só Windows).
- Edição de vídeo ou queima de legendas no vídeo.

## Recursos

- **Arrastar e soltar** em qualquer ponto da janela, "Abrir com" no Explorer ou instância única que recebe novos arquivos;
  escolha da faixa de áudio em vídeos com várias faixas.
- **Arquivos longos** divididos em blocos cortados nos silêncios (Silero VAD). Cada bloco concluído é gravado num diário
  SQLite: dá para pausar, fechar o app ou sobreviver a uma queda e retomar do último bloco.
- **Worker em processo separado** (`spawn`): um travamento do modelo nunca derruba a interface.
- **GPU NVIDIA** em float16 / int8_float16 com fallback automático para CPU se faltar VRAM ou a GPU falhar.
- **Filtro de alucinações** (repetições em loop, "Obrigado por assistir" em silêncio etc.).
- **Resultado interativo**: timestamps clicáveis, destaque palavra a palavra sincronizado com o player, cores por
  confiança, busca com navegação entre resultados.
- **Exportação** TXT, SRT, VTT, JSON e CSV (escrita atômica).
- **Histórico** de transcrições com filtro, tema claro/escuro e interface em português e inglês (segue o sistema).
- **Diagnóstico embutido**: `--selftest` e `--selftest-transcribe` para suporte e CI.

## Interface

A versão 2.0 traz uma interface redesenhada, pensada como produto:

- Layout em duas áreas — **configurações à esquerda**, **documento à direita** — com o botão principal sempre no mesmo lugar.
- Tela inicial com uma grande área de soltar arquivos e a lista de **recentes**.
- **Player com forma de onda** (clique para pular, bolha de tempo no hover, velocidades de 0,75× a 2×).
- Gavetas animadas para **histórico** e **log técnico**, sem espremer o conteúdo.
- Temas **claro e escuro** com paletas próprias e transição em cross-fade.
- Animações discretas: seletores com pílula deslizante, carregamento em esqueleto, progresso suavizado, notificações.
- Atalhos: `Ctrl+O` abrir, `Ctrl+Enter` transcrever, `Ctrl+E` exportar, `Ctrl+F` buscar, `Ctrl+H` histórico,
  `Ctrl+L` log, `Espaço` / `←` / `→` player, `Esc` fecha gavetas.

## Instalação

1. Baixe `WhisperDesk-2.0.0-setup-x64.exe` na página de [Releases](../../releases).
2. Execute o instalador (instalação por usuário, não pede administrador).
3. Abra o WhisperDesk, solte um arquivo e clique em **Transcrever**. O modelo escolhido é baixado na primeira vez.

> O instalador ainda não é assinado digitalmente; o Windows SmartScreen pode exibir "O Windows protegeu o computador".
> Clique em **Mais informações → Executar assim mesmo**.

**Requisitos**

- Windows 10 1809+ ou Windows 11, 64 bits.
- Opcional: GPU NVIDIA com driver **525 ou mais novo** (CUDA 12). Sem GPU, o app usa a CPU.
- Espaço em disco: ~1,5 GB do app + o modelo escolhido (75 MB para o `tiny`, ~1,6 GB para o `large-v3-turbo`,
  ~3 GB para o `large-v3`).

## Modelos

| Modelo          | Download   | Uso sugerido                                                            |
|-----------------|------------|-------------------------------------------------------------------------|
| tiny / base     | 75–145 MB  | testes, máquinas fracas                                                 |
| small / medium  | 0,5–1,5 GB | boa qualidade em CPU                                                    |
| large-v3-turbo  | ~1,6 GB    | **padrão**: quase a qualidade do large-v3, bem mais rápido (não traduz) |
| large-v2 / v3   | ~3 GB      | máxima qualidade e tradução para inglês                                 |

Os modelos vêm do Hugging Face (formato CTranslate2); o download é retomado de onde parou.

## Onde ficam os dados

| O quê                                         | Caminho                                |
|-----------------------------------------------|----------------------------------------|
| Configurações                                 | `%APPDATA%\WhisperDesk\config.json`    |
| Modelos baixados                              | `%LOCALAPPDATA%\WhisperDesk\models`    |
| Trabalhos (retomáveis) e histórico            | `%LOCALAPPDATA%\WhisperDesk\jobs`      |
| Logs (`app.log`, `worker-*.log`, `crash.log`) | `%LOCALAPPDATA%\WhisperDesk\logs`      |

O menu **⋯ → Pasta de dados** abre esse diretório. A variável `WHISPERDESK_HOME` redireciona tudo para outra pasta
(modo portátil). O desinstalador pergunta se esses dados devem ser apagados.

## Diagnóstico

```bat
WhisperDesk.exe --selftest --out selftest.json
WhisperDesk.exe --selftest-transcribe arquivo.mp3 --model tiny --device cuda --out resultado.json
```

O primeiro verifica FFmpeg, CTranslate2, as DLLs CUDA, o VAD e a GPU. O segundo transcreve um arquivo sem interface
(gera `.json` e `.srt`). Ambos retornam código 0 em caso de sucesso.

**Problemas comuns**

- *"CUDA indisponível"*: atualize o driver NVIDIA (≥ 525). O app continua funcionando na CPU.
- *Falta de VRAM*: use `int8_float16` ou um modelo menor; o app já tenta isso sozinho e, em último caso, passa para a CPU.
- *Falha no download do modelo*: verifique proxy/firewall para `huggingface.co`.
- Em qualquer falha, o diálogo de erro mostra o caminho do log do trabalho.

## Desenvolvimento

Requer Python 3.12 (x64).

```bat
install.bat                      :: cria .venv e instala requirements-dev.txt
.venv\Scripts\python.exe main.py
.venv\Scripts\python.exe -m pytest -q tests
```

Para desenvolver, o FFmpeg pode estar no PATH ou em `.\bin\`.

## Build de distribuição

```bat
build.bat [--ffmpeg-dir C:\ffmpeg\bin] [--skip-tests]
```

Etapas: testes → `tools\prepare_build.py` (copia ffmpeg/ffprobe e cuBLAS para `build_cache\`, com hashes em
`MANIFEST.txt`) → PyInstaller (`WhisperDesk.spec`, onedir, sem UPX) → `--selftest` do executável → instalador
Inno Setup 6 (`installer\WhisperDesk.iss`, instalação por usuário sem UAC) se o `ISCC.exe` estiver instalado.

Saídas: `dist\WhisperDesk\` (pasta portátil, pode ser distribuída em ZIP) e
`dist\installer\WhisperDesk-2.0.0-setup-x64.exe`.

Recomendado antes de publicar: assinar `WhisperDesk.exe` e o instalador com um certificado de code signing
(`signtool sign /fd sha256 /tr http://timestamp.digicert.com /td sha256 ...`) para evitar o alerta do SmartScreen.

## Arquitetura

```
main.py                 entrada: GUI com instância única, --selftest, --selftest-transcribe
app/runtime.py          diretórios de DLL (CUDA/ffmpeg) para dev e para o executável congelado
app/paths.py            pastas de dados por usuário
app/pipeline/
  ffmpeg_tools.py       probe, extração de WAV 16 kHz, picos da forma de onda (subprocessos ffmpeg)
  planner.py            divide o áudio em blocos nos silêncios (Silero VAD)
  worker.py             processo de transcrição isolado (spawn), checkpoint por bloco, fallback GPU→CPU
  journal.py            diário SQLite (WAL) dos blocos — retomada após pausa ou queda
  hallucination.py      filtro de alucinações
  segmenter.py          re-segmentação para legendas
  exporters.py          TXT/SRT/VTT/JSON/CSV (escrita atômica)
  service.py            identidade de trabalhos, histórico, exportação
  models.py             catálogo e download de modelos
app/core/               player de áudio, monitor de GPU (NVML)
app/gui/
  theme.py              tokens de design (claro/escuro) e folha de estilo
  widgets.py            componentes animados (botões, seletores, toasts, gavetas, progresso)
  main_window.py        janela, layout, atalhos, arrastar e soltar
  result_view.py        lista virtualizada de segmentos, busca, karaokê, exportação
  player_controls.py    player com forma de onda
  history.py            histórico e recentes
  job_controller.py     ponte entre a UI e o processo worker
app/utils/              configuração, i18n (pt/en), forma de onda
tools/                  prepare_build.py, make_icon.py
packaging/              manifesto, version info, avisos de terceiros
installer/              script do Inno Setup
tests/                  pytest (pipeline, ffmpeg, serviço)
docs/                   auditoria de produção e arquitetura-alvo
```

Fluxo de um trabalho: o arquivo é inspecionado com `ffprobe` → o áudio é extraído para WAV 16 kHz mono →
o planner corta blocos nos silêncios → o worker (processo separado) transcreve bloco a bloco, gravando cada um no
diário → a UI recebe os segmentos em tempo real → ao fim, o resultado vai para o histórico e pode ser exportado.

## Roadmap

- [ ] Assinatura digital do executável e do instalador
- [ ] Diarização (quem falou o quê)
- [ ] Transcrição em tempo real do microfone
- [ ] Editor de texto do resultado com reexportação
- [ ] Fila de vários arquivos

## Licenças

O código do WhisperDesk é distribuído sob a [licença MIT](LICENSE).

Os componentes de terceiros incluídos no build mantêm as próprias licenças — veja
[packaging/THIRD_PARTY_NOTICES.txt](packaging/THIRD_PARTY_NOTICES.txt). O FFmpeg incluído é GPL v3 e roda como
processo separado; o Qt é LGPL v3 e fica em DLLs substituíveis.

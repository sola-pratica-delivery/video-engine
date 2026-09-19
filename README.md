# video-engine

Motor Audiovisual Headless (Remocao de Silencio VAD, Dynamic Zoom, BGM Ducking, Legendas Whisper, Thumbnails e Shorts 9:16)

## Issue #2 - Deteccao de pausas e hesitacoes com Silero VAD

Pacote `video_engine.audio` com deteccao de fala/pausas via **Silero VAD (ONNX)**:

```python
from video_engine.audio import SileroVadDetector, VadConfig

detector = SileroVadDetector()                      # usa models/silero_vad.onnx (vendored)
result = detector.detect_file("exemplo.wav")        # WAV/MP3/MP4 (soundfile + ffmpeg)

for fala in result.speech_segments:                 # fala com padding 80ms e fusao de sobreposicoes
    print(fala.start_sec, fala.end_sec, fala.confidence)
for pausa in result.silence_segments:               # pausas >= min_silence_duration_ms (padrao 500ms)
    print(pausa.start_sec, pausa.end_sec)
```

- `SileroVadDetector.detect(numpy_array, sample_rate=16000)` aceita arrays 1D/2D; reamostra para 16kHz e faz downmix de canais automaticamente.
- `VadConfig` configura `threshold`, `min_speech_duration_ms`, `min_silence_duration_ms`, `padding_ms` e `sample_rate`.
- Array vazio/NaN/Inf -> `ValueError`; arquivo inexistente -> `FileNotFoundError`.

Modelo ONNX oficial ([MIT](https://github.com/snakers4/silero-vad)) vendido em `models/`. Em instalacoes fora do checkout, o detector baixa/cacheia em `~/.cache/video_engine` validando SHA-256.

## Issue #7 - Transcricao fonetica com Faster-Whisper e timestamps por palavra

Pacote `video_engine.captions` com transcricao em portugues e alinhamento palavra-a-palavra em milissegundos:

```python
from video_engine.captions import TranscriberConfig, WhisperTranscriber

transcriber = WhisperTranscriber(TranscriberConfig(model_size="tiny", device="cpu"))
result = transcriber.transcribe_file("apresentacao.mp4")   # WAV/MP3/M4A/MP4/MKV

for word in result.words:              # lista plana cronologica
    print(word.word, word.start_ms, word.end_ms, word.probability)
for segment in result.segments:        # frases com suas palavras alinhadas
    print(segment.id, segment.text, segment.start_ms, segment.end_ms)
```

- `WhisperTranscriber.transcribe_file(path)` decodifica via `soundfile`/ffmpeg; videos/containers sem stream de audio levantam `ValueError`.
- `WhisperTranscriber.transcribe_array(array, sample_rate)` aceita arrays 1D/2D, faz downmix e reamostra para 16kHz float32.
- `TranscriberConfig` configura `model_size`, `device` (`auto`/`cpu`/`cuda`), `compute_type` (`int8` para CPU, `float16` para GPU), `beam_size`, `word_timestamps`, `vad_filter` e `initial_prompt`.
- Injeção de `model_instance` no construtor permite testes deterministicos sem download de pesos.
- Arquivo inexistente -> `FileNotFoundError`; array vazio/NaN/Inf -> `ValueError`; audio silencioso -> resultado valido com `text=""`, `words=[]` e `segments=[]`.

## Issue #8 - Dynamic Punch-in Zoom em transicoes e momentos de enfase

Pacote `video_engine.video` com alternancia harmonica de escala (100% vs 115%) e enquadramento facial:

```python
from video_engine.video import DynamicZoomConfig, DynamicZoomProcessor

processor = DynamicZoomProcessor(DynamicZoomConfig(zoom_scale=1.15, scaling_filter="lanczos"))
result = processor.apply_zoom(
    input_video="video_bruto.mp4",
    output_video="video_com_zoom.mp4",
    face_center=(0.5, 0.40),  # ancoragem no terco superior / face
)

for shot in result.shots:
    print(shot.mode, shot.start_ms, shot.end_ms, shot.scale)
```

- Alternancia automatica a cada 8 a 15 segundos ou sincronizada com pausas estruturais (`pause_intervals`).
- Enquadramento centralizado na face do apresentador com clamping de bordas.
- Sem degradacao perceptivel de resolucao atraves de interpolacao Lanczos ou Bicubic (`scale=W:H:flags=lanczos`).
- Preservacao direta do stream de audio (`-c:a copy`) sem re-encoding nem dessincronizacao labial.

## Issue #9 - Renderizacao e queima de legendas animadas em cortes (estilo karaoke)

Pacote `video_engine.captions` com geracao de legendas ASS e queima hardsub via FFmpeg exclusiva para **cortes/shorts** (o video longo permanece em clean feed):

```python
from video_engine.captions import (
    AssSubtitleGenerator,
    CaptionBurner,
    KaraokeHighlightMode,
    SubtitleStyleConfig,
)

# Estilo tipografico de alto impacto com area de seguranca para Shorts 9:16
style = SubtitleStyleConfig(
    font_name="Montserrat",
    font_size=52,
    highlight_color="#FFF200",  # Amarelo vibrante na palavra falada
    outline_width=3.5,
    margin_v=160,               # Safe area inferior para UI do Shorts/Reels
    highlight_mode=KaraokeHighlightMode.WORD_HIGHLIGHT,
)

burner = CaptionBurner(style_config=style)
result = burner.burn_cut_subtitles(
    cut_video_path="corte_vertical.mp4",
    output_path="corte_legendado.mp4",
    transcription=transcription_result,
    cut_start_ms=15000,  # Reindexa automaticamente os timestamps a partir de 0s
    cut_end_ms=45000,
)
```

- Destaque sincronizado palavra por palavra (`WORD_HIGHLIGHT` com cor vibrante e punch-in sutil, ou `KARAOKE_TAG` nativo `\k`).
- Reindexacao temporal automatica de cortes a partir da transcricao completa do video.
- Margens seguras evitando colisoes com icones e textos nativos de Shorts/Reels/TikTok.
- Preservacao integral do stream de audio via `-c:a copy`.
- O video longo processado pela esteira principal mantem-se como clean feed (sem hardsub).

### Testes

```bash
uv sync         # instala dependencias e dev-tools
uv run pytest   # 309 testes (unitarios + integracao em audio/video real)
uv run ruff check src tests
```
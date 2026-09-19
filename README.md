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

## Issue #19 - Decisao semantica de Dynamic Zoom com modelo do Google AI Studio

Pacote `video_engine.video` com alternancia de enquadramento orientada semanticamente pelo conteudo da fala:

```python
from video_engine.video import (
    DecisionMode,
    DynamicZoomConfig,
    DynamicZoomProcessor,
    GeminiZoomConfig,
)

# Configuracao com modelo gratuito do Google AI Studio
config = DynamicZoomConfig(
    decision_mode=DecisionMode.GEMINI,
    gemini=GeminiZoomConfig(
        model="gemini-2.5-flash",
        api_key="...",  # ou via variavel GEMINI_API_KEY
        timeout_s=10.0,
    ),
)

processor = DynamicZoomProcessor(config=config)
result = processor.apply_zoom(
    input_video="video.mp4",
    output_video="saida.mp4",
    transcription=transcription_result,  # TranscriptionResult do Whisper
)
```

- Analise semantica da fala via API do Google AI Studio com structured JSON schema (Pydantic).
- Identificacao inteligente de argumentos-chave, revelacoes, alertas e punchlines para aplicacao de punch-in zoom.
- Respeito estrito aos limites ergonomicos (`min_shot_duration_s` e `max_shot_duration_s`) e alinhamento com pausas de fala.
- Fallback automatico e transparente para a heuristica temporal (VAD/tempo) caso a chave GEMINI_API_KEY nao esteja configurada, ou em situacoes de timeout, rate limit (HTTP 429) ou payload invalido.
- Suporte a injecao de cliente HTTP para execucao 100% deterministica e offline em suites de teste.

## Issue #11 - Algoritmo de pontuacao e selecao automatica do keyframe mais expressivo

Pacote `video_engine.thumbnail` com selecao inteligente dos melhores frames para thumbnails de alto CTR:

```python
from video_engine.thumbnail import KeyframeSelector, KeyframeSelectorConfig

selector = KeyframeSelector(
    KeyframeSelectorConfig(
        min_sharpness_threshold=80.0,
        discard_blurry=True,
        discard_closed_eyes=True,
        top_n=5,
        min_candidate_distance_ms=1500,
    )
)

result = selector.select_best_keyframes(
    video_path="apresentacao.mp4",
    output_dir="storage/thumbnails",  # exporta frames selecionados como PNG
)

for candidate in result.top_candidates:
    print(f"Rank {candidate.rank}: {candidate.timestamp_ms}ms (Score: {candidate.score:.2f})")
    print(f"  Nitidez: {candidate.metrics.sharpness_variance:.1f} | Iluminacao: {candidate.metrics.lighting_score:.2f}")
    print(f"  Expressividade: {candidate.metrics.face.expression_intensity:.2f} | Imagem: {candidate.image_path}")
```

- Inspecao automatica de frames analisando nitidez (variancia do Laplaciano), iluminacao e expressividade facial (abertura ocular e articulacao labial).
- **Descarte rigoroso** de frames com motion blur (abaixo do limiar de nitidez) e piscadas (olhos fechados).
- **Ranqueamento automatico dos 5 melhores frames candidatos** ordenados por score composto ponderado.
- **Filtro de diversidade temporal** (`min_candidate_distance_ms`) garantindo que os frames selecionados nao sejam de instantes quase consecutivos.
- Decodificacao direta em memoria e suporte a gravacao automatica em disco via PyAV.

## Issue #12 - Segmentacao e remocao de fundo da face/locutor

Pacote `video_engine.thumbnail` com isolamento semantico do apresentador e realce de borda (stroke e glow suave):

```python
from video_engine.thumbnail import (
    GlowConfig,
    OnnxBackgroundSegmenter,
    SegmenterConfig,
    StrokeConfig,
)

segmenter = OnnxBackgroundSegmenter(
    config=SegmenterConfig(
        model_path="models/rmbg.onnx",  # compativel com RMBG-1.4, BiRefNet, U2Net
        feather_radius=2,               # suavizacao de borda / anti-aliasing
        threshold=0.5,
    )
)

# Remocao de fundo e aplicacao de contorno/glow para alto CTR
result = segmenter.remove_background(
    input_image="storage/thumbnails/keyframe_0002500ms.png",
    output_path="storage/thumbnails/apresentador_recortado.png",
    stroke=StrokeConfig(enabled=True, width=8, color=(255, 255, 255)),  # contorno branco
    glow=GlowConfig(enabled=True, radius=18, color=(255, 215, 0), intensity=0.85),  # glow dourado
)

print(f"Sujeito extraido: {result.width}x{result.height} (RGBA com canal alfa)")
```

- Segmentacao semantica de alta resolucao com geracao de mascara alfa suave (`alpha_mask`) e imagem `foreground_rgba`.
- **Eliminacao de serrilhados e halos de recorte** via suavizacao de borda (*feathering / antialiasing*).
- **Contorno customizavel (Stroke)** com espessura em pixels, cor RGB e opacidade.
- **Brilho difuso suave (Glow)** periférico com difusao gaussiana progressiva ao redor da silhueta do apresentador.
- Preservacao integral dos pixels do sujeito, sem oclusao ou escurecimento facial.
- Suporte a modelos ONNX e injecao de sessao para execucao 100% offline em testes.

## Issue #13 - Composicao visual com background tematico, headline contrastante e export 1280x720

Pacote `video_engine.thumbnail` com sintese visual final da capa para o YouTube:

```python
from video_engine.thumbnail import (
    BackgroundConfig,
    BackgroundType,
    GlowConfig,
    HeadlineConfig,
    StrokeConfig,
    SubjectPosition,
    ThumbnailComposer,
    ThumbnailConfig,
)

composer = ThumbnailComposer(
    ThumbnailConfig(
        width=1280,
        height=720,
        subject_position=SubjectPosition.RIGHT,  # sujeito a direita, texto a esquerda
        subject_scale=0.88,
        jpeg_quality=90,
    )
)

result = composer.compose(
    subject="storage/thumbnails/apresentador_recortado.png",
    headline=HeadlineConfig(
        text="VOCE VAI ACREDITAR",  # 3 a 5 palavras de alto impacto para mobile
        font_size=72,
        text_color=(255, 242, 0),   # amarelo vibrante
        stroke_color=(0, 0, 0),     # contorno preto ultra-espesso
        stroke_width=6,
    ),
    background=BackgroundConfig(
        type=BackgroundType.GRADIENT,
        color_start=(15, 23, 42),   # dark navy #0F172A
        color_end=(30, 41, 59),     # slate navy #1E293B
        direction="diagonal",
    ),
    stroke=StrokeConfig(enabled=True, width=8, color=(255, 255, 255)),
    glow=GlowConfig(enabled=True, radius=18, color=(255, 215, 0)),
    output_path="storage/thumbnails/capa_final_1280x720.jpg",
)

print(f"Capa gerada: {result.width}x{result.height} ({result.file_size_bytes} bytes < 2MB)")
```

- Composicao multicamada 16:9 em canvas 1280x720 (Background -> Sujeito com realce -> Headline).
- **Layout balanceado na regra dos terços** com sujeito na lateral e headline tipográfica no terço oposto.
- **Protecao incondicional da Safe Area do YouTube** evitando oclusao no quadrante inferior direito ($x > 1050, y > 600$, reservado ao timestamp de duracao).
- **Limitador de densidade lexical mobile** para 3 a 5 palavras de alto impacto.
- **Exportacao JPEG adaptativa** garantindo deterministamente arquivo estritamente inferior a 2MB.

### Testes

```bash
uv sync         # instala dependencias e dev-tools
uv run pytest   # 425 testes (unitarios + integracao em audio/video real)
uv run ruff check src tests
```
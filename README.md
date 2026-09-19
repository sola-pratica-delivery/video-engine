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

### Testes

```bash
uv sync         # instala dependencias e dev-tools
uv run pytest   # 46 testes (unitarios + integracao em audio real)
uv run ruff check src tests
```
"""Motor de transcricao fonetica com Faster-Whisper e alinhamento por palavra.

Implementa ``WhisperTranscriber`` (Issue #7), que transcreve audio em
portugues a partir de arquivos (WAV/MP3/MP4/MKV via ``soundfile``/ffmpeg)
ou de arrays NumPy em memoria, gerando ``TranscriptionResult`` com
timestamps por palavra em milissegundos.

O carregamento do modelo e feito de forma preguiçosa (lazy) e aceita
injeção de ``model_instance`` para testes deterministicos sem download de
pesos na suite rapida.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np

from video_engine.captions.models import (
    CaptionSegment,
    TranscriberConfig,
    TranscriptionResult,
    WordTimestamp,
)
from video_engine.editing.media_probe import MediaProbe

TARGET_SAMPLE_RATE = 16000

_VIDEO_EXTENSIONS = {".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".ts", ".webm"}


def _lowpass(sequence: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    """Filtro passa-baixa simples (FFT com transicao suave) antes da decimacao."""
    n = len(sequence)
    if n < 64:
        return sequence
    nfft = 1 << (n - 1).bit_length()
    spectrum = np.fft.rfft(sequence, nfft)

    stop = int(np.round((cutoff_hz * 1.3) / (sample_rate / 2) * (nfft // 2)))
    stop = min(stop, len(spectrum) - 1)
    start = int(np.round(cutoff_hz / (sample_rate / 2) * (nfft // 2)))

    if stop > start:
        spectrum[start:stop] *= np.linspace(1.0, 0.0, stop - start, dtype=np.float32)
        spectrum[stop:] = 0.0
    else:
        spectrum[start:] = 0.0

    return np.fft.irfft(spectrum, nfft)[:n].astype(np.float32)


def _resample_audio(
    audio: np.ndarray,
    origin_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Reamostra um sinal 1D por interpolacao linear, com anti-alias na queda."""
    if origin_rate == target_rate:
        return audio
    n = len(audio)
    if n == 0:
        return audio
    if target_rate < origin_rate:
        audio = _lowpass(audio, cutoff_hz=target_rate * 0.45, sample_rate=origin_rate)
    out_length = max(1, int(round(n * target_rate / origin_rate)))
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, out_length, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _read_audio_file(file_path: Path, target_rate: int) -> Tuple[np.ndarray, int]:
    """Le um arquivo de audio em mono float32 mantendo a taxa original.

    Tenta ``soundfile`` primeiro (WAV, MP3, FLAC, OGG...). Se o formato nao
    for suportado (ex.: MP4/MKV), recorre ao ``ffmpeg`` para decodificar.
    """
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependencia opcional
        raise ImportError(
            "soundfile is required for file decoding. Install with 'pip install soundfile'."
        ) from exc

    try:
        data, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
        return mono.astype(np.float32), sample_rate
    except Exception as sf_error:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                f"Nao foi possivel decodificar '{file_path}' (soundfile: {sf_error}). "
                "Instale/conecte o ffmpeg no PATH para formatos nao suportados."
            ) from sf_error
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "decoded.wav"
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(file_path),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    str(target_rate),
                    "-f",
                    "wav",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg falhou ao decodificar '{file_path}': {result.stderr.strip()}"
                ) from sf_error
            data, sample_rate = sf.read(output, dtype="float32", always_2d=True)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return mono.astype(np.float32), sample_rate


def _to_ms(seconds: Optional[float]) -> Optional[int]:
    """Converte segundos (float) em milissegundos inteiros."""
    if seconds is None:
        return None
    try:
        return int(round(float(seconds) * 1000))
    except (TypeError, ValueError):
        return None


def _normalize_text(text: str) -> str:
    """Normaliza espacos duplicados e bordas do texto de um segmento."""
    return " ".join(text.split())


def _clamp_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(probability, 0.0), 1.0)


def _clean_word_text(raw: Any) -> str:
    """Extrai o texto de um objeto de palavra do modelo, removendo espacos."""
    text = getattr(raw, "word", None)
    if text is None:
        text = getattr(raw, "text", None)
    if text is None:
        return ""
    return " ".join(str(text).split())


class WhisperTranscriber:
    """Motor de transcricao fonetica com Faster-Whisper e alinhamento por palavra.

    Args:
        config: Configuracao do Faster-Whisper. Usa ``TranscriberConfig()``
            (padroes) quando omitido.
        model_instance: Instancia de modelo pronta para ``.transcribe(...)``
            (tipicamente um ``WhisperModel``). Quando omitida, o modelo e
            carregado de forma preguiçosa conforme ``config``.
    """

    def __init__(
        self,
        config: Optional[TranscriberConfig] = None,
        model_instance: Optional[Any] = None,
    ) -> None:
        self.config = config or TranscriberConfig()
        self._model = model_instance

    # -- infraestrutura do modelo -------------------------------------------

    def _build_model(self, device: Optional[str] = None) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependencia opcional
            raise ImportError(
                "faster-whisper is required for WhisperTranscriber. "
                "Install with 'pip install faster-whisper'."
            ) from exc
        target_device = device or self.config.device
        if target_device == "auto":
            import os
            target_device = os.getenv("WHISPER_DEVICE", "auto")
        model_path_or_size = self.config.model_path or self.config.model_size
        return WhisperModel(
            model_path_or_size,
            device=target_device,
            compute_type=self.config.compute_type,
            download_root=self.config.download_root,
        )

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    # -- API publica ----------------------------------------------------------

    def transcribe_file(self, media_path: Union[str, Path]) -> TranscriptionResult:
        """Transcreve arquivo de video ou audio e retorna ``TranscriptionResult``.

        Raises:
            FileNotFoundError: Se ``media_path`` nao existir.
            ValueError: Se um video/container nao possuir stream de audio.
            RuntimeError: Se o arquivo nao puder ser decodificado.
        """
        path = Path(media_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de midia nao encontrado: {path}")

        if path.suffix.lower() in _VIDEO_EXTENSIONS:
            info = MediaProbe().probe(path)
            if not info.has_audio:
                raise ValueError("Arquivo de midia sem stream de audio")

        audio, sample_rate = _read_audio_file(path, TARGET_SAMPLE_RATE)
        return self.transcribe_array(audio, sample_rate=sample_rate)

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """Transcreve array NumPy de audio (reamostrando para 16kHz mono se necessario).

        Raises:
            ValueError: Se ``sample_rate`` for invalido ou o array estiver
                vazio, contiver NaN/Inf ou tiver mais de 2 dimensoes.
        """
        if sample_rate <= 0:
            raise ValueError(
                f"sample_rate deve ser um inteiro positivo (recebido: {sample_rate})"
            )

        arr = np.asarray(audio)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = arr.mean(axis=1)
        elif arr.ndim == 0:
            raise ValueError("Audio array cannot be empty or invalid")
        elif arr.ndim > 2:
            raise ValueError(
                f"Audio array deve ter 1 ou 2 dimensoes (recebido: {arr.ndim})"
            )
        arr = arr.astype(np.float32)

        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("Audio array cannot be empty or invalid")

        duration_ms = int(round(arr.size / sample_rate * 1000))
        if sample_rate != TARGET_SAMPLE_RATE:
            arr = _resample_audio(arr, sample_rate, TARGET_SAMPLE_RATE)

        def _run_transcribe(active_model: Any) -> TranscriptionResult:
            segments, info = active_model.transcribe(
                arr,
                language=self.config.language,
                beam_size=self.config.beam_size,
                word_timestamps=self.config.word_timestamps,
                vad_filter=self.config.vad_filter,
                initial_prompt=self.config.initial_prompt,
            )
            return self._to_result(segments, info, duration_ms)

        current_model = self._get_model()
        try:
            return _run_transcribe(current_model)
        except Exception as exc:
            msg = str(exc).lower()
            cuda_err = any(k in msg for k in ("cublas", "cudnn", "cuda", "curand", "cusolver"))
            if cuda_err and getattr(self.config, "device", "auto") != "cpu":
                import logging
                logging.getLogger(__name__).warning(
                    "CUDA/cuBLAS indisponivel no runtime (%s); executando fallback automatico para CPU.",
                    exc,
                )
                self._model = self._build_model(device="cpu")
                return _run_transcribe(self._model)
            raise

    # -- pos-processamento -----------------------------------------------------

    def _to_result(
        self,
        segments: Iterable[Any],
        info: Any,
        duration_ms: int,
    ) -> TranscriptionResult:
        caption_segments: List[CaptionSegment] = []
        flat_words: List[WordTimestamp] = []

        for index, segment in enumerate(segments):
            text = _normalize_text(getattr(segment, "text", "") or "")
            start_ms = _to_ms(getattr(segment, "start", None)) or 0
            end_ms = _to_ms(getattr(segment, "end", None))
            end_ms = max(start_ms, end_ms if end_ms is not None else start_ms)

            words: List[WordTimestamp] = []
            if self.config.word_timestamps:
                words = self._extract_words(segment)

            caption_segments.append(
                CaptionSegment(
                    id=index,
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    words=words,
                )
            )
            flat_words.extend(words)

        language = getattr(info, "language", None) or self.config.language
        language_probability = _clamp_probability(
            getattr(info, "language_probability", 1.0)
        )
        full_text = "\n".join(seg.text for seg in caption_segments if seg.text)

        return TranscriptionResult(
            text=full_text,
            language=str(language),
            language_probability=language_probability,
            duration_ms=duration_ms,
            segments=caption_segments,
            words=flat_words,
        )

    @staticmethod
    def _extract_words(segment: Any) -> List[WordTimestamp]:
        words: List[WordTimestamp] = []
        raw_words = list(getattr(segment, "words", None) or ())

        for raw in raw_words:
            word_text = _clean_word_text(raw)
            if not word_text:
                continue

            start_ms = _to_ms(getattr(raw, "start", None))
            if start_ms is None:
                continue
            start_ms = max(start_ms, 0)

            end_ms = _to_ms(getattr(raw, "end", None))
            if end_ms is None or end_ms < start_ms:
                end_ms = start_ms

            words.append(
                WordTimestamp(
                    word=word_text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    probability=_clamp_probability(getattr(raw, "probability", 0.0)),
                )
            )
        return words

    def __repr__(self) -> str:
        return (
            f"<WhisperTranscriber model=({self.config.model_size!r} "
            f"device={self.config.device!r} compute_type={self.config.compute_type!r})>"
        )


__all__ = ["WhisperTranscriber", "TARGET_SAMPLE_RATE"]

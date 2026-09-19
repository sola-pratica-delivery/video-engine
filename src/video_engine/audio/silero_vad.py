"""Detector Silero VAD baseado em ONNXRuntime.

Implementa ``SileroVadDetector``, que processa audio mono em 16kHz (ou
reamostra outras taxas), infere probabilidades de fala por janela de 32ms
com a rede Silero VAD (ONNX) e produz um ``VadResult`` com segmentos de
fala (padding/fusao aplicados) e pausas elegiveis para corte.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import onnxruntime
except ImportError as exc:  # pragma: no cover - dependencia opcional no carregamento
    raise ImportError(
        "onnxruntime is required for SileroVadDetector. Install with 'pip install onnxruntime'."
    ) from exc

try:
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - dependencia opcional no carregamento
    raise ImportError(
        "soundfile is required for file decoding. Install with 'pip install soundfile'."
    ) from exc

from .models import TimeInterval, VadConfig, VadResult
from .vad_utils import apply_padding_and_merge, extract_silence_intervals

MODEL_FILENAME = "silero_vad.onnx"
MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

_WINDOW_MS = 32


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _default_cache_dir() -> Path:
    cache = Path.home() / ".cache" / "video_engine"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _repo_models_dir() -> Optional[Path]:
    """Retorna o diretorio ``models`` do repositorio quando executado em checkout."""
    root = Path(__file__).resolve().parents[3]
    models = root / "models"
    return models if models.is_dir() else None


def _download_model(destination: Path) -> Path:
    """Baixa o modelo ONNX oficial e valida sua chave SHA-256."""
    temporary = destination.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Falha ao baixar o modelo Silero VAD de {MODEL_URL}: {exc}"
        ) from exc
    if _sha256_file(temporary) != MODEL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Modelo Silero VAD baixado nao corresponde ao SHA-256 esperado. "
            "Verifique a rede ou atualize MODEL_SHA256."
        )
    temporary.replace(destination)
    return destination


def _resolve_model_path(onnx_model_path: Optional[Union[str, Path]]) -> Path:
    """Resolve o caminho do modelo ONNX (explicito, local ou cache/download)."""
    if onnx_model_path is not None:
        path = Path(onnx_model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de modelo ONNX nao encontrado: {path}")
        return path

    local = _repo_models_dir()
    if local is not None:
        candidate = local / MODEL_FILENAME
        if candidate.is_file():
            return candidate

    cached = _default_cache_dir() / MODEL_FILENAME
    if cached.is_file() and _sha256_file(cached) == MODEL_SHA256:
        return cached
    return _download_model(cached)


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


def _bound_confidence(probs: np.ndarray, start_ms: int, end_ms: int) -> Optional[float]:
    """Confianca media das janelas VAD que intersectam [start_ms, end_ms)."""
    if start_ms >= end_ms or probs.size == 0:
        return None
    first = start_ms // _WINDOW_MS
    last = min(int(math.ceil(end_ms / _WINDOW_MS)), probs.size)
    if first >= probs.size or last <= first:
        return None
    return float(np.mean(probs[first:last]))


def _read_audio_file(file_path: Path, target_rate: int) -> Tuple[np.ndarray, int]:
    """Le/decodifica um arquivo de audio em mono float32 mantendo a taxa original.

    Tenta ``soundfile`` primeiro (WAV, MP3, FLAC, OGG...). Se o formato nao
    for suportado (ex.: MP4), recorre ao ``ffmpeg`` para decodificar.
    """
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
                    f"ffmpeg falhou ao decodificar '{file_path}': "
                    f"{result.stderr.strip()}"
                ) from sf_error
            data, sample_rate = sf.read(output, dtype="float32", always_2d=True)
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            return mono.astype(np.float32), sample_rate


class SileroVadDetector:
    """Detector de fala/pausas usando Silero VAD (ONNXRuntime).

    Args:
        config: Configuracao do VAD e pos-processamento. Usa ``VadConfig()``
            (padroes) quando omitido.
        onnx_model_path: Caminho explicito para o modelo ONNX. Quando omitido
            o detector procura em ``models/`` do repositorio e, na ausencia,
            baixa/cacheia o modelo oficial em ``~/.cache/video_engine``.

    Raises:
        FileNotFoundError: Se ``onnx_model_path`` nao existir.
        RuntimeError: Se o modelo padrao nao puder ser obtido/verificado.
    """

    def __init__(
        self,
        config: Optional[VadConfig] = None,
        onnx_model_path: Optional[Union[str, Path]] = None,
    ):
        self.config = config or VadConfig()
        self.model_path = _resolve_model_path(onnx_model_path)
        self.session = onnxruntime.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=_session_options(),
        )
        self._frame_params()
        self.reset_states()

    def _frame_params(self) -> None:
        """Parametros de enquadramento do modelo conforme a taxa de amostragem."""
        if self.config.sample_rate == 16000:
            self._window_samples = 512
            self._context_samples = 64
        elif self.config.sample_rate == 8000:
            self._window_samples = 256
            self._context_samples = 32
        else:
            raise ValueError(
                "O modelo Silero VAD suporta apenas 8000 ou 16000 Hz de "
                f"amostragem (recebido: {self.config.sample_rate})."
            )

    def reset_states(self) -> None:
        """Reinicia o estado recursivo e o contexto de amostras do modelo."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._context_samples), dtype=np.float32)

    def _infer_probs(self, audio_16k: np.ndarray) -> np.ndarray:
        """Executa o modelo por janelas de 32ms e retorna as probabilidades."""
        self.reset_states()
        n = len(audio_16k)
        if n == 0:
            return np.empty(0, dtype=np.float32)

        n_chunks = int(math.ceil(n / self._window_samples))
        padded = np.pad(audio_16k, (0, n_chunks * self._window_samples - n))
        probs = np.empty(n_chunks, dtype=np.float32)
        sample_rate = np.array(self.config.sample_rate, dtype=np.int64)

        for index in range(n_chunks):
            window = padded[index * self._window_samples : (index + 1) * self._window_samples]
            chunk = np.concatenate([self._context, window.reshape(1, -1)], axis=1)
            outputs = self.session.run(
                None,
                {
                    "input": chunk,
                    "state": self._state,
                    "sr": sample_rate,
                },
            )
            speech_prob, state_n = outputs
            self._state = np.asarray(state_n, dtype=np.float32)
            self._context = chunk[:, -self._context_samples:].copy()
            probs[index] = float(np.asarray(speech_prob).reshape(-1)[0])

        return probs

    def _probs_to_raw_segments(
        self,
        probs: np.ndarray,
        total_duration_ms: int,
    ) -> List[TimeInterval]:
        """Converte probabilidades por janela em intervalos crus de fala (ms).

        Utiliza histerese (``neg_threshold = threshold - 0.15``), filtra
        falas menores que ``min_speech_duration_ms`` e encerra cada trecho
        somente apos um silencio sustentado de ``min_silence_duration_ms``.
        """
        threshold = self.config.threshold
        neg_threshold = max(threshold - 0.15, 0.01)
        min_speech = self.config.min_speech_duration_ms
        min_silence = self.config.min_silence_duration_ms

        segments: List[TimeInterval] = []
        triggered = False
        start_ms = 0
        temp_end_ms: Optional[int] = None

        for index, probability in enumerate(probs):
            current_ms = index * _WINDOW_MS

            if not triggered:
                if probability >= threshold:
                    triggered = True
                    start_ms = current_ms
                continue

            if probability >= threshold:
                temp_end_ms = None
                continue

            if probability < neg_threshold:
                if temp_end_ms is None:
                    temp_end_ms = current_ms
                if current_ms - temp_end_ms >= min_silence:
                    if temp_end_ms - start_ms >= min_speech:
                        segments.append(
                            TimeInterval(start_ms=start_ms, end_ms=temp_end_ms)
                        )
                    triggered = False
                    temp_end_ms = None

        if triggered and total_duration_ms - start_ms >= min_speech:
            segments.append(TimeInterval(start_ms=start_ms, end_ms=total_duration_ms))
        return segments

    def detect(self, audio: np.ndarray, sample_rate: int = 16000) -> VadResult:
        """Processa array mono de float32 e retorna ``VadResult``.

        Args:
            audio: Array 1D (mono) ou 2D ``(frames, canais)`` com amostras
                arbitrarias (convertidas para float32).
            sample_rate: Taxa de amostragem de ``audio``.

        Raises:
            ValueError: Se o array estiver vazio, contiver NaN/Inf ou se a
                taxa de amostragem for invalida.
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

        total_duration_ms = int(round(arr.size / sample_rate * 1000))
        if sample_rate != self.config.sample_rate:
            arr = _resample_audio(arr, sample_rate, self.config.sample_rate)

        probs = self._infer_probs(arr)
        raw_segments = self._probs_to_raw_segments(probs, total_duration_ms)
        speech_segments = apply_padding_and_merge(
            raw_segments,
            padding_ms=self.config.padding_ms,
            total_duration_ms=total_duration_ms,
        )

        for segment in speech_segments:
            segment.confidence = _bound_confidence(
                probs, segment.start_ms, segment.end_ms
            )

        silence_segments = extract_silence_intervals(
            speech_segments,
            total_duration_ms=total_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
        )

        return VadResult(
            total_duration_ms=total_duration_ms,
            speech_segments=speech_segments,
            silence_segments=silence_segments,
        )

    def detect_file(self, file_path: Union[str, Path]) -> VadResult:
        """Carrega audio de arquivo (WAV/MP3/MP4 via soundfile/ffmpeg) e detecta.

        Raises:
            FileNotFoundError: Se o arquivo nao existir.
            RuntimeError: Se o arquivo nao puder ser decodificado por nenhum
                mecanismo disponivel.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de audio nao encontrado: {path}")
        mono, sample_rate = _read_audio_file(path, self.config.sample_rate)
        return self.detect(mono, sample_rate=sample_rate)


def _session_options() -> "onnxruntime.SessionOptions":
    options = onnxruntime.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    return options


__all__ = [
    "SileroVadDetector",
    "MODEL_FILENAME",
    "MODEL_URL",
    "MODEL_SHA256",
    "_WINDOW_MS",
]

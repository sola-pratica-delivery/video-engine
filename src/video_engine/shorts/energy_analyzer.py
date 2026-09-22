"""Analisador de energia vocal (RMS) para a deteccao de ganchos (Issue #15).

Calcula a intensidade acustica (RMS) por janelas temporais a partir de um
arquivo de audio (via ``soundfile``) ou de um array NumPy, identifica picos
de energia vocal e normaliza o score para o intervalo [0.0, 1.0].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from video_engine.shorts.models import EnergyAnalysisResult, EnergyPeak

DEFAULT_WINDOW_MS = 250


class EnergyAnalysisError(Exception):
    """Falha ao carregar ou analisar a fonte de audio."""


def _resample(data: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Reamostragem linear deterministica de um array mono (float32)."""
    if src_rate == dst_rate or data.size == 0:
        return data
    n_out = int(round(data.size * dst_rate / src_rate))
    if n_out <= 0:
        raise EnergyAnalysisError("Falha no calculo do numero de amostras na reamostragem.")
    x_old = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


class EnergyAnalyzer:
    """Calcula metricas de intensidade vocal / RMS sobre audio em janelas temporais."""

    def __init__(self, window_ms: int = DEFAULT_WINDOW_MS) -> None:
        if window_ms < 1:
            raise ValueError("window_ms deve ser >= 1")
        self.window_ms = int(window_ms)
        self._profile: Optional[EnergyAnalysisResult] = None

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def analyze_audio(
        self,
        audio_source: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
    ) -> EnergyAnalysisResult:
        """Analisa a fonte de audio e devolve o perfil de energia RMS normalizado.

        ``audio_source`` pode ser um caminho de arquivo de audio suportado pelo
        ``soundfile`` (WAV/FLAC/OGG) ou um array NumPy float.

        Raises:
            FileNotFoundError: arquivo de audio inexistente.
            EnergyAnalysisError: array/arquivo vazio, invalido ou nao decodificavel.
        """
        samples, sr = self._load(audio_source, sample_rate)
        profile = self._compute(samples, sr)
        self._profile = profile
        return profile

    def compute_window_energy(self, start_ms: int, end_ms: int) -> float:
        """Energia media (0.0 a 1.0) na janela temporal ``[start_ms, end_ms)``."""
        profile = self._require_profile()
        return profile.window_energy(start_ms, end_ms)

    # ------------------------------------------------------------------ #
    # Infraestrutura
    # ------------------------------------------------------------------ #
    @classmethod
    def _load(cls, audio_source: Union[str, Path, np.ndarray], sample_rate: int) -> Tuple[np.ndarray, int]:
        if isinstance(audio_source, np.ndarray):
            if audio_source.ndim == 0 or audio_source.size == 0:
                raise EnergyAnalysisError("Array de audio vazio.")
            data = audio_source.astype(np.float32, copy=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            return np.ascontiguousarray(data), int(sample_rate)

        path = Path(audio_source)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo de audio nao encontrado: {path}")
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
        except Exception as exc:
            raise EnergyAnalysisError(f"Falha ao decodificar audio de {path}: {exc}") from exc
        if data.ndim > 1:
            data = data.mean(axis=1)
        if data.size == 0:
            raise EnergyAnalysisError(f"Audio vazio: {path}")
        if sr != int(sample_rate):
            data = _resample(data, int(sr), int(sample_rate))
        return np.ascontiguousarray(data), int(sample_rate)

    def _compute(self, samples: np.ndarray, sample_rate: int) -> EnergyAnalysisResult:
        win = int(round(sample_rate * self.window_ms / 1000.0))
        win = max(win, 1)
        n_windows = max(1, math.ceil(samples.size / win))

        energies: List[float] = []
        for w in range(n_windows):
            chunk = samples[w * win : (w + 1) * win]
            if chunk.size == 0:
                energies.append(0.0)
                continue
            energies.append(float(np.sqrt(np.mean(np.square(chunk)))))

        peak_rms = max(energies) if energies else 0.0
        normalized = [e / peak_rms if peak_rms > 0.0 else 0.0 for e in energies]
        mean_energy = float(np.mean(normalized)) if normalized else 0.0

        duration_ms = int(round(samples.size * 1000 / sample_rate))
        peaks, peak_ms = self._detect_peaks(normalized, duration_ms, mean_energy)

        return EnergyAnalysisResult(
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            window_ms=self.window_ms,
            energies=normalized,
            peaks=peaks,
            mean_energy=round(mean_energy, 6),
            peak_energy=round(max(normalized) if normalized else 0.0, 6),
            peak_ms=peak_ms,
        )

    def _detect_peaks(
        self,
        normalized: List[float],
        duration_ms: int,
        mean_energy: float,
    ) -> Tuple[List[EnergyPeak], Optional[int]]:
        step = self.window_ms
        threshold = max(0.30, mean_energy)
        peak_energy = max(normalized) if normalized else 0.0
        peaks: List[EnergyPeak] = []
        strongest_idx: Optional[int] = None
        strongest_val = -1.0

        for i, val in enumerate(normalized):
            if val > strongest_val:
                strongest_val = val
                strongest_idx = i
            left = normalized[i - 1] if i > 0 else 0.0
            right = normalized[i + 1] if i < len(normalized) - 1 else 0.0
            is_local_max = val >= left and val > right
            is_global_max = val >= peak_energy and peak_energy > 0.0
            if not is_local_max and not is_global_max:
                continue
            if not is_global_max and val < threshold:
                continue
            start_ms = i * step
            end_ms = min(duration_ms, (i + 1) * step)
            peaks.append(
                EnergyPeak(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    peak_ms=(start_ms + end_ms) // 2,
                    intensity=round(val, 6),
                )
            )

        peaks.sort(key=lambda p: p.intensity, reverse=True)
        peak_ms = None
        if strongest_idx is not None:
            start_ms = strongest_idx * step
            end_ms = min(duration_ms, (strongest_idx + 1) * step)
            peak_ms = (start_ms + end_ms) // 2
        return peaks, peak_ms

    def _require_profile(self) -> EnergyAnalysisResult:
        if self._profile is None:
            raise RuntimeError(
                "analyze_audio() deve ser chamado antes de compute_window_energy()."
            )
        return self._profile


__all__ = ["DEFAULT_WINDOW_MS", "EnergyAnalysisError", "EnergyAnalyzer"]

"""Mixagem de trilha ambiente (BGM) com sidechain ducking e loop continuo.

Implementa a Spec da Issue #5 (SDD):
- Insercao de trilha sonora ambiente de fundo (BGM).
- Sidechain ducking com atenuacao suave de -18dB a -22dB durante a fala.
- Fade-up suave nos momentos de respiro, pausas ou introducao.
- Loop continuo com crossfades suaves para cobrir videos de qualquer duracao.
- Direct stream copy (-c:v copy) para preservacao de video sem re-encode.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf

from video_engine.audio.bgm_models import BgmDuckingConfig, BgmDuckingResult
from video_engine.editing.media_probe import MediaProbe


def _fade_weights(fade_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gera curvas equal-power de fade-in e fade-out (seno/cosseno)."""
    if fade_len <= 1:
        return np.ones(max(1, fade_len), dtype=np.float32), np.zeros(max(1, fade_len), dtype=np.float32)
    t = np.linspace(0.0, 1.0, fade_len, dtype=np.float64)
    fade_in = np.sin(np.pi / 2.0 * t).astype(np.float32)
    fade_out = np.cos(np.pi / 2.0 * t).astype(np.float32)
    return fade_in, fade_out


def _loop_audio_array(
    audio: np.ndarray,
    sample_rate: int,
    target_duration_ms: int,
    crossfade_ms: float,
) -> Tuple[np.ndarray, int]:
    """Estende o array de audio em loop continuo com crossfades equal-power.

    Retorna o array concatenado e a quantidade de loops aplicados.
    """
    n_samples = audio.shape[0]
    duration_ms = (n_samples * 1000.0) / sample_rate
    if duration_ms >= target_duration_ms:
        return audio, 0

    fade_samples = min(int(round(crossfade_ms * sample_rate / 1000.0)), n_samples // 2)
    fade_samples = max(1, fade_samples)
    step = n_samples - fade_samples

    target_samples = int(math.ceil(target_duration_ms * sample_rate / 1000.0))
    loops_needed = int(math.ceil((target_samples - n_samples) / step))
    loops_needed = max(1, loops_needed)

    total_repetitions = loops_needed + 1
    total_len = (total_repetitions - 1) * step + n_samples

    fade_in, fade_out = _fade_weights(fade_samples)
    is_2d = audio.ndim == 2
    if is_2d:
        fade_in = fade_in.reshape(-1, 1)
        fade_out = fade_out.reshape(-1, 1)

    out = np.zeros((total_len, audio.shape[1]) if is_2d else (total_len,), dtype=np.float32)

    for i in range(total_repetitions):
        offset = i * step
        if i == 0:
            out[:step] = audio[:step]
            out[step:n_samples] = audio[step:] * fade_out
        elif i == total_repetitions - 1:
            out[offset : offset + fade_samples] += audio[:fade_samples] * fade_in
            out[offset + fade_samples : offset + n_samples] = audio[fade_samples:]
        else:
            out[offset : offset + fade_samples] += audio[:fade_samples] * fade_in
            out[offset + fade_samples : offset + step] = audio[fade_samples:step]
            out[offset + step : offset + n_samples] = audio[step:] * fade_out

    return out, loops_needed


class BgmDucker:
    """Motor de mixagem de BGM com sidechain ducking e loop continuo."""

    def __init__(
        self,
        config: Optional[BgmDuckingConfig] = None,
        ffmpeg_path: Optional[str] = None,
        ffprobe_path: Optional[str] = None,
    ) -> None:
        self.config = config or BgmDuckingConfig()
        self.ffmpeg_path = ffmpeg_path
        self.ffprobe_path = ffprobe_path
        self.probe_service = MediaProbe(ffprobe_path=ffprobe_path)

    def _resolve_ffmpeg(self) -> str:
        binary = self.ffmpeg_path or shutil.which("ffmpeg")
        if not binary:
            raise RuntimeError("ffmpeg nao encontrado no PATH")
        return binary

    def prepare_bgm_stream(
        self,
        bgm_path: Union[str, Path],
        target_duration_ms: int,
        temp_dir: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, int]:
        """Prepara o arquivo de BGM em loop continuo com transicoes invisiveis."""
        bgm = Path(bgm_path)
        if not bgm.is_file():
            raise FileNotFoundError(f"Arquivo BGM nao encontrado: {bgm}")

        info = self.probe_service.probe(bgm)
        if not info.has_audio:
            raise ValueError(f"Arquivo BGM sem stream de audio: {bgm}")

        if info.duration_ms >= target_duration_ms:
            return bgm, 0

        # Carregar via soundfile
        try:
            audio_data, sr = sf.read(str(bgm), dtype="float32")
        except Exception:
            # Fallback para ffmpeg se soundfile nao decodificar diretamente
            ffmpeg = self._resolve_ffmpeg()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_pcm:
                pcm_path = Path(tmp_pcm.name)
            try:
                cmd = [ffmpeg, "-y", "-i", str(bgm), "-vn", "-c:a", "pcm_s16le", str(pcm_path)]
                subprocess.run(cmd, check=True, capture_output=True)
                audio_data, sr = sf.read(str(pcm_path), dtype="float32")
            finally:
                if pcm_path.exists():
                    pcm_path.unlink()

        looped_audio, loops = _loop_audio_array(
            audio_data,
            sample_rate=sr,
            target_duration_ms=target_duration_ms,
            crossfade_ms=self.config.loop_crossfade_ms,
        )

        target_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        target_dir.mkdir(parents=True, exist_ok=True)
        out_bgm = target_dir / f"bgm_looped_{target_duration_ms}ms_{loops}loops.wav"
        sf.write(str(out_bgm), looped_audio, sr)
        return out_bgm, loops

    def build_filter_complex(
        self,
        has_video: bool,
        target_duration_ms: int,
    ) -> str:
        """Monta o grafo de filtros FFmpeg para sidechain compression e mixagem."""
        dur_sec = target_duration_ms / 1000.0
        cfg = self.config

        fade_in_sec = cfg.fade_in_ms / 1000.0
        fade_out_sec = cfg.fade_out_ms / 1000.0
        fade_out_start = max(0.0, dur_sec - fade_out_sec)

        bgm_filters = [
            "aformat=channel_layouts=stereo:sample_rates=44100",
            f"volume={cfg.bgm_volume_db}dB",
        ]
        if cfg.fade_in_ms > 0:
            bgm_filters.append(f"afade=t=in:ss=0:d={fade_in_sec:.3f}")
        if cfg.fade_out_ms > 0 and fade_out_start < dur_sec:
            bgm_filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out_sec:.3f}")
        bgm_filters.append(f"atrim=0:{dur_sec:.3f}")
        bgm_filters.append("asetpts=PTS-STARTPTS")

        bgm_filter_str = ",".join(bgm_filters)

        filter_graph = (
            f"[0:a]aformat=channel_layouts=stereo:sample_rates=44100[voice];"
            f"[1:a]{bgm_filter_str}[bgm_prep];"
            f"[bgm_prep][voice]sidechaincompress=threshold={cfg.threshold}:ratio={cfg.ratio}:"
            f"attack={cfg.attack_ms}:release={cfg.release_ms}:knee={cfg.knee}[ducked_bgm];"
            f"[voice][ducked_bgm]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
        return filter_graph

    def mix(
        self,
        primary_media: Union[str, Path],
        bgm_media: Union[str, Path],
        output_path: Union[str, Path],
    ) -> BgmDuckingResult:
        """Aplica sidechain ducking da BGM disparado pela voz e gera a midia mixada."""
        primary = Path(primary_media)
        bgm = Path(bgm_media)
        out = Path(output_path)

        if not primary.is_file():
            raise FileNotFoundError(f"Arquivo primario nao encontrado: {primary}")
        if not bgm.is_file():
            raise FileNotFoundError(f"Arquivo BGM nao encontrado: {bgm}")

        primary_info = self.probe_service.probe(primary)
        if not primary_info.has_audio:
            raise ValueError(f"Arquivo primario sem stream de audio: {primary}")

        bgm_info = self.probe_service.probe(bgm)
        if not bgm_info.has_audio:
            raise ValueError(f"Arquivo BGM sem stream de audio: {bgm}")

        out.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = self._resolve_ffmpeg()

        with tempfile.TemporaryDirectory(prefix="video_engine_bgm_") as tmp_dir:
            prepared_bgm, loops_applied = self.prepare_bgm_stream(
                bgm_path=bgm,
                target_duration_ms=primary_info.duration_ms,
                temp_dir=tmp_dir,
            )

            filter_complex = self.build_filter_complex(
                has_video=primary_info.has_video,
                target_duration_ms=primary_info.duration_ms,
            )

            cmd = [
                ffmpeg,
                "-y",
                "-i",
                str(primary),
                "-i",
                str(prepared_bgm),
                "-filter_complex",
                filter_complex,
            ]

            if primary_info.has_video:
                cmd.extend(["-map", "0:v", "-c:v", "copy"])

            cmd.extend(["-map", "[aout]"])

            out_suffix = out.suffix.lower()
            if out_suffix in (".wav",):
                cmd.extend(["-c:a", "pcm_s16le"])
            else:
                codec = self.config.audio_codec or "aac"
                cmd.extend(["-c:a", codec, "-b:a", self.config.audio_bitrate])

            cmd.append(str(out))

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg falhou ao mixar BGM com ducking ({proc.returncode}):\n{proc.stderr}"
                )

        return BgmDuckingResult(
            output_path=str(out),
            primary_duration_ms=primary_info.duration_ms,
            bgm_original_duration_ms=bgm_info.duration_ms,
            loops_applied=loops_applied,
            has_video=primary_info.has_video,
            effective_attenuation_db=self.config.ducking_attenuation_db,
        )


__all__ = ["BgmDucker", "_fade_weights", "_loop_audio_array"]

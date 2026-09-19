"""Descoberta de streams de midia via ffprobe.

Extrai presenca de stream de video/audio, fps, taxa de amostragem e duracoes
(necessarios para o ``MediaSplicer`` montar o grafo ``-filter_complex``).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class MediaInfo(BaseModel):
    """Descricao dos streams de um arquivo de midia."""

    path: str
    has_video: bool
    has_audio: bool
    video_fps: Optional[float] = None
    video_width: Optional[int] = None
    video_height: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    duration_ms: int = Field(ge=0)
    video_duration_ms: Optional[int] = None
    audio_duration_ms: Optional[int] = None


def _parse_fps(raw: Optional[str]) -> Optional[float]:
    if not raw or "/" not in raw:
        return None
    num, _, den = raw.partition("/")
    try:
        value = float(num) / float(den)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if value <= 0:
        return None
    return round(value, 6)


def _parse_duration_ms(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    return int(round(seconds * 1000))


def _parse_int(raw: Any) -> Optional[int]:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


class MediaProbe:
    """Inspeciona um arquivo de midia e descreve seus streams."""

    def __init__(self, ffprobe_path: Optional[str] = None) -> None:
        self.ffprobe_path = ffprobe_path

    def _resolve_ffprobe(self) -> str:
        binary = self.ffprobe_path or shutil.which("ffprobe")
        if not binary:
            raise RuntimeError("ffprobe nao encontrado no PATH")
        return binary

    def probe(self, path) -> MediaInfo:
        """Analisa ``path`` e retorna um :class:`MediaInfo`.

        Raises:
            FileNotFoundError: se ``path`` nao existir.
            RuntimeError: se o ffprobe falhar ou a saida for invalida.
        """
        media_path = Path(path)
        if not media_path.is_file():
            raise FileNotFoundError(f"Arquivo de midia nao encontrado: {media_path}")
        binary = self._resolve_ffprobe()
        cmd = [
            binary,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            str(media_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as exc:
            raise RuntimeError(f"Nao foi possivel executar ffprobe ({binary}): {exc}") from exc
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffprobe falhou ao inspecionar {media_path}: {proc.stderr.strip()}"
            )
        try:
            payload: Dict[str, Any] = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Saida invalida do ffprobe para {media_path}: {exc}") from exc
        return self._to_media_info(media_path, payload)

    @classmethod
    def _to_media_info(cls, media_path: Path, payload: Dict[str, Any]) -> MediaInfo:
        streams = payload.get("streams") or []
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        attached_pic = False
        if video_stream is not None:
            disposition = video_stream.get("disposition") or {}
            attached_pic = disposition.get("attached_pic") == 1
        effective_video = video_stream if video_stream is not None and not attached_pic else None

        fps = None
        if effective_video:
            fps = _parse_fps(effective_video.get("r_frame_rate"))
            if fps is None:
                fps = _parse_fps(effective_video.get("avg_frame_rate"))

        video_dur = (
            _parse_duration_ms(effective_video.get("duration")) if effective_video else None
        )
        audio_dur = _parse_duration_ms(audio_stream.get("duration")) if audio_stream else None

        format_info = payload.get("format") or {}
        duration_ms = _parse_duration_ms(format_info.get("duration"))
        if duration_ms is None:
            candidates = [d for d in (video_dur, audio_dur) if d is not None]
            duration_ms = max(candidates) if candidates else 0

        return MediaInfo(
            path=str(media_path),
            has_video=effective_video is not None,
            has_audio=audio_stream is not None,
            video_fps=fps,
            video_width=(
                _parse_int(effective_video.get("width")) if effective_video else None
            ),
            video_height=(
                _parse_int(effective_video.get("height")) if effective_video else None
            ),
            audio_sample_rate=_parse_int(audio_stream.get("sample_rate")) if audio_stream else None,
            audio_channels=_parse_int(audio_stream.get("channels")) if audio_stream else None,
            duration_ms=duration_ms,
            video_duration_ms=video_dur,
            audio_duration_ms=audio_dur,
        )


def probe_media(path) -> MediaInfo:
    """Atalho conveniente para :meth:`MediaProbe.probe`."""
    return MediaProbe().probe(path)


__all__ = ["MediaInfo", "MediaProbe", "probe_media"]

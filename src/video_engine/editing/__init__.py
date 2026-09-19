"""Camada de edicao audiovisual (Issue #3).

Splicing cirurgico de midia com micro-crossfades (15ms default) e
sincronizacao A/V estrita frame-a-frame.
"""

from __future__ import annotations

from video_engine.editing.media_probe import MediaInfo, MediaProbe, probe_media
from video_engine.editing.media_splicer import MediaSplicer
from video_engine.editing.models import FadeCurve, SplicerConfig, SpliceResult

__all__ = [
    "FadeCurve",
    "MediaInfo",
    "MediaProbe",
    "MediaSplicer",
    "SpliceResult",
    "SplicerConfig",
    "probe_media",
]

"""Testes do modulo de Dynamic Punch-in Zoom (Issue #8).

Cobre:
- Modelos de dados e validacoes de configuracao (DynamicZoomConfig, ZoomShot, DynamicZoomResult)
- Algoritmo de planejamento e agendamento de planos (plan_zoom_shots):
  - Alternancia harmonica entre 8 e 15 segundos (CA1)
  - Alinhamento de cortes com pausas estruturais (CA1)
  - Respeito aos guardrails de duracao minima e maxima
  - Videos curtos (< 8s) com plano unico sem corte
  - Enquadramento e ancoragem customizada de face (CA2)
- Construcao do grafo de filtros FFmpeg (crop dinamico + interpolacao Lanczos/Bicubic) (CA3)
- Tratamento de casos de borda e erros (arquivos ausentes, midia sem video)
- Integracao real com FFmpeg:
  - Preservacao estrita de resolucao original (W x H)
  - Preservacao labial e stream de audio direto (-c:a copy) (CA4)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_engine.audio.models import TimeInterval
from video_engine.editing.media_probe import MediaProbe
from video_engine.video.models import (
    DynamicZoomConfig,
    DynamicZoomResult,
    ZoomMode,
    ZoomShot,
)
from video_engine.video.zoom import DynamicZoomProcessor

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _create_test_video(path: Path, duration_sec: float = 20.0, width: int = 320, height: int = 240) -> None:
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size={width}x{height}:rate=24",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={duration_sec}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


# --------------------------------------------------------------------------- #
# 1. Modelos e Validacoes
# --------------------------------------------------------------------------- #
def test_dynamic_zoom_config_defaults():
    cfg = DynamicZoomConfig()
    assert cfg.zoom_scale == 1.15
    assert cfg.min_shot_duration_s == 8.0
    assert cfg.max_shot_duration_s == 15.0
    assert cfg.target_shot_duration_s == 10.0
    assert cfg.anchor_x == 0.5
    assert cfg.anchor_y == 0.40
    assert cfg.scaling_filter == "lanczos"
    assert cfg.video_codec == "libx264"
    assert cfg.crf == 18
    assert cfg.preset == "veryfast"


def test_dynamic_zoom_config_validations():
    with pytest.raises(Exception):
        DynamicZoomConfig(zoom_scale=1.0)  # deve ser >= 1.05
    with pytest.raises(Exception):
        DynamicZoomConfig(zoom_scale=2.5)  # deve ser <= 1.50
    with pytest.raises(Exception):
        DynamicZoomConfig(min_shot_duration_s=16.0, max_shot_duration_s=10.0)  # min > max
    with pytest.raises(Exception):
        DynamicZoomConfig(min_shot_duration_s=8.0, max_shot_duration_s=12.0, target_shot_duration_s=15.0)


def test_zoom_shot_model():
    shot = ZoomShot(
        start_ms=0,
        end_ms=10000,
        mode=ZoomMode.NORMAL,
        scale=1.0,
        anchor_x=0.5,
        anchor_y=0.40,
    )
    assert shot.duration_ms == 10000
    assert shot.mode == ZoomMode.NORMAL
    assert shot.scale == 1.0


def test_dynamic_zoom_result_model():
    res = DynamicZoomResult(
        output_path="out.mp4",
        total_duration_ms=20000,
        shots=[
            ZoomShot(start_ms=0, end_ms=10000, mode=ZoomMode.NORMAL, scale=1.0),
            ZoomShot(start_ms=10000, end_ms=20000, mode=ZoomMode.ZOOM, scale=1.15),
        ],
        zoom_shots_count=1,
        normal_shots_count=1,
        video_width=1920,
        video_height=1080,
        scaling_filter="lanczos",
    )
    assert res.total_duration_ms == 20000
    assert len(res.shots) == 2
    assert res.zoom_shots_count == 1
    assert res.normal_shots_count == 1


# --------------------------------------------------------------------------- #
# 2. Algoritmo de Agendamento (plan_zoom_shots) - CA1 e CA2
# --------------------------------------------------------------------------- #
def test_plan_zoom_shots_short_video():
    """Video menor que a duracao minima (ex: 5s) permanece em plano normal sem corte."""
    processor = DynamicZoomProcessor()
    shots = processor.plan_zoom_shots(total_duration_ms=5000)
    assert len(shots) == 1
    assert shots[0].mode == ZoomMode.NORMAL
    assert shots[0].start_ms == 0
    assert shots[0].end_ms == 5000
    assert shots[0].scale == 1.0


def test_plan_zoom_shots_without_pauses_alternates_harmonically():
    """Video de 30s sem pausas deve alternar a cada ~10s (entre 8s e 15s)."""
    cfg = DynamicZoomConfig(target_shot_duration_s=10.0, min_shot_duration_s=8.0, max_shot_duration_s=15.0)
    processor = DynamicZoomProcessor(config=cfg)
    shots = processor.plan_zoom_shots(total_duration_ms=30000)

    # 3 planos de 10s: Normal -> Zoom -> Normal
    assert len(shots) == 3
    assert shots[0].mode == ZoomMode.NORMAL
    assert shots[0].start_ms == 0
    assert shots[0].end_ms == 10000

    assert shots[1].mode == ZoomMode.ZOOM
    assert shots[1].start_ms == 10000
    assert shots[1].end_ms == 20000
    assert shots[1].scale == 1.15

    assert shots[2].mode == ZoomMode.NORMAL
    assert shots[2].start_ms == 20000
    assert shots[2].end_ms == 30000

    # Todos os planos respeitam a janela de 8s a 15s
    for shot in shots:
        dur_s = shot.duration_ms / 1000.0
        assert 8.0 <= dur_s <= 15.0


def test_plan_zoom_shots_synchronizes_with_structural_pauses():
    """Cortes devem sincronizar com pausas quando disponiveis na janela [8s, 15s]."""
    cfg = DynamicZoomConfig(min_shot_duration_s=8.0, max_shot_duration_s=15.0, target_shot_duration_s=10.0)
    processor = DynamicZoomProcessor(config=cfg)

    # Pausa aos 9.5s (9500ms a 10000ms) e aos 18.2s (18200ms a 18700ms)
    pauses = [
        TimeInterval(start_ms=9500, end_ms=10000),
        TimeInterval(start_ms=18200, end_ms=18700),
    ]
    shots = processor.plan_zoom_shots(total_duration_ms=28000, pause_intervals=pauses)

    assert len(shots) == 3
    # O primeiro corte deve ocorrer na primeira pausa (~9750ms ou 10000ms)
    assert 9500 <= shots[0].end_ms <= 10000
    assert shots[1].start_ms == shots[0].end_ms
    assert shots[1].mode == ZoomMode.ZOOM

    # O segundo corte deve ocorrer na segunda pausa
    assert 18200 <= shots[1].end_ms <= 18700
    assert shots[2].start_ms == shots[1].end_ms
    assert shots[2].mode == ZoomMode.NORMAL


def test_plan_zoom_shots_ignores_frequent_pauses_before_min_duration():
    """Pausas antes de min_shot_duration_s (ex: aos 2s e 4s) devem ser ignoradas."""
    cfg = DynamicZoomConfig(min_shot_duration_s=8.0, max_shot_duration_s=15.0, target_shot_duration_s=10.0)
    processor = DynamicZoomProcessor(config=cfg)
    pauses = [
        TimeInterval(start_ms=2000, end_ms=2500),
        TimeInterval(start_ms=4000, end_ms=4500),
        TimeInterval(start_ms=9000, end_ms=9500),
    ]
    shots = processor.plan_zoom_shots(total_duration_ms=20000, pause_intervals=pauses)
    # Primeiro plano nao pode ter durado 2s ou 4s
    assert shots[0].duration_ms >= 8000
    # Cortou na pausa de 9.0s
    assert 9000 <= shots[0].end_ms <= 9500


def test_plan_zoom_shots_forces_cut_on_max_duration():
    """Se nao houver pausas, o corte e forcado antes de exceder max_shot_duration_s."""
    cfg = DynamicZoomConfig(min_shot_duration_s=8.0, max_shot_duration_s=12.0, target_shot_duration_s=10.0)
    processor = DynamicZoomProcessor(config=cfg)
    shots = processor.plan_zoom_shots(total_duration_ms=25000)
    for shot in shots[:-1]:
        assert shot.duration_ms <= 12000


def test_plan_zoom_shots_custom_face_anchor():
    """Valida ancoragem customizada de face transmitida para os planos de zoom."""
    processor = DynamicZoomProcessor()
    custom_face = (0.65, 0.35)
    shots = processor.plan_zoom_shots(total_duration_ms=20000, anchor_override=custom_face)
    for shot in shots:
        if shot.mode == ZoomMode.ZOOM:
            assert shot.anchor_x == 0.65
            assert shot.anchor_y == 0.35


# --------------------------------------------------------------------------- #
# 3. Construcao de Grafo de Filtros (CA3)
# --------------------------------------------------------------------------- #
def test_build_filter_complex_lanczos_and_crop():
    cfg = DynamicZoomConfig(scaling_filter="lanczos", zoom_scale=1.15)
    processor = DynamicZoomProcessor(config=cfg)
    shots = [
        ZoomShot(start_ms=0, end_ms=10000, mode=ZoomMode.NORMAL, scale=1.0),
        ZoomShot(start_ms=10000, end_ms=20000, mode=ZoomMode.ZOOM, scale=1.15, anchor_x=0.5, anchor_y=0.4),
    ]
    filter_graph = processor.build_filter_complex(shots, width=1920, height=1080)
    assert "crop=" in filter_graph
    assert "scale=1920:1080:flags=lanczos" in filter_graph
    assert "between(t" in filter_graph


def test_build_filter_complex_bicubic():
    cfg = DynamicZoomConfig(scaling_filter="bicubic", zoom_scale=1.20)
    processor = DynamicZoomProcessor(config=cfg)
    shots = [
        ZoomShot(start_ms=0, end_ms=10000, mode=ZoomMode.ZOOM, scale=1.20),
    ]
    filter_graph = processor.build_filter_complex(shots, width=1280, height=720)
    assert "scale=1280:720:flags=bicubic" in filter_graph


# --------------------------------------------------------------------------- #
# 4. Tratamento de Erros e Casos de Borda
# --------------------------------------------------------------------------- #
def test_apply_zoom_file_not_found(tmp_path):
    processor = DynamicZoomProcessor()
    with pytest.raises(FileNotFoundError, match="Arquivo de video nao encontrado"):
        processor.apply_zoom(tmp_path / "nao_existe.mp4", tmp_path / "out.mp4")


def test_apply_zoom_audio_only_raises_value_error(tmp_path):
    audio_only = tmp_path / "audio.wav"
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=2",
        str(audio_only),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    processor = DynamicZoomProcessor()
    with pytest.raises(ValueError, match="Arquivo de midia sem stream de video"):
        processor.apply_zoom(audio_only, tmp_path / "out.mp4")


# --------------------------------------------------------------------------- #
# 5. Integracao Real com FFmpeg (CA1, CA2, CA3, CA4)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="FFmpeg ou FFprobe nao disponivel")
def test_apply_zoom_integration_preserves_audio_and_resolution(tmp_path):
    """Valida pipeline completa de Dynamic Zoom: resolucao exata, audio preservado e alternancia."""
    input_video = tmp_path / "input_test.mp4"
    out_video = tmp_path / "output_zoom.mp4"
    _create_test_video(input_video, duration_sec=20.0, width=320, height=240)

    cfg = DynamicZoomConfig(
        zoom_scale=1.15,
        min_shot_duration_s=8.0,
        max_shot_duration_s=12.0,
        target_shot_duration_s=10.0,
        scaling_filter="lanczos",
    )
    processor = DynamicZoomProcessor(config=cfg)
    result = processor.apply_zoom(
        input_video=input_video,
        output_video=out_video,
        face_center=(0.5, 0.40),
    )

    assert Path(result.output_path).is_file()
    assert result.zoom_shots_count >= 1
    assert result.normal_shots_count >= 1
    assert result.video_width == 320
    assert result.video_height == 240
    assert result.scaling_filter == "lanczos"

    # Inspecao com MediaProbe
    probe_in = MediaProbe().probe(input_video)
    probe_out = MediaProbe().probe(out_video)

    assert probe_out.has_video is True
    assert probe_out.has_audio is True
    assert probe_out.video_width == 320
    assert probe_out.video_height == 240
    assert abs(probe_out.duration_ms - probe_in.duration_ms) <= 100

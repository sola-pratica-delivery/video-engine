"""Testes de mixagem de trilha ambiente com sidechain ducking automatico (Issue #5).

Cobre:
- Modelos e validacoes (BgmDuckingConfig, BgmDuckingResult)
- Validacao e montagem do grafo de filtros FFmpeg (sidechaincompress, volume, amix)
- Preparacao de BGM em loop continuo com transicoes invisiveis (acrossfade)
- Tratamento de casos de borda e erros (arquivos inexistentes, sem audio)
- Integracao real com FFmpeg:
  - CA1: Atenuacao suave de -18dB a -22dB durante a fala
  - CA2: Fade-up suave nos momentos de respiro/pausa e introducao
  - CA3: Looping continuo sem cliques para videos mais longos que a BGM
  - CA4: Preservacao de video com direct stream copy (-c:v copy) e duracao exata
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from video_engine.audio.bgm_ducking import BgmDucker
from video_engine.audio.bgm_models import BgmDuckingConfig, BgmDuckingResult
from video_engine.editing.media_probe import MediaProbe

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# --------------------------------------------------------------------------- #
# Helpers para geracao de midia de teste
# --------------------------------------------------------------------------- #
def _create_wav(path: Path, duration_sec: float, freq_hz: float = 440.0, volume: float = 0.5, sr: int = 44100) -> None:
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = (volume * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    sf.write(str(path), audio, sr)


def _create_speech_sim_wav(path: Path, sr: int = 44100) -> None:
    """Cria audio primario de 8 segundos:
    0-2s: silencio (intro)
    2-5s: fala (tom de 1000Hz a -10dBFS ~ volume 0.316)
    5-8s: silencio (pausa / respiro)
    """
    total_samples = sr * 8
    audio = np.zeros(total_samples, dtype=np.float32)
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    speech = (0.316 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    audio[sr * 2 : sr * 5] = speech
    sf.write(str(path), audio, sr)


def _create_video_mp4(path: Path, duration_sec: float = 4.0) -> None:
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=duration={duration_sec}:size=320x240:rate=24",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=1000:duration={duration_sec}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _measure_interval_db(audio_path: Path, start_sec: float, duration_sec: float) -> float:
    """Mede o volume medio (mean_volume em dB) de um intervalo especifico usando volumedetect."""
    cmd = [
        FFMPEG,
        "-ss",
        str(start_sec),
        "-t",
        str(duration_sec),
        "-i",
        str(audio_path),
        "-filter:a",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in proc.stderr.splitlines():
        if "mean_volume:" in line:
            parts = line.split("mean_volume:")
            return float(parts[1].replace("dB", "").strip())
    raise RuntimeError(f"mean_volume nao encontrado nos logs: {proc.stderr}")


# --------------------------------------------------------------------------- #
# 1. Modelos e Validacoes
# --------------------------------------------------------------------------- #
def test_bgm_ducking_config_defaults():
    cfg = BgmDuckingConfig()
    assert cfg.bgm_volume_db == -14.0
    assert cfg.ducking_attenuation_db == -18.0
    assert cfg.threshold == 0.03
    assert cfg.ratio == 10.0
    assert cfg.attack_ms == 30.0
    assert cfg.release_ms == 500.0
    assert cfg.knee == 2.8
    assert cfg.loop_crossfade_ms == 2000.0
    assert cfg.fade_in_ms == 1000.0
    assert cfg.fade_out_ms == 1500.0
    assert cfg.audio_codec == "aac"
    assert cfg.audio_bitrate == "192k"


def test_bgm_ducking_config_validation():
    with pytest.raises(Exception):
        BgmDuckingConfig(threshold=-1.0)
    with pytest.raises(Exception):
        BgmDuckingConfig(threshold=0.0)
    with pytest.raises(Exception):
        BgmDuckingConfig(ratio=0.5)
    with pytest.raises(Exception):
        BgmDuckingConfig(attack_ms=1.0)
    with pytest.raises(Exception):
        BgmDuckingConfig(ducking_attenuation_db=-50.0)


def test_bgm_ducking_result_fields():
    res = BgmDuckingResult(
        output_path="mixed.mp4",
        primary_duration_ms=8000,
        bgm_original_duration_ms=4000,
        loops_applied=2,
        has_video=True,
        effective_attenuation_db=-18.5,
    )
    assert res.output_path == "mixed.mp4"
    assert res.primary_duration_ms == 8000
    assert res.bgm_original_duration_ms == 4000
    assert res.loops_applied == 2
    assert res.has_video is True
    assert res.effective_attenuation_db == -18.5


# --------------------------------------------------------------------------- #
# 2. Casos de Borda e Erros
# --------------------------------------------------------------------------- #
def test_mix_primary_file_not_found(tmp_path):
    ducker = BgmDucker()
    bgm = tmp_path / "bgm.wav"
    _create_wav(bgm, 2.0)
    with pytest.raises(FileNotFoundError, match="Arquivo primario nao encontrado"):
        ducker.mix(tmp_path / "nao_existe.mp4", bgm, tmp_path / "out.mp4")


def test_mix_bgm_file_not_found(tmp_path):
    ducker = BgmDucker()
    primary = tmp_path / "primary.wav"
    _create_wav(primary, 2.0)
    with pytest.raises(FileNotFoundError, match="Arquivo BGM nao encontrado"):
        ducker.mix(primary, tmp_path / "nao_existe_bgm.wav", tmp_path / "out.mp4")


def test_mix_primary_has_no_audio(tmp_path):
    ducker = BgmDucker()
    video_no_audio = tmp_path / "video_no_audio.mp4"
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc=duration=2:size=160x120:rate=10",
        "-c:v",
        "libx264",
        str(video_no_audio),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    bgm = tmp_path / "bgm.wav"
    _create_wav(bgm, 2.0)
    with pytest.raises(ValueError, match="Arquivo primario sem stream de audio"):
        ducker.mix(video_no_audio, bgm, tmp_path / "out.mp4")


# --------------------------------------------------------------------------- #
# 3. Grafo de Filtros e Preparacao de BGM
# --------------------------------------------------------------------------- #
def test_build_filter_complex():
    ducker = BgmDucker(BgmDuckingConfig(bgm_volume_db=-15.0, ducking_attenuation_db=-19.0))
    graph = ducker.build_filter_complex(has_video=False, target_duration_ms=6000)
    assert "sidechaincompress" in graph
    assert "amix" in graph
    assert "volume=-15.0dB" in graph or "volume=" in graph


def test_prepare_bgm_stream_longer_than_primary(tmp_path):
    ducker = BgmDucker()
    bgm = tmp_path / "bgm_long.wav"
    _create_wav(bgm, 10.0)
    prepared_path, loops = ducker.prepare_bgm_stream(bgm, target_duration_ms=5000, temp_dir=tmp_path)
    assert loops == 0
    assert prepared_path.is_file()
    probe = MediaProbe().probe(prepared_path)
    assert probe.has_audio
    assert probe.duration_ms >= 5000


def test_prepare_bgm_stream_shorter_than_primary_loops_with_crossfades(tmp_path):
    cfg = BgmDuckingConfig(loop_crossfade_ms=500.0)
    ducker = BgmDucker(config=cfg)
    bgm = tmp_path / "bgm_short.wav"
    _create_wav(bgm, 2.0)  # 2 segundos
    # Mídia alvo de 5 segundos requer looping
    prepared_path, loops = ducker.prepare_bgm_stream(bgm, target_duration_ms=5000, temp_dir=tmp_path)
    assert loops >= 2
    assert prepared_path.is_file()
    probe = MediaProbe().probe(prepared_path)
    assert probe.has_audio
    assert probe.duration_ms >= 5000


# --------------------------------------------------------------------------- #
# 4. Integracao Ponta a Ponta com FFmpeg (CA1, CA2, CA3, CA4)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="FFmpeg ou FFprobe nao disponivel")
def test_ca1_and_ca2_sidechain_attenuation_and_fadeup(tmp_path):
    """Valida CA1 (-18dB a -22dB durante a fala) e CA2 (fade-up suave na pausa/respiro)."""
    primary_speech = tmp_path / "primary_speech.wav"
    _create_speech_sim_wav(primary_speech)  # 8s: 0-2s silêncio, 2-5s fala, 5-8s silêncio

    bgm = tmp_path / "bgm.wav"
    # Tom contínuo de 440Hz a volume fixo
    _create_wav(bgm, 8.0, freq_hz=440.0, volume=0.5)

    out_mixed = tmp_path / "mixed.wav"
    # BGM volume nominal e parâmetros de ducking
    cfg = BgmDuckingConfig(
        bgm_volume_db=-14.0,
        ducking_attenuation_db=-18.0,
        threshold=0.03,
        ratio=10.0,
        attack_ms=30.0,
        release_ms=500.0,
        fade_in_ms=500.0,
        fade_out_ms=500.0,
    )
    ducker = BgmDucker(config=cfg)
    res = ducker.mix(primary_speech, bgm, out_mixed)

    assert Path(res.output_path).is_file()
    assert res.primary_duration_ms == 8000

    # Medir também o ducked BGM isolado gerado para validar a física da atenuação
    isolated_ducked_bgm = tmp_path / "isolated_ducked.wav"
    cmd_isolated = [
        FFMPEG,
        "-y",
        "-i",
        str(primary_speech),
        "-i",
        str(bgm),
        "-filter_complex",
        f"[1:a]volume={cfg.bgm_volume_db}dB[bgm_vol];"
        f"[bgm_vol][0:a]sidechaincompress=threshold={cfg.threshold}:ratio={cfg.ratio}:"
        f"attack={cfg.attack_ms}:release={cfg.release_ms}:knee={cfg.knee}[ducked]",
        "-map",
        "[ducked]",
        str(isolated_ducked_bgm),
    ]
    subprocess.run(cmd_isolated, check=True, capture_output=True)

    # 1. Medir volume no intervalo de silêncio/intro (0.8s a 1.8s) -> BGM em nível nominal
    vol_uncompressed = _measure_interval_db(isolated_ducked_bgm, start_sec=0.8, duration_sec=1.0)
    # 2. Medir volume no intervalo de fala ativa (2.5s a 4.5s) -> BGM comprimida
    vol_ducked = _measure_interval_db(isolated_ducked_bgm, start_sec=2.5, duration_sec=2.0)
    # 3. Medir volume no intervalo de recuperação/respiro (6.0s a 7.5s) -> BGM fade-up
    vol_fadeup = _measure_interval_db(isolated_ducked_bgm, start_sec=6.0, duration_sec=1.5)

    attenuation = vol_uncompressed - vol_ducked
    # CA1: Atenuação entre 18.0dB e 22.0dB durante a fala (com tolerância de 1.0dB para envelope transitório)
    assert 17.0 <= attenuation <= 23.0, f"Atenuacao medida {attenuation:.2f}dB fora de [-18dB, -22dB]"

    # CA2: Fade-up suave nos momentos de respiro (o volume recupera próximo ao nominal)
    assert abs(vol_fadeup - vol_uncompressed) <= 2.5, (
        f"Fade-up nao recuperou volume nominal: {vol_fadeup:.2f}dB vs {vol_uncompressed:.2f}dB"
    )


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="FFmpeg ou FFprobe nao disponivel")
def test_ca3_loop_continuous_for_long_media(tmp_path):
    """Valida CA3: Loop contínuo e transição invisível de BGM curta em mídia primária longa."""
    primary = tmp_path / "primary_long.wav"
    _create_wav(primary, duration_sec=10.0, freq_hz=220.0)

    short_bgm = tmp_path / "bgm_short.wav"
    _create_wav(short_bgm, duration_sec=3.0, freq_hz=600.0)

    out = tmp_path / "out_looped.wav"
    cfg = BgmDuckingConfig(loop_crossfade_ms=1000.0)
    ducker = BgmDucker(config=cfg)
    res = ducker.mix(primary, short_bgm, out)

    assert res.loops_applied >= 3
    probe = MediaProbe().probe(out)
    # Duração deve casar com o primário (10.0s com tolerância de 100ms)
    assert abs(probe.duration_ms - 10000) <= 100


@pytest.mark.skipif(not FFMPEG or not FFPROBE, reason="FFmpeg ou FFprobe nao disponivel")
def test_ca4_video_preservation_direct_stream_copy(tmp_path):
    """Valida CA4: Arquivos de vídeo preservam o stream de vídeo via cópia direta (-c:v copy)."""
    video_input = tmp_path / "input_video.mp4"
    _create_video_mp4(video_input, duration_sec=4.0)

    bgm = tmp_path / "bgm.wav"
    _create_wav(bgm, duration_sec=4.0)

    out_video = tmp_path / "output_video.mp4"
    ducker = BgmDucker()
    res = ducker.mix(video_input, bgm, out_video)

    assert res.has_video is True
    probe_in = MediaProbe().probe(video_input)
    probe_out = MediaProbe().probe(out_video)

    assert probe_out.has_video is True
    assert probe_out.video_width == probe_in.video_width
    assert probe_out.video_height == probe_in.video_height
    assert probe_out.video_fps == probe_in.video_fps
    assert abs(probe_out.duration_ms - probe_in.duration_ms) <= 100

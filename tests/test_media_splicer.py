"""Testes da camada de midia FFmpeg (``editing.media_splicer``).

Cobertura: geracao do grafo ``-filter_complex`` (espelhando a camada de sinal),
uso de ``-filter_complex_script`` (limite Windows de ~8191 chars), erros de API
e emendas reais end-to-end incl. varios/centenas de cortes com sincronizacao
A/V estrita (CA2).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import video_engine.editing.media_splicer as msp
from video_engine.audio.models import TimeInterval
from video_engine.editing.media_probe import MediaInfo, MediaProbe
from video_engine.editing.media_splicer import MediaSplicer
from video_engine.editing.models import FadeCurve, SplicerConfig

DATA_DIR = Path(__file__).resolve().parent / "data"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def segs_ms(*pairs):
    return [TimeInterval(start_ms=s, end_ms=e) for s, e in pairs]


@pytest.fixture(scope="module")
def av_source(tmp_path_factory):
    if not FFMPEG or not FFPROBE:
        pytest.skip("ffmpeg/ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("splice") / "source.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=6:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=6",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return path


@pytest.fixture(scope="module")
def short_av_source(tmp_path_factory):
    if not FFMPEG or not FFPROBE:
        pytest.skip("ffmpeg/ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("splice") / "short_source.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=3:size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=44100:duration=3",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return path


# --------------------------------------------------------------------------- #
# Grafo de filtros (nao requer executaveis)
# --------------------------------------------------------------------------- #
def test_build_script_video_has_trims_fades_and_concat():
    script = MediaSplicer().build_filter_complex_script(
        segs_ms((0, 1000), (2000, 3000)), has_video=True
    )
    assert "[0:v]trim=start=0.000000:end=1.000000,setpts=PTS-STARTPTS[v0]" in script
    assert "[0:a]atrim=start=2.000000:end=3.000000,asetpts=PTS-STARTPTS" in script
    assert "afade=t=out:" in script
    assert "afade=t=in:" in script
    assert "concat=n=2:v=1:a=1[outv][outa]" in script


def test_build_script_audio_only_uses_v0concat():
    script = MediaSplicer().build_filter_complex_script(
        segs_ms((0, 1000), (2000, 3000)), has_video=False
    )
    assert "[0:v]" not in script
    assert "concat=n=2:v=0:a=1[outa]" in script


def test_build_script_single_segment_has_no_fade():
    script = MediaSplicer().build_filter_complex_script(
        segs_ms((0, 1000)), has_video=True
    )
    assert "afade" not in script
    assert "concat=n=1:v=1:a=1[outv][outa]" in script


def test_build_script_empty_segments_raises():
    with pytest.raises(ValueError, match="Segments list cannot be empty"):
        MediaSplicer().build_filter_complex_script([], has_video=True)


def test_build_script_equal_power_uses_qsin_iqsin():
    script = MediaSplicer().build_filter_complex_script(
        segs_ms((0, 1000), (2000, 3000)), has_video=False
    )
    assert "afade=t=out:" in script and "c=iqsin" in script
    assert "afade=t=in:" in script and "c=qsin" in script


def test_build_script_linear_uses_tri():
    splicer = MediaSplicer(config=SplicerConfig(curve=FadeCurve.LINEAR))
    script = splicer.build_filter_complex_script(
        segs_ms((0, 1000), (2000, 3000)), has_video=False
    )
    assert "c=tri" in script
    assert "c=qsin" not in script and "c=iqsin" not in script


def test_build_script_many_segments_exceeds_windows_cmdline_limit():
    segments = segs_ms(*[(i * 25, i * 25 + 25) for i in range(120)])
    script = MediaSplicer().build_filter_complex_script(segments, has_video=True)
    assert len(script) > 8191


def test_build_script_first_segment_only_fade_out_and_last_only_fade_in():
    script = MediaSplicer().build_filter_complex_script(
        segs_ms((0, 1000), (2000, 3000), (4000, 5000)), has_video=False
    )
    # Primeiro segmento: apenas fade-out; ultimo: apenas fade-in.
    first, mid, last = script.splitlines()[:3]
    assert "afade=t=out" in first and "afade=t=in" not in first
    assert "afade=t=in" in mid and "afade=t=out" in mid
    assert "afade=t=out" not in last and "afade=t=in" in last


# --------------------------------------------------------------------------- #
# Erros de API
# --------------------------------------------------------------------------- #
def test_splice_file_empty_segments_raises(tmp_path):
    with pytest.raises(ValueError, match="Segments list cannot be empty"):
        MediaSplicer().splice_file(DATA_DIR / "controlled.wav", tmp_path / "out.m4a", [])


def test_splice_file_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        MediaSplicer().splice_file(
            DATA_DIR / "nao_existe.wav", tmp_path / "out.m4a", segs_ms((0, 500))
        )


def test_splice_file_ffmpeg_unavailable_raises_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(msp.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg nao encontrado"):
        MediaSplicer().splice_file(
            DATA_DIR / "controlled.wav", tmp_path / "out.m4a", segs_ms((0, 500))
        )


def test_splice_file_ffmpeg_failure_raises_runtime_error(tmp_path, monkeypatch):
    def fake_run(self, cmd):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="falha sintetica")

    monkeypatch.setattr(MediaSplicer, "_run_ffmpeg", fake_run)
    with pytest.raises(RuntimeError, match="falha sintetica"):
        MediaSplicer().splice_file(
            DATA_DIR / "controlled.wav", tmp_path / "out.m4a", segs_ms((0, 500))
        )


# --------------------------------------------------------------------------- #
# Isolamento: invocacao real do FFmpeg via -filter_complex_script
# --------------------------------------------------------------------------- #
class StubProbe:
    def __init__(self, **kwargs):
        pass

    def probe(self, path):
        name = str(path)
        if name.endswith(".wav") and "out" not in name:
            info = MediaInfo(
                path=name, has_video=False, has_audio=True,
                audio_sample_rate=16000, duration_ms=7000,
            )
        else:
            info = MediaInfo(
                path=name, has_video=False, has_audio=True,
                audio_sample_rate=16000, duration_ms=1234,
            )
        return info


def test_splice_file_uses_filter_complex_script_file(tmp_path, monkeypatch):
    captured = {}

    def fake_run(self, cmd):
        captured["cmd"] = cmd
        flag = cmd.index("-filter_complex_script")
        script_arg = cmd[flag + 1]
        captured["script"] = Path(script_arg).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(msp, "MediaProbe", StubProbe)
    monkeypatch.setattr(MediaSplicer, "_run_ffmpeg", fake_run)

    out = tmp_path / "out.m4a"
    result = MediaSplicer().splice_file(
        DATA_DIR / "controlled.wav", out, segs_ms((0, 500), (1000, 1500))
    )
    assert "-filter_complex_script" in captured["cmd"]
    assert "-filter_complex" not in captured["cmd"]
    assert "concat=n=2:v=0:a=1[outa]" in captured["script"]
    assert result.num_segments == 2
    assert result.num_junctions == 1
    assert result.has_video is False
    assert result.total_duration_ms == 1234


# --------------------------------------------------------------------------- #
# End-to-end (FFmpeg real)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_splice_file_audio_only_mp3(tmp_path):
    out = tmp_path / "splice_out.m4a"
    result = MediaSplicer().splice_file(
        DATA_DIR / "sample.mp3", out, segs_ms((0, 1000), (1200, 2000))
    )
    assert result.has_video is False
    assert result.num_segments == 2
    assert result.num_junctions == 1
    assert result.total_duration_ms == pytest.approx(1800, abs=60)
    assert out.is_file()


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_splice_file_single_segment_has_no_junction(tmp_path):
    out = tmp_path / "single.m4a"
    result = MediaSplicer().splice_file(
        DATA_DIR / "sample.mp3", out, segs_ms((0, 1000))
    )
    assert result.num_segments == 1
    assert result.num_junctions == 0
    assert result.has_video is False


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_splice_file_video_av_sync(av_source, tmp_path):
    out = tmp_path / "sync.mp4"
    result = MediaSplicer().splice_file(
        av_source, out, segs_ms((0, 1000), (1500, 2500), (4000, 6000))
    )
    assert result.has_video is True
    assert result.num_segments == 3
    assert result.num_junctions == 2

    info = MediaProbe().probe(out)
    assert info.has_video and info.has_audio
    assert info.video_duration_ms is not None
    assert info.audio_duration_ms is not None
    # 1s + 1s + 2s = 4s de saida (tolerancia de buffering/re-encode).
    assert info.video_duration_ms == pytest.approx(4000, abs=60)
    # CA2: diferenca A/V < 1 frame (~33ms @30fps).
    assert abs(info.video_duration_ms - info.audio_duration_ms) <= 60


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_splice_file_many_cuts_av_sync(short_av_source, tmp_path):
    segments = segs_ms(*[(i * 25, i * 25 + 25) for i in range(120)])
    out = tmp_path / "many.mp4"
    result = MediaSplicer().splice_file(short_av_source, out, segments)
    assert result.has_video is True
    assert result.num_segments == 120
    assert result.num_junctions == 119

    info = MediaProbe().probe(out)
    assert info.has_video and info.has_audio
    assert info.video_duration_ms == pytest.approx(3000, abs=60)
    assert abs(info.video_duration_ms - info.audio_duration_ms) <= 60

"""Testes da camada de descoberta de streams via ffprobe (``media_probe``)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_engine.editing.media_probe import MediaInfo, MediaProbe, probe_media

DATA_DIR = Path(__file__).resolve().parent / "data"
FFPROBE = shutil.which("ffprobe")


@pytest.fixture(scope="module")
def av_source(tmp_path_factory):
    if not shutil.which("ffmpeg") or not FFPROBE:
        pytest.skip("ffmpeg/ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("av") / "av_source.mp4"
    cmd = [
        shutil.which("ffmpeg"),
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


def test_probe_wav_audio_only():
    info = MediaProbe().probe(DATA_DIR / "controlled.wav")
    assert info.has_audio is True
    assert info.has_video is False
    assert info.audio_sample_rate == 16000
    assert info.duration_ms == 7000
    assert isinstance(info, MediaInfo)


def test_probe_mp3_audio_only():
    info = probe_media(DATA_DIR / "sample.mp3")
    assert info.has_audio is True
    assert info.has_video is False
    assert info.audio_sample_rate == 44100
    assert info.duration_ms == 2534


def test_probe_video_has_both_streams(av_source):
    info = MediaProbe().probe(av_source)
    assert info.has_video is True
    assert info.has_audio is True
    assert info.video_fps is not None
    assert abs(info.video_fps - 30.0) < 0.5
    assert info.video_width == 320
    assert info.video_height == 240
    assert info.duration_ms == pytest.approx(3000, abs=60)


def test_probe_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        MediaProbe().probe(DATA_DIR / "nao_existe.mp4")


def test_probe_invalid_file_raises_runtime_error(tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("isto nao e midia", encoding="utf-8")
    with pytest.raises(RuntimeError):
        MediaProbe().probe(bogus)


def test_probe_ffprobe_unavailable_raises_runtime_error(monkeypatch, tmp_path):
    fake = tmp_path / "media.wav"
    fake.write_bytes(b"RIFF----WAVEfmt ")
    monkeypatch.setattr("video_engine.editing.media_probe.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffprobe nao encontrado"):
        MediaProbe().probe(fake)


def test_ffprobe_path_can_be_injected(tmp_path):
    bogus = tmp_path / "bogus.mp4"
    bogus.write_text("nao e midia", encoding="utf-8")
    probe = MediaProbe(ffprobe_path=str(tmp_path / "ffprobe_falso.exe"))
    with pytest.raises(RuntimeError):
        probe.probe(bogus)


@pytest.mark.skipif(FFPROBE is None, reason="ffprobe nao disponivel")
def test_probe_returns_path_normalized(av_source):
    info = MediaProbe().probe(str(av_source))
    assert Path(info.path).resolve() == Path(av_source).resolve()

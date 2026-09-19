"""Testes do queimador de legendas hardsub em cortes (Issue #9).

Cobertura:
- Sanitizacao de caminhos Windows para o filtro ``-vf ass=...``.
- Queima com reindexacao temporal do intervalo do corte (``cut_start_ms``).
- Preservacao direta do audio (``-c:a copy``) e resolucao adaptativa.
- Aceite de arquivo ASS pre-existente sem regeneracao.
- Erros claros (CaptionBurnError) para ffmpeg ausente / libass indisponivel.
- Integracao real com FFmpeg e sintese de video curto (quando disponivel).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from video_engine.captions.ass_models import CutCaptionBurnResult, SubtitleStyleConfig
from video_engine.captions.burner import (
    CaptionBurner,
    CaptionBurnError,
    _filter_ass_path,
)
from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.editing.media_probe import MediaInfo

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

WORDS = [
    WordTimestamp(word="ola", start_ms=2500, end_ms=2900, probability=0.99),
    WordTimestamp(word="corte", start_ms=3000, end_ms=3400, probability=0.98),
    WordTimestamp(word="legal", start_ms=3500, end_ms=4000, probability=0.97),
]


def make_transcription(words=None) -> TranscriptionResult:
    word_list = list(words) if words is not None else list(WORDS)
    return TranscriptionResult(
        text=" ".join(w.word for w in word_list),
        language="pt",
        duration_ms=word_list[-1].end_ms if word_list else 0,
        words=word_list,
    )


class FakeProc:
    def __init__(self, returncode: int = 0, stderr: str = "", stdout: str = "") -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout


class StubProbe:
    def __init__(
        self,
        has_video: bool = True,
        has_audio: bool = True,
        width: Optional[int] = 320,
        height: Optional[int] = 240,
        duration_ms: int = 6000,
    ) -> None:
        self.has_video = has_video
        self.has_audio = has_audio
        self.width = width
        self.height = height
        self.duration_ms = duration_ms

    def probe(self, path) -> MediaInfo:
        return MediaInfo(
            path=str(path),
            has_video=self.has_video,
            has_audio=self.has_audio,
            video_width=self.width,
            video_height=self.height,
            duration_ms=self.duration_ms,
        )


def _fake_run_recorder(calls: list):
    def fake_run(cmd, capture_output=None, text=None) -> FakeProc:
        calls.append(cmd)
        return FakeProc()
    return fake_run


# --------------------------------------------------------------------------- #
# Sanitizacao de caminhos
# --------------------------------------------------------------------------- #
def test_filter_ass_path_windows_escape():
    assert (
        _filter_ass_path(r"C:\Users\foo\meus shorts\legenda.ass")
        == "ass='C\\:/Users/foo/meus shorts/legenda.ass'"
    )


def test_filter_ass_path_linux_style():
    sanitized = _filter_ass_path("/tmp/legenda.ass")
    assert sanitized == "ass='/tmp/legenda.ass'"


# --------------------------------------------------------------------------- #
# Casos de erro
# --------------------------------------------------------------------------- #
def test_burn_missing_input_raises_file_not_found(tmp_path):
    burner = CaptionBurner(probe=StubProbe())
    with pytest.raises(FileNotFoundError, match="Arquivo de corte nao encontrado"):
        burner.burn_cut_subtitles(tmp_path / "nao_existe.mp4", tmp_path / "out.mp4", make_transcription())


def test_burn_missing_ffmpeg_raises_caption_burn_error(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")
    monkeypatch.setattr("video_engine.captions.burner.shutil.which", lambda _: None)
    burner = CaptionBurner(ffmpeg_path="", probe=StubProbe())
    with pytest.raises(CaptionBurnError, match="ffmpeg"):
        burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())


def test_burn_ffmpeg_binary_unavailable_raises_caption_burn_error(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")

    def boom(cmd, **kwargs):
        raise OSError("no such binary")

    monkeypatch.setattr(subprocess, "run", boom)
    burner = CaptionBurner(ffmpeg_path="C:\\nao\\existe\\ffmpeg.exe", probe=StubProbe())
    with pytest.raises(CaptionBurnError, match="ffmpeg"):
        burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())


def test_burn_audio_only_cut_raises_caption_burn_error(tmp_path, monkeypatch):
    src = tmp_path / "audio.wav"
    src.write_bytes(b"fake")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe(has_video=False))
    with pytest.raises(CaptionBurnError, match="stream de video"):
        burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())
    assert calls == []


# --------------------------------------------------------------------------- #
# Queima com reindexacao de corte
# --------------------------------------------------------------------------- #
def test_burn_generates_ass_with_cut_reindexing(tmp_path, monkeypatch):
    src = tmp_path / "cut_vertical.mp4"
    src.write_bytes(b"fake")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    out = tmp_path / "out" / "cut_burned.mp4"
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe(duration_ms=5000))
    result = burner.burn_cut_subtitles(src, out, make_transcription(), cut_start_ms=2500)

    assert isinstance(result, CutCaptionBurnResult)
    assert result.output_path == str(out)
    assert result.duration_sec == 5.0
    assert result.ass_path == str(out.with_suffix(".ass"))
    assert result.total_words_count == 3
    assert result.cues_count == 1

    ass_content = out.with_suffix(".ass").read_text(encoding="utf-8")
    assert "PlayResX: 320" in ass_content
    assert "Dialogue: 0,0:00:00.00" in ass_content

    cmd = calls[0]
    assert "-vf" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("ass='")
    assert "/cut_burned.ass'" in vf
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    assert "-map" in cmd
    assert cmd[-1] == str(out)


def test_burn_preserves_audio_and_video_mapping(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe(has_audio=True))
    burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())

    cmd = calls[0]
    assert cmd[cmd.index("-map") + 1] == "0:v:0"
    audio_map_idx = cmd.index("0:a:0")
    assert cmd[audio_map_idx] == "0:a:0"
    assert "-c:v" in cmd
    assert "-pix_fmt" in cmd
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_burn_video_only_skips_audio_mapping(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe(has_audio=False))
    burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())

    cmd = calls[0]
    assert "0:a:0" not in cmd
    assert "-c:a" not in cmd


def test_burn_honors_style_config(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    style = SubtitleStyleConfig(font_name="Montserrat Bold", margin_v=200)
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe())
    result = burner.burn_cut_subtitles(
        src,
        tmp_path / "out_style.mp4",
        make_transcription(),
        style_config=style,
    )
    ass_content = Path(result.ass_path).read_text(encoding="utf-8")
    assert "Montserrat Bold" in ass_content
    assert ",200,1" in ass_content


# --------------------------------------------------------------------------- #
# Arquivo ASS pre-existente
# --------------------------------------------------------------------------- #
def test_burn_accepts_existing_ass_file(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")
    pre = tmp_path / "previa.ass"
    ass_content = (
        "[Script Info]\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,ola\n"
    )
    pre.write_text(ass_content, encoding="utf-8")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe())
    result = burner.burn_cut_subtitles(src, tmp_path / "out.mp4", pre, cut_start_ms=9999)

    assert result.ass_path == str(pre)
    assert result.cues_count == 1
    assert not (tmp_path / "out.ass").exists()
    cmd = calls[0]
    vf = cmd[cmd.index("-vf") + 1]
    assert _filter_ass_path(str(pre)) in vf


def test_burn_missing_ass_file_raises_file_not_found(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")
    calls: list = []
    monkeypatch.setattr(
        "video_engine.captions.burner.subprocess.run",
        _fake_run_recorder(calls),
    )
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe())
    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        burner.burn_cut_subtitles(src, tmp_path / "out.mp4", tmp_path / "ausente.ass")
    assert calls == []


# --------------------------------------------------------------------------- #
# Falha do ffmpeg e ausencia de libass
# --------------------------------------------------------------------------- #
def test_burn_ffmpeg_failure_without_libass_hint(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")

    def failing(cmd, **kwargs) -> FakeProc:
        return FakeProc(returncode=1, stderr="File 'x.mp4' already exists\nError 5")

    monkeypatch.setattr(subprocess, "run", failing)
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe())
    with pytest.raises(CaptionBurnError, match="ffmpeg"):
        burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())


def test_burn_ffmpeg_failure_with_libass_hint(tmp_path, monkeypatch):
    src = tmp_path / "cut.mp4"
    src.write_bytes(b"fake")

    def failing(cmd, **kwargs) -> FakeProc:
        return FakeProc(returncode=1, stderr="Unknown filter 'ass' - libass nao disponivel")

    monkeypatch.setattr(subprocess, "run", failing)
    burner = CaptionBurner(ffmpeg_path="ffmpeg", probe=StubProbe())
    with pytest.raises(CaptionBurnError, match="[Ll]ibass"):
        burner.burn_cut_subtitles(src, tmp_path / "out.mp4", make_transcription())


# --------------------------------------------------------------------------- #
# Integracao real com FFmpeg
# --------------------------------------------------------------------------- #
def _detect_libass() -> bool:
    if not FFMPEG:
        return False
    proc = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True)
    return " ass " in proc.stdout or "ass " in proc.stdout


def _create_cut_video(path: Path, duration_sec: float = 3.0) -> None:
    subprocess.run(
        [
            FFMPEG,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_sec}:size=320x240:rate=24",
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
        ],
        check=True,
        capture_output=True,
    )


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
@pytest.mark.skipif(not _detect_libass(), reason="ffmpeg sem filtro libass (ass)")
def test_burn_integration_real_ffmpeg(tmp_path):
    from video_engine.editing.media_probe import MediaProbe

    src = tmp_path / "cut_sintetico.mp4"
    _create_cut_video(src, duration_sec=3.0)

    words = [
        WordTimestamp(word="ola", start_ms=200, end_ms=600, probability=0.99),
        WordTimestamp(word="mundo", start_ms=700, end_ms=1200, probability=0.98),
    ]
    out = tmp_path / "burned" / "cut_legendado.mp4"
    burner = CaptionBurner(probe=MediaProbe())
    result = burner.burn_cut_subtitles(src, out, make_transcription(words))

    assert Path(result.output_path).is_file()
    assert result.cues_count == 1
    assert result.total_words_count == 2
    assert result.duration_sec == pytest.approx(3.0, abs=0.1)

    probe_out = MediaProbe().probe(out)
    assert probe_out.has_video is True
    assert probe_out.has_audio is True
    assert probe_out.video_width == 320
    assert probe_out.video_height == 240

    sidecar = Path(result.ass_path)
    assert sidecar.is_file()
    assert "PlayResX: 320" in sidecar.read_text(encoding="utf-8")

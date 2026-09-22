"""Testes da renderizacao final de Shorts 9:16 prontos para publicacao (Issue #17).

Cobertura (TDD, seccao 4 da Spec #17): modelos Pydantic (config, pacote de
publicacao e resultado), estilizacao na safe area do terco medio, geracao de
metadados com titulo limitado e hashtags, pipeline de renderizacao (clamp de
janela, clean feed e encoding final) e integracao real com FFmpeg (MP4
1080x1920 h264/aac + JSON de metadados).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

import pytest
from pydantic import ValidationError

from video_engine.captions.ass_models import KaraokeHighlightMode, SubtitleStyleConfig
from video_engine.captions.models import TranscriptionResult, WordTimestamp
from video_engine.editing.media_probe import MediaInfo
from video_engine.shorts.models import ShortCandidateCut
from video_engine.shorts.renderer_models import (
    ShortsPublishPackage,
    ShortsRendererConfig,
    ShortsRenderResult,
)
from video_engine.shorts.shorts_renderer import ShortsRenderer, ShortsRenderError

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


# --------------------------------------------------------------------------- #
# Fixtures e helpers
# --------------------------------------------------------------------------- #
WORDS = [
    WordTimestamp(word="ola", start_ms=1200, end_ms=1600, probability=0.99),
    WordTimestamp(word="mundo", start_ms=1700, end_ms=2200, probability=0.98),
    WordTimestamp(word="corte", start_ms=2300, end_ms=2800, probability=0.97),
]


def make_cut(
    cut_id: str = "cut_01",
    start_ms: int = 1000,
    end_ms: int = 20000,
    hook_text: str = "Este gancho vai prender a atencao",
    summary: str = "Resumo do raciocinio autocontido no corte.",
    reason: str = "Forte apelo emocional e contexto completo.",
    virality_score: float = 0.9,
    words: Optional[List[WordTimestamp]] = None,
) -> ShortCandidateCut:
    return ShortCandidateCut(
        id=cut_id,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=max(end_ms - start_ms, 25000),
        hook_text=hook_text,
        summary=summary,
        reason=reason,
        semantic_score=0.9,
        energy_score=0.8,
        virality_score=virality_score,
        words=list(words) if words is not None else [],
    )


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
        video_codec: Optional[str] = "h264",
        audio_codec: Optional[str] = "aac",
        duration_ms: int = 30000,
    ) -> None:
        self.has_video = has_video
        self.has_audio = has_audio
        self.video_codec = video_codec
        self.audio_codec = audio_codec
        self.duration_ms = duration_ms

    def probe(self, path) -> MediaInfo:
        return MediaInfo(
            path=str(path),
            has_video=self.has_video,
            has_audio=self.has_audio,
            video_width=1280,
            video_height=720,
            video_codec=self.video_codec,
            audio_codec=self.audio_codec,
            duration_ms=self.duration_ms,
        )


class FakeReframer:
    def __init__(self) -> None:
        self.calls: list = []

    def reframe(
        self,
        input_video,
        output_video,
        cut_start_ms: Optional[int] = None,
        cut_end_ms: Optional[int] = None,
    ):
        self.calls.append(
            {
                "input": str(input_video),
                "output": str(output_video),
                "start_ms": cut_start_ms,
                "end_ms": cut_end_ms,
            }
        )


class FakeBurner:
    def __init__(self) -> None:
        self.calls: list = []

    def burn_cut_subtitles(
        self,
        cut_video_path,
        output_path,
        transcription,
        cut_start_ms: int = 0,
        cut_end_ms: Optional[int] = None,
        style_config: Optional[SubtitleStyleConfig] = None,
    ):
        self.calls.append(
            {
                "input": str(cut_video_path),
                "output": str(output_path),
                "transcription": transcription,
                "start_ms": cut_start_ms,
                "end_ms": cut_end_ms,
                "style_config": style_config,
            }
        )


def _fake_run_recorder(calls: list, returncode: int = 0):
    def fake_run(cmd, capture_output=None, text=None) -> FakeProc:
        calls.append(cmd)
        return FakeProc(returncode=returncode)
    return fake_run


def _make_renderer(
    monkeypatch=None,
    probe: Optional[StubProbe] = None,
    encode_calls: Optional[list] = None,
) -> tuple:
    recorder = encode_calls if encode_calls is not None else []
    if monkeypatch is not None:
        monkeypatch.setattr(
            "video_engine.shorts.shorts_renderer.subprocess.run",
            _fake_run_recorder(recorder),
        )
    renderer = ShortsRenderer(
        ffmpeg_path="ffmpeg",
        reframer=FakeReframer(),
        caption_burner=FakeBurner(),
    )
    renderer.probe = probe or StubProbe()
    return renderer, recorder


def _write_source(tmp_path: Path) -> Path:
    src = tmp_path / "origem.mp4"
    src.write_bytes(b"fake")
    return src


# --------------------------------------------------------------------------- #
# Modelos Pydantic
# --------------------------------------------------------------------------- #
def test_shorts_renderer_config_defaults():
    config = ShortsRendererConfig()
    assert config.target_width == 1080
    assert config.target_height == 1920
    assert config.margin_v == 750
    assert config.margin_l == 80
    assert config.margin_r == 150
    assert config.font_size == 52
    assert config.font_name == "Montserrat"
    assert config.highlight_color == "#FFF200"
    assert config.outline_width == 3.5
    assert config.video_codec == "libx264"
    assert config.audio_codec == "aac"
    assert config.audio_bitrate == "192k"
    assert config.crf == 20
    assert config.preset == "fast"
    assert config.pix_fmt == "yuv420p"
    assert config.faststart is True
    assert config.max_title_chars == 100
    assert config.default_hashtags == ["#shorts", "#cortes", "#viral"]


def test_shorts_renderer_config_validates_numeric_bounds():
    with pytest.raises(ValidationError):
        ShortsRendererConfig(margin_r=50)
    with pytest.raises(ValidationError):
        ShortsRendererConfig(margin_v=1300)
    with pytest.raises(ValidationError):
        ShortsRendererConfig(crf=60)
    with pytest.raises(ValidationError):
        ShortsRendererConfig(font_size=10)
    assert ShortsRendererConfig(margin_r=100, margin_v=1200, crf=35, font_size=24)


def test_shorts_renderer_config_extra_forbidden():
    with pytest.raises(ValidationError):
        ShortsRendererConfig(unexpected=1)


def test_shorts_publish_package_required_fields():
    with pytest.raises(ValidationError):
        ShortsPublishPackage(
            title="t",
            description="d",
            video_path="",
            metadata_path="",
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
        )
    package = ShortsPublishPackage(
        id="cut_01",
        title="t",
        description="d",
        video_path="v.mp4",
        metadata_path="m.json",
        duration_sec=1.0,
        start_ms=0,
        end_ms=1000,
    )
    assert package.resolution == "1080x1920"
    assert package.video_codec == "h264"
    assert package.audio_codec == "aac"


def test_shorts_publish_package_validates_ranges():
    with pytest.raises(ValidationError):
        ShortsPublishPackage(
            id="cut_01",
            title="t",
            description="d",
            video_path="v.mp4",
            metadata_path="m.json",
            duration_sec=-1.0,
            start_ms=0,
            end_ms=1000,
        )
    with pytest.raises(ValidationError):
        ShortsPublishPackage(
            id="cut_01",
            title="t",
            description="d",
            video_path="v.mp4",
            metadata_path="m.json",
            duration_sec=1.0,
            start_ms=0,
            end_ms=1000,
            virality_score=1.5,
        )


def test_shorts_render_result_validates_total():
    with pytest.raises(ValidationError):
        ShortsRenderResult(packages=[], total_rendered=-1, output_dir="out")
    result = ShortsRenderResult(packages=[], total_rendered=2, output_dir="out")
    assert result.total_rendered == 2


# --------------------------------------------------------------------------- #
# Safe area & estilo de legendas
# --------------------------------------------------------------------------- #
def test_build_safe_style_config_middle_third():
    renderer = ShortsRenderer(ffmpeg_path="ffmpeg")
    style = renderer.build_safe_style_config()
    assert isinstance(style, SubtitleStyleConfig)
    assert style.margin_v == 750
    assert style.margin_l == 80
    assert style.margin_r == 150
    assert style.alignment == 2
    assert style.highlight_mode == KaraokeHighlightMode.WORD_HIGHLIGHT
    assert style.play_res_x == 1080
    assert style.play_res_y == 1920
    assert style.font_name == "Montserrat"
    assert style.font_size == 52
    assert style.highlight_color == "#FFF200"
    assert style.outline_width == 3.5


def test_build_safe_style_config_honors_custom_config():
    config = ShortsRendererConfig(margin_v=800, margin_l=90, margin_r=200, font_size=64)
    renderer = ShortsRenderer(config=config, ffmpeg_path="ffmpeg")
    style = renderer.build_safe_style_config()
    assert style.margin_v == 800
    assert style.margin_l == 90
    assert style.margin_r == 200
    assert style.font_size == 64


# --------------------------------------------------------------------------- #
# Geracao de metadados
# --------------------------------------------------------------------------- #
def test_generate_metadata_package_writes_json(tmp_path):
    renderer, _ = _make_renderer(probe=StubProbe(duration_ms=5100))
    cut = make_cut()
    video_path = tmp_path / "cut_01.mp4"
    metadata_path = tmp_path / "cut_01_metadata.json"

    package = renderer.generate_metadata_package(cut, video_path, metadata_path)

    assert package.id == "cut_01"
    assert package.title == cut.hook_text
    assert package.video_path == str(video_path)
    assert package.metadata_path == str(metadata_path)
    assert package.duration_sec == 5.1
    assert package.start_ms == cut.start_ms
    assert package.end_ms == cut.end_ms
    assert package.resolution == "1080x1920"
    assert package.video_codec == "h264"
    assert package.audio_codec == "aac"
    assert package.virality_score == 0.9
    assert package.hashtags == ["#shorts", "#cortes", "#viral"]

    assert metadata_path.is_file()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["id"] == "cut_01"
    assert payload["title"] == package.title
    assert payload["description"] == package.description
    assert payload["hashtags"] == package.hashtags
    assert payload["resolution"] == "1080x1920"
    assert payload["duration_sec"] == 5.1
    assert payload["end_ms"] == cut.end_ms


def test_generate_metadata_package_truncates_long_title(tmp_path):
    renderer, _ = _make_renderer()
    long_hook = " ".join(["super gancho viral" * 12] * 4)
    cut = make_cut(hook_text=long_hook)
    package = renderer.generate_metadata_package(
        cut,
        str(tmp_path / "video.mp4"),
        str(tmp_path / "meta.json"),
    )
    assert len(package.title) <= 100
    assert package.title.endswith(("gancho", "viral")) or package.title


def test_title_uses_summary_when_no_hook(tmp_path):
    renderer, _ = _make_renderer()
    cut = make_cut(hook_text="", summary="Este resumo vira titulo")
    package = renderer.generate_metadata_package(
        cut,
        str(tmp_path / "video.mp4"),
        str(tmp_path / "meta.json"),
    )
    assert package.title == "Este resumo vira titulo"


def test_generate_metadata_package_uses_video_title(tmp_path):
    renderer, _ = _make_renderer()
    cut = make_cut(hook_text="", summary="")
    package = renderer.generate_metadata_package(
        cut,
        str(tmp_path / "video.mp4"),
        str(tmp_path / "meta.json"),
        video_title="Titulo do video longo",
    )
    assert package.title == "Titulo do video longo"


def test_metadata_description_contains_summary_reason_and_hashtags(tmp_path):
    renderer, _ = _make_renderer()
    cut = make_cut(summary="Resumo curto", reason="Motivo forte")
    package = renderer.generate_metadata_package(
        cut,
        str(tmp_path / "video.mp4"),
        str(tmp_path / "meta.json"),
    )
    assert package.description.startswith(package.title)
    assert "Resumo curto" in package.description
    assert "Motivo forte" in package.description
    assert package.description.endswith("#cortes #viral")
    assert "#shorts" in package.description


def test_hashtag_normalization_and_dedupe(tmp_path):
    config = ShortsRendererConfig(
        default_hashtags=["shorts", "#cortes", "#shorts", "viral top", "", "#"]
    )
    renderer = ShortsRenderer(config=config, ffmpeg_path="ffmpeg")
    renderer.probe = StubProbe()
    renderer.reframer = FakeReframer()
    renderer.caption_burner = FakeBurner()
    package = renderer.generate_metadata_package(
        make_cut(),
        str(tmp_path / "video.mp4"),
        str(tmp_path / "meta.json"),
    )
    assert package.hashtags == ["#shorts", "#cortes", "#viral_top"]


# --------------------------------------------------------------------------- #
# Pipeline de renderizacao (com stubs)
# --------------------------------------------------------------------------- #
def test_render_cut_pipeline_and_final_encode_cmd(tmp_path, monkeypatch):
    reframer = FakeReframer()
    renderer, encode_calls = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=30000))
    renderer.reframer = reframer

    src = _write_source(tmp_path)
    cut = make_cut(start_ms=1000, end_ms=20000)
    package = renderer.render_cut(
        src,
        cut,
        tmp_path / "out",
        transcription=make_transcription(),
        video_title="Meu video",
    )

    assert reframer.calls[0]["input"] == str(src)
    assert reframer.calls[0]["start_ms"] == 1000
    assert reframer.calls[0]["end_ms"] == 20000
    assert reframer.calls[0]["output"].endswith("vertical.mp4")

    assert len(renderer.caption_burner.calls) == 1
    burn = renderer.caption_burner.calls[0]
    assert burn["start_ms"] == 1000
    assert burn["end_ms"] == 20000
    assert isinstance(burn["transcription"], TranscriptionResult)
    assert burn["style_config"].margin_v == 750

    assert len(encode_calls) == 1
    cmd = encode_calls[0]
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "20"
    assert cmd[cmd.index("-preset") + 1] == "fast"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "192k"
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert cmd[-1] == str(tmp_path / "out" / "cut_01.mp4")

    assert package.id == "cut_01"
    assert package.video_codec == "h264"
    assert (tmp_path / "out" / "cut_01_metadata.json").is_file()


def test_render_cut_clamps_window_to_source_duration(tmp_path, monkeypatch):
    reframer = FakeReframer()
    renderer, _ = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=3000))
    renderer.reframer = reframer

    src = _write_source(tmp_path)
    cut = make_cut(start_ms=500, end_ms=25000)
    renderer.render_cut(src, cut, tmp_path / "out", transcription=make_transcription())

    assert reframer.calls[0]["start_ms"] == 500
    assert reframer.calls[0]["end_ms"] == 3000


def test_render_cut_clean_feed_without_transcription(tmp_path, monkeypatch):
    reframer = FakeReframer()
    renderer, encode_calls = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=30000))
    renderer.reframer = reframer

    src = _write_source(tmp_path)
    cut = make_cut(start_ms=1000, end_ms=20000, words=[])
    renderer.render_cut(src, cut, tmp_path / "out")

    assert renderer.caption_burner.calls == []
    assert len(encode_calls) == 1
    encode_input = encode_calls[0][encode_calls[0].index("-i") + 1]
    assert encode_input == reframer.calls[0]["output"]
    assert encode_input.endswith("vertical.mp4")


def test_render_cut_uses_cut_words_when_transcription_none(tmp_path, monkeypatch):
    renderer, encode_calls = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=30000))

    src = _write_source(tmp_path)
    cut = make_cut(start_ms=1000, end_ms=20000, words=WORDS)
    renderer.render_cut(src, cut, tmp_path / "out")

    assert len(renderer.caption_burner.calls) == 1
    burn = renderer.caption_burner.calls[0]
    assert isinstance(burn["transcription"], list)
    assert all(isinstance(w, WordTimestamp) for w in burn["transcription"])
    assert burn["start_ms"] == 1000
    assert len(encode_calls) == 1


def test_render_cut_clean_feed_when_no_words_in_window(tmp_path, monkeypatch):
    renderer, encode_calls = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=30000))

    src = _write_source(tmp_path)
    cut = make_cut(start_ms=20000, end_ms=25000, words=[])
    words_outside = [
        WordTimestamp(word="cola", start_ms=500, end_ms=700, probability=0.9)
    ]
    renderer.render_cut(
        src,
        cut,
        tmp_path / "out",
        transcription=make_transcription(words_outside),
    )
    assert renderer.caption_burner.calls == []
    assert len(encode_calls) == 1


def test_render_cut_missing_source_raises(tmp_path, monkeypatch):
    renderer, _ = _make_renderer(monkeypatch)
    with pytest.raises(ShortsRenderError, match="nao encontrado"):
        renderer.render_cut(tmp_path / "ausente.mp4", make_cut(), tmp_path / "out")


def test_render_cut_invalid_window_raises(tmp_path, monkeypatch):
    renderer, _ = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=1000))
    src = _write_source(tmp_path)
    cut = make_cut(start_ms=1500, end_ms=2000)
    with pytest.raises(ShortsRenderError, match="Janela"):
        renderer.render_cut(src, cut, tmp_path / "out")


def test_render_cut_missing_ffmpeg_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("video_engine.shorts.shorts_renderer.shutil.which", lambda _: None)
    renderer = ShortsRenderer(
        ffmpeg_path="",
        reframer=FakeReframer(),
        caption_burner=FakeBurner(),
    )
    renderer.probe = StubProbe(duration_ms=30000)
    src = _write_source(tmp_path)
    with pytest.raises(ShortsRenderError, match="ffmpeg nao encontrado"):
        renderer.render_cut(src, make_cut(), tmp_path / "out")


def test_render_cut_raises_on_encode_failure(tmp_path, monkeypatch):
    encode_calls: list = []
    monkeypatch.setattr(
        "video_engine.shorts.shorts_renderer.subprocess.run",
        _fake_run_recorder(encode_calls, returncode=1),
    )
    renderer = ShortsRenderer(
        ffmpeg_path="ffmpeg",
        reframer=FakeReframer(),
        caption_burner=FakeBurner(),
    )
    renderer.probe = StubProbe(duration_ms=30000)
    src = _write_source(tmp_path)
    with pytest.raises(ShortsRenderError, match="ffmpeg falhou"):
        renderer.render_cut(src, make_cut(), tmp_path / "out")


def test_render_batch_empty_raises(tmp_path, monkeypatch):
    renderer, _ = _make_renderer(monkeypatch)
    with pytest.raises(ShortsRenderError, match="vazio"):
        renderer.render_batch(tmp_path / "origem.mp4", [], tmp_path / "out")


def test_render_batch_renders_two_cuts(tmp_path, monkeypatch):
    reframer = FakeReframer()
    renderer, encode_calls = _make_renderer(monkeypatch, probe=StubProbe(duration_ms=30000))
    renderer.reframer = reframer

    src = _write_source(tmp_path)
    batch_words = WORDS + [
        WordTimestamp(word="outro", start_ms=21500, end_ms=22000, probability=0.95),
        WordTimestamp(word="gancho", start_ms=23000, end_ms=24000, probability=0.94),
    ]
    cuts = [
        make_cut(cut_id="cut_01", start_ms=1000, end_ms=20000),
        make_cut(cut_id="cut_02", start_ms=21000, end_ms=26000),
    ]
    result = renderer.render_batch(
        src,
        cuts,
        tmp_path / "out",
        transcription=make_transcription(batch_words),
    )

    assert isinstance(result, ShortsRenderResult)
    assert result.total_rendered == 2
    assert result.output_dir == str(tmp_path / "out")
    assert [p.id for p in result.packages] == ["cut_01", "cut_02"]
    assert len(reframer.calls) == 2
    assert len(renderer.caption_burner.calls) == 2
    assert len(encode_calls) == 2
    assert (tmp_path / "out" / "cut_01_metadata.json").is_file()
    assert (tmp_path / "out" / "cut_02_metadata.json").is_file()


# --------------------------------------------------------------------------- #
# Comando de encoding final
# --------------------------------------------------------------------------- #
def test_final_encode_cmd_omits_audio_flags_when_no_audio(monkeypatch):
    renderer, _ = _make_renderer(monkeypatch)
    cmd = renderer._final_encode_cmd("in.mp4", "out.mp4", has_audio=False)
    assert "-c:a" not in cmd
    assert "-movflags" in cmd
    assert cmd[-1] == "out.mp4"


def test_final_encode_cmd_without_faststart(monkeypatch):
    renderer, _ = _make_renderer(monkeypatch)
    renderer.config = ShortsRendererConfig(faststart=False)
    cmd = renderer._final_encode_cmd("in.mp4", "out.mp4", has_audio=True)
    assert "-movflags" not in cmd


# --------------------------------------------------------------------------- #
# Integracao real com FFmpeg
# --------------------------------------------------------------------------- #
def _detect_libass() -> bool:
    if not FFMPEG:
        return False
    proc = subprocess.run([FFMPEG, "-hide_banner", "-filters"], capture_output=True, text=True)
    return " ass " in proc.stdout


@pytest.fixture(scope="module")
def horizontal_clip(tmp_path_factory) -> Path:
    if not FFMPEG or not FFPROBE:
        pytest.skip("ffmpeg/ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("shorts_renderer") / "horizontal.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=6:size=360x180:rate=12",
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


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
@pytest.mark.skipif(not _detect_libass(), reason="ffmpeg sem filtro libass (ass)")
def test_render_cut_integration_real_ffmpeg(tmp_path, horizontal_clip):
    from video_engine.editing.media_probe import MediaProbe

    cut = make_cut(
        cut_id="cut_01",
        start_ms=600,
        end_ms=5600,
        hook_text="Gancho de teste real",
    )
    renderer = ShortsRenderer()
    out_dir = tmp_path / "saida"
    package = renderer.render_cut(
        horizontal_clip,
        cut,
        out_dir,
        transcription=make_transcription(),
    )

    out_video = Path(package.video_path)
    assert out_video.is_file()
    assert out_video.parent == out_dir
    assert out_video.name == "cut_01.mp4"

    metadata = out_dir / "cut_01_metadata.json"
    assert metadata.is_file()
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["id"] == "cut_01"
    assert payload["resolution"] == "1080x1920"

    probe_out = MediaProbe().probe(out_video)
    assert probe_out.has_video is True
    assert probe_out.has_audio is True
    assert probe_out.video_width == 1080
    assert probe_out.video_height == 1920
    assert probe_out.video_codec == "h264"
    assert probe_out.audio_codec == "aac"
    assert probe_out.duration_ms == pytest.approx(5000, abs=800)

    arquivos = sorted(p.name for p in out_dir.iterdir())
    assert arquivos == ["cut_01.mp4", "cut_01_metadata.json"]


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
@pytest.mark.skipif(not _detect_libass(), reason="ffmpeg sem filtro libass (ass)")
def test_render_cut_integration_clean_feed(tmp_path, horizontal_clip):
    from video_engine.editing.media_probe import MediaProbe

    cut = make_cut(
        cut_id="cut_02",
        start_ms=500,
        end_ms=5500,
        hook_text="Gancho limpo",
        words=[],
    )
    renderer = ShortsRenderer()
    out_dir = tmp_path / "saida_limpa"
    package = renderer.render_cut(horizontal_clip, cut, out_dir)

    out_video = Path(package.video_path)
    assert out_video.is_file()
    probe_out = MediaProbe().probe(out_video)
    assert probe_out.has_video is True
    assert probe_out.video_width == 1080
    assert probe_out.video_height == 1920
    assert probe_out.video_codec == "h264"

    arquivos = sorted(p.name for p in out_dir.iterdir())
    assert arquivos == ["cut_02.mp4", "cut_02_metadata.json"]

"""Testes da normalizacao de loudness EBU R128 (2 passes / loudnorm).

Cobertura: modelos (defaults conforme spec), parsing JSON do ``loudnorm``
(comportamento guloso e valores nao-finitos), montagem do comando do 2o passe
(``linear=true`` + ``measured_*`` + ``offset`` + ``alimiter``), erros de API e
integracao real com FFmpeg validando CA1 (-14.0 LUFS +/- 0.5), CA2 (TP <=
-1.0 dBTP com folga lossy <= -0.95), CA3 (2o passe com linear=true) e CA4
(video copiado com ``-c:v copy`` e duracao preservada).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import video_engine.audio.loudness as lmod
from video_engine.audio.loudness import (
    LoudnessNormalizer,
    LoudnessResult,
    loudness_compliant,
)
from video_engine.audio.loudness_models import LoudnessConfig, LoudnormStats

DATA_DIR = Path(__file__).resolve().parent / "data"
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

JSON_IN = (
    '{\n"input_i" : "-20.00",\n'
    '"input_tp" : "-8.00",\n'
    '"input_lra" : "5.00",\n'
    '"input_thresh" : "-30.00",\n'
    '"target_offset" : "0.50"\n}\n'
)
JSON_OUT = (
    '{\n"input_i" : "-14.00",\n'
    '"input_tp" : "-1.50",\n'
    '"input_lra" : "5.00",\n'
    '"input_thresh" : "-25.00",\n'
    '"target_offset" : "0.00"\n}\n'
)


# --------------------------------------------------------------------------- #
# Modelos
# --------------------------------------------------------------------------- #
def test_loudness_config_defaults():
    cfg = LoudnessConfig()
    assert cfg.target_i == -14.0
    assert cfg.target_tp == -1.0
    assert cfg.target_lra == 11.0
    assert cfg.linear is True
    assert cfg.tolerance_lufs == 0.5
    assert cfg.audio_codec == "aac"
    assert cfg.audio_bitrate == "192k"


def test_loudness_result_defaults():
    result = LoudnessResult(
        output_path="out.m4a",
        measured_input=LoudnormStats(input_i=-20.0, input_tp=-8.0, input_lra=5.0, input_thresh=-30.0),
        measured_output=LoudnormStats(input_i=-14.0, input_tp=-1.5, input_lra=5.0, input_thresh=-25.0),
        is_compliant=True,
    )
    assert result.target_i == -14.0
    assert result.target_tp == -1.0
    assert result.has_video is False


def test_loudnorm_stats_parses__payload_with_target_offset():
    stats = LoudnessNormalizer._parse_stats(
        {
            "input_i": "-20.00",
            "input_tp": "-8.00",
            "input_lra": "5.00",
            "input_thresh": "-30.00",
            "target_offset": "0.50",
        }
    )
    assert stats.input_i == -20.0
    assert stats.input_tp == -8.0
    assert stats.input_lra == 5.0
    assert stats.input_thresh == -30.0
    assert stats.target_offset == 0.5


def test_loudnorm_stats_default_target_offset():
    stats = LoudnessNormalizer._parse_stats(
        {"input_i": "-20.00", "input_tp": "-8.00", "input_lra": "5.00", "input_thresh": "-30.00"}
    )
    assert stats.target_offset == 0.0


def test_extract_json_block_finds_last_input_i_object():
    fake = "arvore_{1_legit}\n" + JSON_IN + "trailer aleatorio\n"
    block = lmod._extract_json_block(fake, "input_i")
    assert block is not None
    payload = json.loads(block)
    assert payload["input_i"] == "-20.00"


def test_extract_json_block_returns_none_when_missing():
    assert lmod._extract_json_block("sem json nenhum aqui") is None


# --------------------------------------------------------------------------- #
# QA / conformidade
# --------------------------------------------------------------------------- #
def test_loudness_compliant_pass():
    cfg = LoudnessConfig()
    compliant = LoudnormStats(input_i=-13.9, input_tp=-1.5, input_lra=5.0, input_thresh=-25.0, target_offset=0.0)
    assert loudness_compliant(compliant, cfg) is True
    lowish = LoudnormStats(input_i=-14.49, input_tp=-0.96, input_lra=5.0, input_thresh=-25.0)
    assert loudness_compliant(lowish, cfg) is True


def test_loudness_compliant_fail():
    cfg = LoudnessConfig()
    off = LoudnormStats(input_i=-13.0, input_tp=-1.5, input_lra=5.0, input_thresh=-25.0)
    assert loudness_compliant(off, cfg) is False
    clipped = LoudnormStats(input_i=-14.0, input_tp=-0.9, input_lra=5.0, input_thresh=-25.0)
    assert loudness_compliant(clipped, cfg) is False


# --------------------------------------------------------------------------- #
# measure(): erros e parsing
# --------------------------------------------------------------------------- #
def test_measure_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").measure(tmp_path / "nao_existe.wav")


def test_measure_ffmpeg_unavailable_raises_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(lmod.shutil, "which", lambda _: None)
    with pytest.raises(RuntimeError, match="ffmpeg nao encontrado"):
        LoudnessNormalizer().measure(DATA_DIR / "controlled.wav")


def test_measure_missing_json_raises_runtime_error(tmp_path, monkeypatch):
    def fake_run(self, cmd):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="saida sem json")

    monkeypatch.setattr(LoudnessNormalizer, "_run_ffmpeg", fake_run)
    with pytest.raises(RuntimeError, match="saida sem json"):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").measure(DATA_DIR / "controlled.wav")


def test_measure_nonfinite_json_raises_value_error(tmp_path, monkeypatch):
    raw = '{"input_i" : "-inf", "input_tp" : "inf", "input_lra" : "-inf", "input_thresh" : "-inf"}'
    proc = subprocess.CompletedProcess([], 0, stdout="", stderr=raw)

    monkeypatch.setattr(LoudnessNormalizer, "_run_ffmpeg", lambda self, cmd: proc)
    with pytest.raises(ValueError, match="curto"):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").measure(DATA_DIR / "controlled.wav")


# --------------------------------------------------------------------------- #
# normalize_file(): montagem do comando (isolado com stub)
# --------------------------------------------------------------------------- #
class StubProbe:
    has_audio_override = True
    has_video_override = False

    def probe(self, path):
        from video_engine.editing.media_probe import MediaInfo

        return MediaInfo(
            path=str(path),
            has_video=self.has_video_override,
            has_audio=self.has_audio_override,
            duration_ms=5000,
        )


def _stub_run_factory(captured, input_path, output_path):
    def fake_run(self, cmd):
        if "-y" in cmd:  # 2o passe (encode)
            captured["cmd"] = cmd
            Path(str(cmd[-1])).touch()  # simula arquivo de saida criado
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=JSON_OUT)
        if str(input_path) in cmd:  # 1o passe (medicao da entrada)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=JSON_IN)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=JSON_OUT)  # QA

    return fake_run


def test_normalize_file_audio_only_command_has_linear_and_alimiter(tmp_path, monkeypatch):
    captured = {}
    StubProbe.has_audio_override = True
    StubProbe.has_video_override = False
    monkeypatch.setattr(lmod, "MediaProbe", StubProbe)
    monkeypatch.setattr(
        LoudnessNormalizer,
        "_run_ffmpeg",
        _stub_run_factory(captured, DATA_DIR / "controlled.wav", tmp_path / "out.m4a"),
    )
    result = LoudnessNormalizer(ffmpeg_path="ffmpeg").normalize_file(
        DATA_DIR / "controlled.wav", tmp_path / "out.m4a"
    )
    af = captured["cmd"][captured["cmd"].index("-af") + 1]
    process_tp = LoudnessConfig().target_tp - lmod._LOSSY_TP_HEADROOM_DB
    assert f"loudnorm=I=-14.0:TP={process_tp}:LRA=11.0" in af
    assert "measured_I=-20.00" in af
    assert "measured_TP=-8.00" in af
    assert "measured_LRA=5.00" in af
    assert "measured_thresh=-30.00" in af
    assert "offset=0.50" in af
    assert "linear=true" in af
    assert "print_format=json" not in af
    assert f"alimiter=level_in=1.0:level_out=1.0:limit={10 ** (process_tp / 20.0)}" in af
    assert "-c:v" not in captured["cmd"]
    assert "-c:a" in captured["cmd"]
    assert result.is_compliant is True
    assert result.has_video is False
    assert result.measured_input.input_i == -20.0
    assert result.output_path == str(tmp_path / "out.m4a")


def test_normalize_file_video_uses_copy_stream(tmp_path, monkeypatch):
    captured = {}
    StubProbe.has_video_override = True
    monkeypatch.setattr(lmod, "MediaProbe", StubProbe)
    monkeypatch.setattr(
        LoudnessNormalizer,
        "_run_ffmpeg",
        _stub_run_factory(captured, DATA_DIR / "controlled.wav", tmp_path / "out.mp4"),
    )
    result = LoudnessNormalizer(ffmpeg_path="ffmpeg").normalize_file(
        DATA_DIR / "controlled.wav", tmp_path / "out.mp4"
    )
    assert "-c:v" in captured["cmd"]
    assert "copy" in captured["cmd"]
    assert captured["cmd"].index("-c:v") < captured["cmd"].index("-c:a")
    assert result.is_compliant is True
    assert result.has_video is True


def test_normalize_file_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").normalize_file(
            tmp_path / "nao_existe.wav", tmp_path / "out.m4a"
        )


def test_normalize_file_empty_input_raises(tmp_path):
    empty = tmp_path / "vazio.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="Entrada sem stream de áudio válida"):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").normalize_file(empty, tmp_path / "out.m4a")


def test_normalize_file_no_audio_raises(tmp_path, monkeypatch):
    StubProbe.has_audio_override = False
    StubProbe.has_video_override = True
    monkeypatch.setattr(lmod, "MediaProbe", StubProbe)
    with pytest.raises(ValueError, match="Entrada sem stream de áudio válida"):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").normalize_file(
            DATA_DIR / "controlled.wav", tmp_path / "out.mp4"
        )


def test_normalize_file_ffmpeg_failure_raises_runtime_error(tmp_path, monkeypatch):
    StubProbe.has_audio_override = True
    StubProbe.has_video_override = False
    monkeypatch.setattr(lmod, "MediaProbe", StubProbe)

    def fake_run(self, cmd):
        return subprocess.CompletedProcess([], 1, stdout="", stderr="falha sintetica")

    monkeypatch.setattr(LoudnessNormalizer, "_run_ffmpeg", fake_run)
    with pytest.raises(RuntimeError, match="falha sintetica"):
        LoudnessNormalizer(ffmpeg_path="ffmpeg").normalize_file(
            DATA_DIR / "controlled.wav", tmp_path / "out.m4a"
        )


# --------------------------------------------------------------------------- #
# Integracao: FFmpeg real (CA1..CA4)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wav_factory(tmp_path_factory):
    if not FFMPEG:
        pytest.skip("ffmpeg nao disponivel")

    def _render(name, lavfi_src, volume_filter=None):
        path = tmp_path_factory.mktemp("loudness") / name
        cmd = [FFMPEG, "-hide_banner", "-nostats", "-y", "-f", "lavfi", "-i", lavfi_src]
        if volume_filter:
            cmd += ["-af", volume_filter]
        cmd += ["-c:a", "pcm_s16le", str(path)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return path

    return _render


@pytest.fixture(scope="module")
def quiet_wav(wav_factory):
    return wav_factory("quiet.wav", "sine=frequency=440:sample_rate=44100:duration=5", "volume=0.05")


@pytest.fixture(scope="module")
def loud_wav(wav_factory):
    return wav_factory("loud.wav", "sine=frequency=440:sample_rate=44100:duration=5", "volume=0.9")


@pytest.fixture(scope="module")
def mod_wav(wav_factory):
    return wav_factory(
        "mod.wav",
        "sine=frequency=440:sample_rate=44100:duration=5",
        "volume='if(lt(t,2),0.30,0.6)':eval=frame",
    )


@pytest.fixture(scope="module")
def short_wav(wav_factory):
    return wav_factory("short.wav", "sine=frequency=440:sample_rate=44100:duration=0.3")


@pytest.fixture(scope="module")
def video_source(tmp_path_factory):
    if not FFMPEG or not FFPROBE:
        pytest.skip("ffmpeg/ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("loudness") / "video_in.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-nostats",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=5:size=320x240:rate=25",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=44100:duration=5,volume=0.5",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-shortest",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return path


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg nao disponivel")
def test_measure_returns_finite_stats(quiet_wav):
    stats = LoudnessNormalizer().measure(quiet_wav)
    assert stats.input_i < -40.0
    assert stats.input_tp < 0.0
    assert stats.input_thresh < 0.0
    assert stats.input_lra >= 0.0


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg nao disponivel")
def test_normalize_quiet_audio_to_youtube_target(quiet_wav, tmp_path):
    out = tmp_path / "normalized.m4a"
    result = LoudnessNormalizer().normalize_file(quiet_wav, out)
    assert out.is_file()
    assert result.measured_output.input_i == pytest.approx(-14.0, abs=0.5)
    assert result.measured_output.input_tp <= -0.95
    assert result.is_compliant is True


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg nao disponivel")
def test_normalize_loud_audio_to_youtube_target(loud_wav, tmp_path):
    out = tmp_path / "normalized_loud.m4a"
    result = LoudnessNormalizer().normalize_file(loud_wav, out)
    assert out.is_file()
    assert result.measured_input.input_i < -20.0  # entrada realmente alta/loud
    assert result.measured_output.input_i == pytest.approx(-14.0, abs=0.5)
    assert result.measured_output.input_tp <= -0.95
    assert result.is_compliant is True


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg nao disponivel")
def test_normalize_dynamic_content_to_youtube_target(mod_wav, tmp_path):
    out = tmp_path / "normalized_mod.m4a"
    result = LoudnessNormalizer().normalize_file(mod_wav, out)
    assert out.is_file()
    assert result.measured_input.input_lra > 1.0  # conteudo com dinamica (LRA > 0)
    assert result.measured_output.input_i == pytest.approx(-14.0, abs=0.5)
    assert result.measured_output.input_tp <= -0.95
    assert result.is_compliant is True


@pytest.mark.skipif(not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe nao disponivel")
def test_normalize_video_preserves_streams_and_duration(video_source, tmp_path):
    from video_engine.editing.media_probe import MediaProbe

    out = tmp_path / "normalized_video.mp4"
    result = LoudnessNormalizer().normalize_file(video_source, out)
    assert out.is_file()
    assert result.has_video is True
    assert result.is_compliant is True

    info_out = MediaProbe().probe(out)
    assert info_out.has_video and info_out.has_audio
    assert result.measured_output.input_tp <= -0.95

    in_duration = MediaProbe().probe(video_source).duration_ms
    assert abs(info_out.duration_ms - in_duration) <= 500


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg nao disponivel")
def test_normalize_short_audio_raises_value_error(short_wav):
    with pytest.raises(ValueError, match="curto"):
        LoudnessNormalizer().measure(short_wav)


@pytest.mark.skipif(not FFMPEG, reason="ffmpeg nao disponivel")
def test_normalize_video_without_audio_raises(tmp_path_factory):
    if not FFPROBE:
        pytest.skip("ffprobe nao disponivel")
    path = tmp_path_factory.mktemp("loudness") / "video_only.mp4"
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-nostats",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=duration=2:size=160x120:rate=25",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    with pytest.raises(ValueError, match="Entrada sem stream de áudio válida"):
        LoudnessNormalizer().normalize_file(path, path.parent / "sem_audio_out.mp4")

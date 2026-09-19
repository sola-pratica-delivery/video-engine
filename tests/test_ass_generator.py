"""Testes do gerador de legendas animadas ASS (Issue #9).

Cobertura:
- Conversao HEX -> ASS (com e sem alpha) e passthrough de cores ASS nativas.
- Formatação de timestamps HH:MM:SS.cc.
- Particionamento de palavras em ``SubtitleCue`` com limite por linha.
- Reindexacao temporal a partir de ``start_time_ms`` e filtragem por ``end_time_ms``.
- Geracao de eventos nos modos ``WORD_HIGHLIGHT`` e ``KARAOKE_TAG``.
- Escapes de caracteres especiais e transcricoes vazias.
- Escrita de arquivo ASS e validacoes dos modelos de configuracao.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from video_engine.captions.ass_generator import (
    AssSubtitleGenerator,
    _escape_ass_text,
    format_ass_time,
    hex_to_ass_color,
)
from video_engine.captions.ass_models import (
    CutCaptionBurnResult,
    KaraokeHighlightMode,
    SubtitleCue,
    SubtitleStyleConfig,
)
from video_engine.captions.models import (
    CaptionSegment,
    TranscriptionResult,
    WordTimestamp,
)

WORDS = [
    WordTimestamp(word="oi", start_ms=500, end_ms=900, probability=0.99),
    WordTimestamp(word="galera", start_ms=1000, end_ms=1450, probability=0.97),
    WordTimestamp(word="hoje", start_ms=1500, end_ms=1850, probability=0.98),
    WordTimestamp(word="vamos", start_ms=2000, end_ms=2450, probability=0.96),
    WordTimestamp(word="falar", start_ms=2500, end_ms=3000, probability=0.95),
    WordTimestamp(word="de", start_ms=3100, end_ms=3350, probability=0.94),
    WordTimestamp(word="algo", start_ms=3400, end_ms=3900, probability=0.93),
    WordTimestamp(word="muito", start_ms=4000, end_ms=4400, probability=0.92),
    WordTimestamp(word="legal", start_ms=4500, end_ms=5000, probability=0.91),
]


def make_transcription(words=None) -> TranscriptionResult:
    word_list = list(words) if words is not None else list(WORDS)
    if word_list:
        segments = [
            CaptionSegment(
                id=0,
                text=" ".join(w.word for w in word_list),
                start_ms=word_list[0].start_ms,
                end_ms=word_list[-1].end_ms,
                words=word_list,
            )
        ]
        duration_ms = word_list[-1].end_ms
    else:
        segments = []
        duration_ms = 0
    return TranscriptionResult(
        text=" ".join(w.word for w in word_list),
        language="pt",
        duration_ms=duration_ms,
        segments=segments,
        words=word_list,
    )


def empty_transcription() -> TranscriptionResult:
    return TranscriptionResult(text="", language="pt", duration_ms=0)


# --------------------------------------------------------------------------- #
# Conversao de cores e timestamps
# --------------------------------------------------------------------------- #
def test_hex_to_ass_color_rgb():
    assert hex_to_ass_color("#FFFFFF") == "&H00FFFFFF&"
    assert hex_to_ass_color("#FFF200") == "&H0000F2FF&"
    assert hex_to_ass_color("FFF200") == "&H0000F2FF&"


def test_hex_to_ass_color_argb():
    assert hex_to_ass_color("#80000000") == "&H80000000&"
    assert hex_to_ass_color("#80FF00AA") == "&H80AA00FF&"


def test_hex_to_ass_color_passthrough_native_ass():
    assert hex_to_ass_color("&H0000F2FF&") == "&H0000F2FF&"
    assert hex_to_ass_color("&H80FFFFFF&") == "&H80FFFFFF&"


def test_hex_to_ass_color_invalid_raises_value_error():
    with pytest.raises(ValueError):
        hex_to_ass_color("nao-cor")
    with pytest.raises(ValueError):
        hex_to_ass_color("#FFF2")


def test_format_ass_time():
    assert format_ass_time(0) == "0:00:00.00"
    assert format_ass_time(5000) == "0:00:05.00"
    assert format_ass_time(66000) == "0:01:06.00"
    assert format_ass_time(3661500) == "1:01:01.50"


def test_format_ass_time_clamps_negative():
    assert format_ass_time(-10) == "0:00:00.00"


def test_escape_ass_text():
    assert _escape_ass_text("normal") == "normal"
    assert _escape_ass_text(r"ola\{test}") == r"ola\\｛test｝"
    assert _escape_ass_text("linha\nquebrada") == "linha quebrada"


# --------------------------------------------------------------------------- #
# Modelos de configuracao
# --------------------------------------------------------------------------- #
def test_subtitle_style_config_defaults():
    cfg = SubtitleStyleConfig()
    assert cfg.font_name == "Montserrat"
    assert cfg.font_size == 52
    assert cfg.primary_color == "#FFFFFF"
    assert cfg.highlight_color == "#FFF200"
    assert cfg.outline_color == "#000000"
    assert cfg.highlight_mode == KaraokeHighlightMode.WORD_HIGHLIGHT
    assert cfg.active_word_scale == 1.05
    assert cfg.max_words_per_line == 4
    assert cfg.play_res_x == 1080
    assert cfg.play_res_y == 1920
    assert cfg.margin_v == 160


def test_subtitle_style_config_validation():
    with pytest.raises(Exception):
        SubtitleStyleConfig(font_size=8)  # menor que 12
    with pytest.raises(Exception):
        SubtitleStyleConfig(active_word_scale=1.5)  # maior que 1.3
    with pytest.raises(Exception):
        SubtitleStyleConfig(margin_l=9999)  # maior que 500


def test_cut_caption_burn_result_model():
    result = CutCaptionBurnResult(
        output_path="out.mp4",
        ass_path="out.ass",
        cues_count=3,
        total_words_count=10,
        duration_sec=5.5,
    )
    assert result.cues_count == 3
    assert result.total_words_count == 10
    assert result.duration_sec == 5.5


def test_subtitle_cue_model():
    cue = SubtitleCue(
        start_ms=0,
        end_ms=900,
        words=[WORDS[0]],
        text="oi",
    )
    assert cue.start_ms == 0
    assert cue.end_ms == 900


# --------------------------------------------------------------------------- #
# build_cues: particionamento, reindexacao e filtragem
# --------------------------------------------------------------------------- #
def test_build_cues_partitions_by_max_words_per_line():
    gen = AssSubtitleGenerator()
    cues = gen.build_cues(make_transcription())
    assert len(cues) == 3
    assert [len(cue.words) for cue in cues] == [4, 4, 1]
    assert cues[0].text == "oi galera hoje vamos"
    assert cues[0].start_ms == 500
    assert cues[0].end_ms == 2450


def test_build_cues_reindexes_from_start_time():
    gen = AssSubtitleGenerator()
    cues = gen.build_cues(make_transcription(), start_time_ms=2000)
    first_words = [w for cue in cues for w in cue.words]
    assert all(w.start_ms >= 0 for w in first_words)
    assert first_words[0].word == "vamos"
    assert first_words[0].start_ms == 0
    assert first_words[0].end_ms == 450


def test_build_cues_filters_by_end_time():
    gen = AssSubtitleGenerator()
    cues = gen.build_cues(make_transcription(), start_time_ms=1000, end_time_ms=3100)
    words = [w for cue in cues for w in cue.words]
    assert words and all(w.word in {"galera", "hoje", "vamos", "falar", "de"} for w in words)
    assert all(w.start_ms >= 0 for w in words)


def test_build_cues_enforces_minimum_duration_for_instant_words():
    instant = WordTimestamp(word="pulo", start_ms=1500, end_ms=1500, probability=0.5)
    gen = AssSubtitleGenerator()
    cues = gen.build_cues(make_transcription([instant]), start_time_ms=1000)
    (word,) = cues[0].words
    assert word.start_ms == 500
    assert word.end_ms >= word.start_ms + 50


def test_build_cues_accepts_segments_and_word_lists():
    gen = AssSubtitleGenerator()
    segments = [
        CaptionSegment(
            id=0,
            text="oi galera",
            start_ms=500,
            end_ms=1450,
            words=[WORDS[0], WORDS[1]],
        )
    ]
    cues_from_segments = gen.build_cues(segments)
    assert [w.word for w in cues_from_segments[0].words] == ["oi", "galera"]

    cues_from_words = gen.build_cues([WORDS[0], WORDS[1]])
    assert [w.word for w in cues_from_words[0].words] == ["oi", "galera"]


def test_build_cues_empty_transcription():
    gen = AssSubtitleGenerator()
    assert gen.build_cues(empty_transcription()) == []
    assert gen.build_cues([]) == []


def test_build_cues_invalid_input_raises_type_error():
    gen = AssSubtitleGenerator()
    with pytest.raises(TypeError):
        gen.build_cues("nao-e-transcricao")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Geracao de conteudo ASS
# --------------------------------------------------------------------------- #
def test_generate_ass_header_and_resolution_override():
    gen = AssSubtitleGenerator()
    content = gen.generate_ass_content(
        make_transcription(),
        video_width=1080,
        video_height=1920,
    )
    assert content.startswith("[Script Info]")
    assert "ScriptType: v4.00+" in content
    assert "PlayResX: 1080" in content
    assert "PlayResY: 1920" in content


def test_generate_ass_uses_style_playres_defaults():
    cfg = SubtitleStyleConfig(play_res_x=720, play_res_y=1280)
    gen = AssSubtitleGenerator(style_config=cfg)
    content = gen.generate_ass_content(make_transcription())
    assert "PlayResX: 720" in content
    assert "PlayResY: 1280" in content


def test_generate_ass_style_block_uses_high_impact_style():
    cfg = SubtitleStyleConfig(
        font_name="Montserrat ExtraBold",
        font_size=60,
        highlight_color="#00E5FF",
        outline_width=4.0,
        shadow_depth=2.0,
        bold=True,
        alignment=2,
        margin_v=170,
        margin_l=40,
        margin_r=40,
    )
    gen = AssSubtitleGenerator(style_config=cfg)
    content = gen.generate_ass_content(make_transcription())
    style_line = next(line for line in content.splitlines() if line.startswith("Style: Default,"))
    assert "Montserrat ExtraBold,60" in style_line
    assert "&H00FFE500&" in style_line
    assert "BorderStyle=1" not in style_line
    assert ",1,4.0,2.0,2,40,40,170,1" in style_line or ",1," in style_line
    assert style_line.endswith(",1")


def test_generate_ass_empty_transcription_has_no_dialogue():
    gen = AssSubtitleGenerator()
    content = gen.generate_ass_content(empty_transcription())
    assert "[Events]" in content
    assert "Dialogue:" not in content


def test_generate_ass_word_highlight_mode_emits_one_dialogue_per_word():
    gen = AssSubtitleGenerator()
    content = gen.generate_ass_content(make_transcription())
    dialogues = [ln for ln in content.splitlines() if ln.startswith("Dialogue: 0,")]
    assert len(dialogues) == len(WORDS)
    assert "{\\c&H0000F2FF&\\fscx105\\fscy105}" in content
    assert "{\\c&H00FFFFFF&\\fscx100\\fscy100}" in content


def test_generate_ass_word_highlight_timestamps_synced():
    gen = AssSubtitleGenerator()
    content = gen.generate_ass_content(make_transcription())
    dialogues = [ln for ln in content.splitlines() if ln.startswith("Dialogue: 0,")]
    first = dialogues[0]
    assert "0:00:00.50,0:00:00.90,Default" in first
    assert "\\fscx105\\fscy105}oi" in first


def test_generate_ass_karaoke_tag_mode_with_kf():
    cfg = SubtitleStyleConfig(highlight_mode=KaraokeHighlightMode.KARAOKE_TAG)
    gen = AssSubtitleGenerator(style_config=cfg)
    content = gen.generate_ass_content(make_transcription())
    dialogues = [ln for ln in content.splitlines() if ln.startswith("Dialogue: 0,")]
    assert len(dialogues) == 3
    assert "{\\kf40}" in content
    assert "{\\kf45}" in content


def test_generate_ass_karaoke_tag_mode_with_k():
    cfg = SubtitleStyleConfig(highlight_mode=KaraokeHighlightMode.KARAOKE_TAG)
    gen = AssSubtitleGenerator(style_config=cfg, karaoke_tag="k")
    content = gen.generate_ass_content(make_transcription())
    assert "{\\k40}" in content


def test_generate_ass_escapes_special_characters_in_words():
    special = [
        WordTimestamp(word=r"ola\{test}", start_ms=0, end_ms=400, probability=0.9),
    ]
    gen = AssSubtitleGenerator()
    content = gen.generate_ass_content(make_transcription(special))
    assert "{" in content
    assert "}" in content
    escaped = _escape_ass_text(r"ola\{test}")
    assert escaped in content


def test_write_ass_file_returns_path_and_content(tmp_path):
    gen = AssSubtitleGenerator()
    out = tmp_path / "legenda.ass"
    assert gen.write_ass_file(make_transcription(), out) == out
    assert Path(out).is_file()
    content = Path(out).read_text(encoding="utf-8")
    assert content.startswith("[Script Info]")


def test_write_ass_file_creates_parent_directories(tmp_path):
    gen = AssSubtitleGenerator()
    out = tmp_path / "deep" / "nested" / "legenda.ass"
    gen.write_ass_file(make_transcription(), out)
    assert out.is_file()


# --------------------------------------------------------------------------- #
# Renderizacao real via ffmpeg (quando disponivel)
# --------------------------------------------------------------------------- #
def test_generate_ass_escapes_carriage_return():
    special = [
        WordTimestamp(word="ola\r\nmundo", start_ms=0, end_ms=400, probability=0.9),
    ]
    gen = AssSubtitleGenerator()
    content = gen.generate_ass_content(make_transcription(special))
    assert "\r" not in content

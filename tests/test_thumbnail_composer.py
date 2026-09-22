"""Testes do compositor visual de thumbnails 1280x720 (Issue #13).

Cobre composicao ponta a ponta, validacao de densidade lexical, posicionamento
na regra dos tercos, Safe Area do YouTube, variantes de background, compressão
JPEG adaptativa < 2MB e os principais casos de erro especificados.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from video_engine.thumbnail.composer import ThumbnailComposer
from video_engine.thumbnail.models import (
    BackgroundConfig,
    BackgroundType,
    GlowConfig,
    HeadlineConfig,
    SegmentationResult,
    StrokeConfig,
    SubjectPosition,
    ThumbnailConfig,
)

SAFE_AREA_X = 1050
SAFE_AREA_Y = 600
TWO_MB = 2 * 1024 * 1024

# Cores sinteticas bem separadas das demais para analise de pixels pos-JPEG
SUBJECT_RGB = (255, 40, 120)  # magenta
HEADLINE_RGB = (255, 242, 0)  # amarelo #FFF200


def _make_subject(width: int = 300, height: int = 400, color: tuple = SUBJECT_RGB) -> np.ndarray:
    """Gera sujeito RGBA com circulo solido central e fundo transparente."""
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    mask = dist <= radius
    rgba[mask, 0] = color[0]
    rgba[mask, 1] = color[1]
    rgba[mask, 2] = color[2]
    rgba[mask, 3] = 255
    return rgba


def _save_checkerboard_image(path: Path, width: int = 1280, height: int = 720) -> Path:
    """Grava imagem PNG 1280x720 de alto detalhe (checkerboard 6px)."""
    grid = (np.indices((height, width)) // 6).sum(axis=0) % 2
    rgb = np.full((height, width, 3), 30, dtype=np.uint8)
    rgb[grid == 1] = [240, 240, 240]
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG")
    return path


def _count_yellow_pixels(image: np.ndarray) -> int:
    """Conta pixels da faixa amarela da headline no array HxWx3."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return int(np.sum((r > 180) & (g > 180) & (b < 150)))


def _count_magenta_pixels(image: np.ndarray) -> int:
    """Conta pixels do sujeito magenta no array HxWx3."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return int(np.sum((r > 180) & (g < 120) & (b > 60) & (b < 200)))


def _magenta_centroid_x(image: np.ndarray) -> float:
    mask = (image[..., 0] > 180) & (image[..., 1] < 120) & (image[..., 2] > 60) & (image[..., 2] < 200)
    xs = np.where(mask)[1].astype(np.float64)
    return float(xs.mean()) if xs.size else -1.0


def _yellow_centroid_x(image: np.ndarray) -> float:
    mask = (image[..., 0] > 180) & (image[..., 1] > 180) & (image[..., 2] < 150)
    xs = np.where(mask)[1].astype(np.float64)
    return float(xs.mean()) if xs.size else -1.0


class TestComposeEndToEnd:
    def test_generates_1280x720_jpeg_strictly_below_2mb(self, tmp_path: Path) -> None:
        output = tmp_path / "thumb.jpg"
        composer = ThumbnailComposer()
        subject = _make_subject()

        result = composer.compose(
            subject=subject,
            headline=HeadlineConfig(text="VOCE VAI ACREDITAR"),
            output_path=output,
        )

        assert isinstance(result, object)
        assert output.is_file()
        assert result.output_path == str(output)
        assert result.file_size_bytes < TWO_MB
        assert result.file_size_bytes > 0
        assert result.width == 1280
        assert result.height == 720
        assert result.headline == "VOCE VAI ACREDITAR"
        assert result.word_count == 3
        assert result.subject_position == SubjectPosition.RIGHT

        with Image.open(output) as img:
            assert img.size == (1280, 720)
            assert img.format == "JPEG"

    def test_accepts_segmentation_result_and_rgb_array(self, tmp_path: Path) -> None:
        composer = ThumbnailComposer()
        array = _make_subject()

        rgba = array[:, :, :3]
        mask = array[:, :, 3]
        seg = SegmentationResult(
            width=array.shape[1],
            height=array.shape[0],
            foreground_rgba=array,
            alpha_mask=mask,
        )

        out1 = composer.compose(seg, "OFERTA UNICA HOJE", output_path=tmp_path / "a.jpg")
        out2 = composer.compose(rgba, "OFERTA UNICA HOJE", output_path=tmp_path / "b.jpg")

        assert out1.file_size_bytes < TWO_MB
        assert out2.file_size_bytes < TWO_MB

    def test_default_output_path_when_not_informed(self) -> None:
        try:
            composer = ThumbnailComposer()
            result = composer.compose(_make_subject(), "RESULTADO FINAL")
            assert Path(result.output_path).is_file()
            assert Path(result.output_path).suffix == ".jpg"
        finally:
            default = Path.cwd() / "thumbnail_1280x720.jpg"
            if default.is_file():
                default.unlink()


class TestHeadlineWordLimit:
    def test_accepts_3_to_5_words(self, tmp_path: Path) -> None:
        composer = ThumbnailComposer()
        for words in ("TRES PALAVRAS CERTO", "CINCO PALAVRAS COM EXTRA IMPACTO"):
            result = composer.compose(_make_subject(), words, output_path=tmp_path / "ok.jpg")
            assert result.word_count <= 5
            assert result.file_size_bytes < TWO_MB

    def test_rejects_six_words_with_value_error(self, tmp_path: Path) -> None:
        composer = ThumbnailComposer()
        with pytest.raises(ValueError, match="limite recomendado de 5 palavras"):
            HeadlineConfig(text="um dois tres quatro cinco seis")
        with pytest.raises(ValueError, match="limite recomendado de 5 palavras"):
            composer.compose(_make_subject(), "um dois tres quatro cinco seis", output_path=tmp_path / "x.jpg")

    def test_strict_word_limit_false_allows_six_words(self, tmp_path: Path) -> None:
        headline = HeadlineConfig(text="um dois tres quatro cinco seis", strict_word_limit=False)
        result = ThumbnailComposer().compose(
            _make_subject(), headline, output_path=tmp_path / "flex.jpg"
        )
        assert result.word_count == 6
        assert result.file_size_bytes < TWO_MB


class TestSubjectPositioning:
    def test_subject_right_and_text_left(self, tmp_path: Path) -> None:
        output = tmp_path / "right.jpg"
        composer = ThumbnailComposer(
            config=ThumbnailConfig(subject_position=SubjectPosition.RIGHT)
        )
        result = composer.compose(_make_subject(), "VOCE VAI ACREDITAR", output_path=output)

        canvas = np.asarray(Image.open(output).convert("RGB"))
        assert result.subject_position == SubjectPosition.RIGHT
        assert _magenta_centroid_x(canvas) > 640
        assert 0 < _yellow_centroid_x(canvas) < 640

    def test_subject_left_and_text_right(self, tmp_path: Path) -> None:
        output = tmp_path / "left.jpg"
        composer = ThumbnailComposer(
            config=ThumbnailConfig(subject_position=SubjectPosition.LEFT)
        )
        result = composer.compose(_make_subject(), "VOCE VAI ACREDITAR", output_path=output)

        canvas = np.asarray(Image.open(output).convert("RGB"))
        assert result.subject_position == SubjectPosition.LEFT
        assert 0 < _magenta_centroid_x(canvas) < 640
        assert _yellow_centroid_x(canvas) > 640
        assert _count_magenta_pixels(canvas) > 0
        assert _count_yellow_pixels(canvas) > 0


class TestSafeArea:
    def test_layout_never_intersects_bottom_right_quadrant(self) -> None:
        for side in (SubjectPosition.RIGHT, SubjectPosition.LEFT):
            headline = HeadlineConfig(text="CURSO COMPLETO PRA DOMINAR TUDO")
            composer = ThumbnailComposer(config=ThumbnailConfig(subject_position=side))
            layout = composer.compute_headline_layout(headline)
            left, top, right, bottom = layout.bbox

            assert right <= SAFE_AREA_X
            assert not (right > SAFE_AREA_X and bottom > SAFE_AREA_Y)
            assert left >= 0 and top >= 0
            assert layout.font_size >= 24
            assert len(layout.lines) >= 1

    def test_composed_image_has_no_headline_in_bottom_right_quadrant(self, tmp_path: Path) -> None:
        output = tmp_path / "safe.jpg"
        composer = ThumbnailComposer(
            config=ThumbnailConfig(subject_position=SubjectPosition.LEFT)
        )
        composer.compose(
            _make_subject(),
            "CURSO COMPLETO PRA DOMINAR TUDO",
            output_path=output,
        )

        canvas = np.asarray(Image.open(output).convert("RGB"))
        forbidden = canvas[SAFE_AREA_Y:, SAFE_AREA_X:]
        assert _count_yellow_pixels(forbidden) == 0


class TestBackgroundVariants:
    def test_gradient_diagonal_background(self, tmp_path: Path) -> None:
        output = tmp_path / "gradient.jpg"
        background = BackgroundConfig(
            type=BackgroundType.GRADIENT,
            direction="diagonal",
            color_start=(15, 23, 42),
            color_end=(100, 50, 200),
        )
        result = ThumbnailComposer().compose(
            _make_subject(), "FUNDO LEGAL", background=background, output_path=output
        )
        assert result.file_size_bytes < TWO_MB

    def test_solid_background(self, tmp_path: Path) -> None:
        output = tmp_path / "solid.jpg"
        background = BackgroundConfig(type=BackgroundType.SOLID, color_start=(20, 20, 20))
        result = ThumbnailComposer().compose(
            _make_subject(), "COR SOLIDA", background=background, output_path=output
        )
        assert result.file_size_bytes < TWO_MB

    def test_image_background_with_blur_and_darken(self, tmp_path: Path) -> None:
        image_path = tmp_path / "bg_small.png"
        grid = (np.indices((180, 320)) // 10).sum(axis=0) % 2
        rgb = np.full((180, 320, 3), 90, dtype=np.uint8)
        rgb[grid == 1] = [220, 120, 30]
        Image.fromarray(rgb, mode="RGB").save(image_path, format="PNG")

        output = tmp_path / "image_bg.jpg"
        background = BackgroundConfig(
            type=BackgroundType.IMAGE,
            image_path=str(image_path),
            blur_radius=10,
            darken_factor=0.3,
        )
        result = ThumbnailComposer().compose(
            _make_subject(), "IMAGEM TEMATICA", background=background, output_path=output
        )
        assert result.file_size_bytes < TWO_MB
        with Image.open(output) as img:
            assert img.size == (1280, 720)


class TestAdaptiveCompression:
    def test_default_limit_below_2mb_with_high_noise(self, tmp_path: Path) -> None:
        noise = np.random.default_rng(12345).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
        bg_path = tmp_path / "noise.png"
        Image.fromarray(noise, mode="RGB").save(bg_path, format="PNG")

        background = BackgroundConfig(
            type=BackgroundType.IMAGE,
            image_path=str(bg_path),
            blur_radius=0,
            darken_factor=0.0,
        )
        result = ThumbnailComposer().compose(
            _make_subject(), "RUIDO TOTAL", background=background, output_path=tmp_path / "n.jpg"
        )
        assert result.file_size_bytes < TWO_MB
        assert result.file_size_bytes > 0

    def test_adaptive_quality_reduction_forces_size_under_limit(self, tmp_path: Path) -> None:
        bg_path = _save_checkerboard_image(tmp_path / "busy.png")
        background = BackgroundConfig(
            type=BackgroundType.IMAGE,
            image_path=str(bg_path),
            blur_radius=0,
            darken_factor=0.0,
        )

        # Precondicao: na qualidade 90 o checkerboard sozinho ja estoura o teto,
        # portanto o algoritmo obrigatoriamente precisa reduzir a qualidade.
        with Image.open(bg_path) as img:
            img.save(tmp_path / "probe_q90.jpg", format="JPEG", quality=90, optimize=True)
        probe_size = (tmp_path / "probe_q90.jpg").stat().st_size

        limit = 360_000
        assert probe_size > limit

        composer = ThumbnailComposer(config=ThumbnailConfig(max_file_size_bytes=limit))
        result = composer.compose(
            _make_subject(), "BUSY CHECKER", background=background, output_path=tmp_path / "adapt.jpg"
        )
        assert result.file_size_bytes < limit
        assert result.file_size_bytes < probe_size


class TestStrokeAndGlow:
    def test_stroke_glow_applied_around_subject(self, tmp_path: Path) -> None:
        output = tmp_path / "stroked.jpg"
        stroke = StrokeConfig(enabled=True, width=10, color=(255, 255, 255), opacity=1.0)
        glow = GlowConfig(enabled=True, radius=20, color=(80, 160, 255), intensity=0.8)

        plain = tmp_path / "plain.jpg"
        ThumbnailComposer().compose(_make_subject(), "SEM EFEITO", output_path=plain)
        ThumbnailComposer().compose(
            _make_subject(), "COM EFEITO", stroke=stroke, glow=glow, output_path=output
        )

        plain_arr = np.asarray(Image.open(plain).convert("RGB"))
        stroked_arr = np.asarray(Image.open(output).convert("RGB"))

        white = (plain_arr[..., 0] > 240) & (plain_arr[..., 1] > 240) & (plain_arr[..., 2] > 240)
        white_stroked = (
            (stroked_arr[..., 0] > 240) & (stroked_arr[..., 1] > 240) & (stroked_arr[..., 2] > 240)
        )
        assert int(np.sum(white_stroked)) > int(np.sum(white))


class TestErrors:
    def test_missing_subject_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="sujeito nao encontrado"):
            ThumbnailComposer().compose(tmp_path / "nao_existe.png", "SEM SUJEITO")

    def test_missing_background_image_raises_file_not_found(self, tmp_path: Path) -> None:
        background = BackgroundConfig(
            type=BackgroundType.IMAGE, image_path=str(tmp_path / "nao_existe.jpg")
        )
        with pytest.raises(FileNotFoundError, match="background nao encontrada"):
            ThumbnailComposer().compose(
                _make_subject(), "SEM FUNDO", background=background, output_path=tmp_path / "f.jpg"
            )

    def test_image_background_without_path_raises_value_error(self, tmp_path: Path) -> None:
        background = BackgroundConfig(type=BackgroundType.IMAGE, image_path=None)
        with pytest.raises(ValueError, match="requer image_path"):
            ThumbnailComposer().compose(
                _make_subject(), "SEM CAMINHO", background=background, output_path=tmp_path / "g.jpg"
            )

    def test_invalid_subject_array_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="HxWx3"):
            ThumbnailComposer().compose(
                np.zeros((100, 100), dtype=np.uint8), "INVALIDO", output_path=tmp_path / "h.jpg"
            )

    def test_invalid_headline_type_raises_type_error(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="str ou HeadlineConfig"):
            ThumbnailComposer().compose(
                _make_subject(), 123, output_path=tmp_path / "i.jpg"
            )


class TestComposeFromFrame:
    def _make_frame(self, width: int = 1280, height: int = 720) -> np.ndarray:
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[:, : width // 2] = [70, 70, 210]
        rgb[:, width // 2 :] = [210, 120, 40]
        return rgb

    def test_compose_from_frame_creates_1280x720_jpeg(self, tmp_path: Path) -> None:
        output = tmp_path / "from_frame.jpg"
        composer = ThumbnailComposer()
        frame = self._make_frame(960, 540)

        result = composer.compose_from_frame(frame, "OFERTA DO DIA", output_path=output)

        assert output.is_file()
        assert result.output_path == str(output)
        assert result.file_size_bytes < TWO_MB
        assert result.file_size_bytes > 0
        assert result.width == 1280
        assert result.height == 720
        with Image.open(output) as img:
            assert img.size == (1280, 720)
            assert img.format == "JPEG"

    def test_compose_from_frame_applies_headline(self, tmp_path: Path) -> None:
        output = tmp_path / "headline.jpg"

        result = ThumbnailComposer().compose_from_frame(
            self._make_frame(), "VOCE VAI ACREDITAR", output_path=output
        )

        assert result.headline == "VOCE VAI ACREDITAR"
        assert result.word_count == 3
        canvas = np.asarray(Image.open(output).convert("RGB"))
        assert _count_yellow_pixels(canvas) > 0

    def test_compose_from_frame_accepts_image_path(self, tmp_path: Path) -> None:
        frame_path = tmp_path / "frame.png"
        Image.fromarray(self._make_frame(), mode="RGB").save(frame_path, format="PNG")

        result = ThumbnailComposer().compose_from_frame(
            frame_path, "PATH ACEITO", output_path=tmp_path / "path.jpg"
        )

        assert Path(result.output_path).is_file()
        assert result.file_size_bytes < TWO_MB

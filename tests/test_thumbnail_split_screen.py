"""Testes do layout Split-Screen 50/50 na geracao de thumbnails (Issue #23).

Valida conformidade com a Spec da Issue #23:
- Divisao simetrica em dois paineis de 640x720 (totalizando 1280x720).
- Painel de frame com center-crop e sem distorcao.
- Painel de texto com fundo institucional de alto contraste e headline renderizada.
- Respeito a Safe Area do YouTube (canto inferior direito x > 1050 e y > 600).
- Suporte a divisoria configuravel.
- Exportacao em JPEG com tamanho estritamente < 2 MB.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from PIL import Image

from video_engine.thumbnail.composer import ThumbnailComposer
from video_engine.thumbnail.models import (
    BackgroundConfig,
    BackgroundType,
    HeadlineConfig,
    SplitScreenConfig,
    TextPanelSide,
    ThumbnailCompositionResult,
)

TWO_MB = 2 * 1024 * 1024
SAFE_AREA_X = 1050
SAFE_AREA_Y = 600

# Cores sinteticas para analise discriminativa de paineis
FRAME_COLOR = (45, 120, 210)  # azul
TEXT_BG_START = (15, 23, 42)  # slate escuro
TEXT_COLOR = (255, 242, 0)    # amarelo #FFF200


def _make_test_frame(
    width: int = 1280,
    height: int = 720,
    color: Tuple[int, int, int] = FRAME_COLOR,
) -> np.ndarray:
    """Gera um frame sintetico RGB solido com dimensao configuravel."""
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:, :] = color
    return arr


def _count_yellow_pixels(image: np.ndarray) -> int:
    """Conta pixels da faixa amarela da headline no array HxWx3."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return int(np.sum((r > 200) & (g > 200) & (b < 100)))


def _count_blue_pixels(image: np.ndarray) -> int:
    """Conta pixels do frame azul de teste no array HxWx3."""
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return int(np.sum((r < 80) & (g > 90) & (g < 150) & (b > 180)))


class TestThumbnailSplitScreen:
    def test_compose_split_screen_dimensions_and_file_size(self, tmp_path: Path) -> None:
        """Gera JPEG 1280x720 e valida teto de 2MB conforme especificacao."""
        output = tmp_path / "thumb_split.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1920, 1080)

        result = composer.compose_split_screen(
            frame=frame,
            headline=HeadlineConfig(text="SEGREDO DO YOUTUBE"),
            output_path=output,
        )

        assert isinstance(result, ThumbnailCompositionResult)
        assert output.is_file()
        assert result.output_path == str(output)
        assert result.width == 1280
        assert result.height == 720
        assert 0 < result.file_size_bytes < TWO_MB
        assert result.headline == "SEGREDO DO YOUTUBE"
        assert result.word_count == 3
        assert result.layout_mode == "split_screen"

        with Image.open(output) as img:
            assert img.size == (1280, 720)
            assert img.format == "JPEG"

    def test_compose_split_screen_panels_integrity_text_left(self, tmp_path: Path) -> None:
        """Com texto a esquerda (padrao): [0, 640] tem texto e [640, 1280] tem frame."""
        output = tmp_path / "split_left.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1280, 720, color=FRAME_COLOR)

        composer.compose_split_screen(
            frame=frame,
            headline=HeadlineConfig(text="ESTRATEGIA TOTAL"),
            split_config=SplitScreenConfig(text_side=TextPanelSide.LEFT),
            output_path=output,
        )

        img = np.asarray(Image.open(output).convert("RGB"))
        left_panel = img[:, :640, :]
        right_panel = img[:, 640:, :]

        # Painel esquerdo deve conter a headline amarela
        assert _count_yellow_pixels(left_panel) > 50

        # Painel direito deve conter a maioria dos pixels azuis do frame
        assert _count_blue_pixels(right_panel) > (640 * 720 * 0.7)

        # O frame azul nao deve contaminar o painel de texto esquerdo
        assert _count_blue_pixels(left_panel) < 500

    def test_compose_split_screen_panels_integrity_text_right(self, tmp_path: Path) -> None:
        """Com texto a direita: [0, 640] tem frame e [640, 1280] tem texto."""
        output = tmp_path / "split_right.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1280, 720, color=FRAME_COLOR)

        composer.compose_split_screen(
            frame=frame,
            headline=HeadlineConfig(text="ESTRATEGIA TOTAL"),
            split_config=SplitScreenConfig(text_side=TextPanelSide.RIGHT),
            output_path=output,
        )

        img = np.asarray(Image.open(output).convert("RGB"))
        left_panel = img[:, :640, :]
        right_panel = img[:, 640:, :]

        # Painel direito deve conter a headline amarela
        assert _count_yellow_pixels(right_panel) > 50

        # Painel esquerdo deve conter o frame azul
        assert _count_blue_pixels(left_panel) > (640 * 720 * 0.7)

        # O frame azul nao deve contaminar o painel de texto direito
        assert _count_blue_pixels(right_panel) < 500

    def test_compose_split_screen_text_left_inherently_respects_safe_area(
        self, tmp_path: Path
    ) -> None:
        """Com texto no painel esquerdo ([0, 640]), a Safe Area (x > 1050, y > 600) e 100% imune a texto."""
        output = tmp_path / "safe_area_left.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1280, 720)

        composer.compose_split_screen(
            frame=frame,
            headline=HeadlineConfig(text="PROVA REAL AGORA"),
            split_config=SplitScreenConfig(text_side=TextPanelSide.LEFT),
            output_path=output,
        )

        img = np.asarray(Image.open(output).convert("RGB"))
        safe_area_zone = img[SAFE_AREA_Y:, SAFE_AREA_X:, :]

        # Nenhum pixel amarelo da headline pode existir na Safe Area
        assert _count_yellow_pixels(safe_area_zone) == 0

    def test_compose_split_screen_text_right_respects_safe_area(self, tmp_path: Path) -> None:
        """Com texto a direita, o layout tipografico nao invade a Safe Area do YouTube."""
        output = tmp_path / "safe_area_right.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1280, 720)

        composer.compose_split_screen(
            frame=frame,
            headline=HeadlineConfig(text="CLIQUE AQUI AGORA"),
            split_config=SplitScreenConfig(text_side=TextPanelSide.RIGHT),
            output_path=output,
        )

        img = np.asarray(Image.open(output).convert("RGB"))
        safe_area_zone = img[SAFE_AREA_Y:, SAFE_AREA_X:, :]

        # Nenhum pixel amarelo da headline na Safe Area
        assert _count_yellow_pixels(safe_area_zone) == 0

    def test_compose_split_screen_with_divider(self, tmp_path: Path) -> None:
        """Verifica a inclusao de linha divisoria visivel entre os paineis em x=640."""
        output = tmp_path / "divider.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1280, 720)

        composer.compose_split_screen(
            frame=frame,
            headline="COM DIVISORIA",
            split_config=SplitScreenConfig(
                divider_width=6,
                divider_color=(255, 255, 255),
            ),
            output_path=output,
        )

        img = np.asarray(Image.open(output).convert("RGB"))
        # Na divisoria (x em torno de 640), os pixels devem ser brancos
        divider_strip = img[100:600, 638:642, :]
        # Media dos canais proxima de branco (> 230)
        assert np.mean(divider_strip) > 200

    @pytest.mark.parametrize(
        ("frame_w", "frame_h"),
        [
            (1920, 1080),  # 16:9 tradicional
            (1080, 1920),  # 9:16 vertical (Shorts)
            (800, 600),    # 4:3 classico
            (2560, 1080),  # 21:9 ultrawide
        ],
    )
    def test_compose_split_screen_handles_various_frame_aspect_ratios(
        self, tmp_path: Path, frame_w: int, frame_h: int
    ) -> None:
        """Garante que qualquer aspect ratio do frame de entrada e cortado sem distorcao para 640x720."""
        output = tmp_path / f"aspect_{frame_w}x{frame_h}.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(frame_w, frame_h)

        result = composer.compose_split_screen(
            frame=frame,
            headline="PROPORCAO PERFEITA",
            output_path=output,
        )

        assert result.width == 1280
        assert result.height == 720
        with Image.open(output) as img:
            assert img.size == (1280, 720)

    def test_compose_split_screen_accepts_file_path(self, tmp_path: Path) -> None:
        """Aceita str ou Path apontando para o arquivo de imagem do frame."""
        frame_file = tmp_path / "raw_frame.png"
        Image.fromarray(_make_test_frame(1280, 720), mode="RGB").save(frame_file)

        output = tmp_path / "from_path.jpg"
        composer = ThumbnailComposer()

        result = composer.compose_split_screen(
            frame=frame_file,
            headline="CARREGADO DO ARQUIVO",
            output_path=output,
        )

        assert Path(result.output_path).is_file()

    def test_compose_split_screen_missing_frame_file_raises_error(self, tmp_path: Path) -> None:
        """Levanta FileNotFoundError se o caminho do frame nao existir."""
        composer = ThumbnailComposer()
        with pytest.raises(FileNotFoundError, match="Arquivo do frame"):
            composer.compose_split_screen(
                frame=tmp_path / "inexistente.png",
                headline="ERRO ESPERADO",
            )

    def test_compose_split_screen_custom_background(self, tmp_path: Path) -> None:
        """Permite customizacao do fundo do painel de texto via BackgroundConfig."""
        output = tmp_path / "custom_bg.jpg"
        composer = ThumbnailComposer()
        frame = _make_test_frame(1280, 720)
        bg = BackgroundConfig(
            type=BackgroundType.SOLID,
            color_start=(80, 20, 20),
        )

        result = composer.compose_split_screen(
            frame=frame,
            headline="FUNDO CUSTOM",
            split_config=SplitScreenConfig(background=bg),
            output_path=output,
        )

        assert Path(result.output_path).is_file()

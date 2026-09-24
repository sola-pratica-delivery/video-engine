"""Testes da correcao geometrica de layout com Zero Overlap na Thumbnail (Issue #22).

Garante que o texto renderizado pelo ThumbnailComposer nunca sobreponha
a area do apresentador/sujeito recortado, respeitando a regra dos tercos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from video_engine.thumbnail.composer import ThumbnailComposer
from video_engine.thumbnail.models import (
    HeadlineConfig,
    SegmentationResult,
    SubjectPosition,
    ThumbnailConfig,
)


def _make_dummy_subject(width: int = 400, height: int = 600) -> SegmentationResult:
    """Cria um sujeito sintetico retangular magenta opaco."""
    fg = np.zeros((height, width, 4), dtype=np.uint8)
    fg[:, :] = [255, 0, 255, 255]
    mask = np.full((height, width), 255, dtype=np.uint8)
    return SegmentationResult(
        width=width,
        height=height,
        foreground_rgba=fg,
        alpha_mask=mask,
    )


class TestZeroOverlapLayout:
    def test_compute_headline_layout_respects_subject_dest_x_on_right(self) -> None:
        composer = ThumbnailComposer(
            config=ThumbnailConfig(
                width=1280,
                height=720,
                subject_position=SubjectPosition.RIGHT,
                subject_margin_x=40,
            )
        )
        headline = HeadlineConfig(text="CURSO COMPLETO PRA DOMINAR TUDO", font_size=72)
        # Sujeito posicionado em dest_x = 760
        subject_dest_x = 760
        margin = 40

        layout = composer.compute_headline_layout(
            headline,
            subject_dest_x=subject_dest_x,
        )

        left, top, right, bottom = layout.bbox
        # A borda direita da caixa de texto DEVE ser estritamente menor ou igual a subject_dest_x - margin
        assert right <= subject_dest_x - margin
        assert right <= 720

    def test_compute_headline_layout_respects_subject_dest_x_on_left(self) -> None:
        composer = ThumbnailComposer(
            config=ThumbnailConfig(
                width=1280,
                height=720,
                subject_position=SubjectPosition.LEFT,
                subject_margin_x=40,
            )
        )
        headline = HeadlineConfig(text="CURSO COMPLETO PRA DOMINAR TUDO", font_size=72)
        # Sujeito na esquerda ocupando x=[40, 440]
        subject_dest_x = 40
        subject_width = 400
        margin = 40

        layout = composer.compute_headline_layout(
            headline,
            subject_dest_x=subject_dest_x,
            subject_width=subject_width,
        )

        left, top, right, bottom = layout.bbox
        # A borda esquerda do texto DEVE comecar apos o sujeito + margin
        assert left >= subject_dest_x + subject_width + margin
        assert left >= 480

    def test_compose_zero_overlap_pixels(self, tmp_path: Path) -> None:
        output_path = tmp_path / "zero_overlap.jpg"
        composer = ThumbnailComposer(
            config=ThumbnailConfig(
                width=1280,
                height=720,
                subject_position=SubjectPosition.RIGHT,
                subject_margin_x=40,
            )
        )
        subject = _make_dummy_subject(width=450, height=600)
        # Headline chamativa amarela (#FFF200) com contorno preto
        headline = HeadlineConfig(
            text="O SEGREDO REVELADO",
            text_color=(255, 242, 0),
            stroke_width=0,  # sem stroke para verificar estritamente o amarelo
        )

        _ = composer.compose(
            subject=subject,
            headline=headline,
            output_path=output_path,
        )


        # Carrega a imagem composta
        img = np.asarray(Image.open(output_path).convert("RGB"))

        # Determina a area ocupada pelo sujeito na composicao
        # dest_x = 1280 - scaled_w - 40
        # scaled_h = int(round(0.88 * 720)) = 634
        # scale = 634 / 600 = 1.0567
        # scaled_w = int(round(450 * 1.0567)) = 476
        # dest_x = 1280 - 476 - 40 = 764
        # dest_y = 720 - 634 = 86
        # Qualquer pixel com x >= dest_x pertence a regiao horizontal do sujeito
        # Verifica que NENHUM pixel amarelo da headline foi desenhado dentro da regiao do sujeito
        dest_x, dest_y = composer._place_subject(
            composer._scale_subject(Image.fromarray(subject.foreground_rgba), 720).size,
            1280,
            720,
        )

        subject_region = img[:, dest_x:, :]
        # Pixel amarelo puro da headline: R > 200, G > 200, B < 50
        yellow_mask = (
            (subject_region[:, :, 0] > 200)
            & (subject_region[:, :, 1] > 200)
            & (subject_region[:, :, 2] < 50)
        )
        yellow_pixel_count = int(np.sum(yellow_mask))
        assert yellow_pixel_count == 0, f"Encontrados {yellow_pixel_count} pixels da headline sobrepostos ao sujeito!"

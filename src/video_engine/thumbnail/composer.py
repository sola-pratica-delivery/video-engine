"""Motor de composicao visual de thumbnails 1280x720 (Issue #13).

Sintetiza a imagem final do Epic #10 combinando background tematico
(gradiente, cor solida ou imagem com blur), sujeito segmentado da Issue #12
(escalado e ancorado ao terco lateral) e headline tipografica de alto CTR
com respeito incondicional a Safe Area do YouTube (canto inferior direito)
e exportacao JPEG adaptatica estritamente inferior a 2MB.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from video_engine.thumbnail.edge_enhancer import apply_stroke_and_glow
from video_engine.thumbnail.models import (
    BackgroundConfig,
    BackgroundType,
    GlowConfig,
    HeadlineConfig,
    SegmentationResult,
    StrokeConfig,
    SubjectPosition,
    ThumbnailCompositionResult,
    ThumbnailConfig,
)

# Zona de exclusao do timestamp nativo do YouTube (x > 1050 e y > 600).
SAFE_AREA_X = 1050
SAFE_AREA_Y = 600

_MEASURE = ImageDraw.Draw(Image.new("RGB", (1, 1)))


@dataclass(frozen=True)
class HeadlineLayout:
    """Layout deterministico da headline: linhas quebradas e caixa delimitadora.

    A caixa ``bbox`` e ``bbox[2] < SAFE_AREA_X e bbox[3] < SAFE_AREA_Y``
    garantida pela gramatica de composicao.
    """

    lines: Tuple[str, ...]
    font_size: int
    bbox: Tuple[int, int, int, int]
    line_boxes: Tuple[Tuple[int, int, int, int], ...]


def _load_font(size: int) -> ImageFont.ImageFont:
    """Carrega fonte bold deterministica, com fallback embutido offline."""
    for name in ("arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


class ThumbnailComposer:
    """Compositor multicamada de thumbnails 1280x720 (16:9).

    Args:
        config: Parametros gerais de layout e exportacao.
    """

    def __init__(self, config: Optional[ThumbnailConfig] = None) -> None:
        self.config = config or ThumbnailConfig()

    def compose(
        self,
        subject: Union[SegmentationResult, np.ndarray, str, Path],
        headline: Union[str, HeadlineConfig],
        background: Optional[BackgroundConfig] = None,
        stroke: Optional[StrokeConfig] = None,
        glow: Optional[GlowConfig] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> ThumbnailCompositionResult:
        """Executa a composicao completa e grava arquivo JPEG 1280x720 < 2MB.

        Raises:
            FileNotFoundError: se o arquivo do sujeito ou do background nao existir.
            ValueError: se a headline violar o limite estrito de palavras.
        """
        if isinstance(headline, str):
            headline = HeadlineConfig(text=headline)
        elif not isinstance(headline, HeadlineConfig):
            raise TypeError("headline deve ser str ou HeadlineConfig")

        background_cfg = background or BackgroundConfig()
        subject_img = self._load_subject_image(subject)

        if stroke or glow:
            enhanced = apply_stroke_and_glow(np.asarray(subject_img), stroke=stroke, glow=glow)
            subject_img = Image.fromarray(enhanced, mode="RGBA")

        width, height = self.config.width, self.config.height
        bg = self._render_background(background_cfg).convert("RGBA")

        subject_img = self._scale_subject(subject_img, height)
        dest_x, dest_y = self._place_subject(subject_img.size, width, height)

        canvas = bg.copy()
        canvas.alpha_composite(subject_img, dest=(dest_x, dest_y))

        layout = self.compute_headline_layout(
            headline,
            width,
            height,
            subject_position=self.config.subject_position,
            subject_dest_x=dest_x,
            subject_width=subject_img.width,
        )
        self._draw_headline(canvas, layout, headline)

        canvas_rgb = canvas.convert("RGB")
        out = Path(output_path) if output_path else Path.cwd() / "thumbnail_1280x720.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        file_size = self._export_jpeg(canvas_rgb, out)

        return ThumbnailCompositionResult(
            output_path=str(out),
            file_size_bytes=file_size,
            width=width,
            height=height,
            headline=headline.text,
            word_count=len(headline.text.split()),
            subject_position=self.config.subject_position,
        )

    def compose_from_frame(
        self,
        frame: Union[np.ndarray, str, Path],
        headline: Union[str, HeadlineConfig],
        output_path: Optional[Union[str, Path]] = None,
        darken_factor: float = 0.15,
    ) -> ThumbnailCompositionResult:
        """Composicao de fallback: frame bruto com crop 1280x720 e headline.

        Usado quando o segmentador falha ou nao ha apresentador segmentavel:
        o keyframe original e recortado cobrindo integralmente o canvas
        (cover-crop 16:9), leve escurecimento para contraste tipografico e a
        headline renderizada respeitando a Safe Area do YouTube.

        Args:
            frame: Frame RGB (HxWx3 / HxWx4 uint8), ndarray ou arquivo de imagem.
            headline: Texto ou ``HeadlineConfig`` da capa.
            output_path: Destino do JPEG 1280x720 (< 2MB).
            darken_factor: Escurecimento aplicado ao frame (0.0 = sem escurecer).

        Raises:
            FileNotFoundError: se ``frame`` for caminho inexistente.
            TypeError: se ``frame``/``headline`` forem de tipagem invalida.
        """
        if isinstance(headline, str):
            headline = HeadlineConfig(text=headline)
        elif not isinstance(headline, HeadlineConfig):
            raise TypeError("headline deve ser str ou HeadlineConfig")

        width, height = self.config.width, self.config.height
        base = self._cover_crop(self._load_frame_rgb(frame), width, height).convert("RGB")
        if darken_factor > 0:
            arr = np.asarray(base, dtype=np.float32) * (1.0 - darken_factor)
            base = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")

        canvas = base.convert("RGBA")
        layout = self.compute_headline_layout(headline, width, height)
        self._draw_headline(canvas, layout, headline)

        canvas_rgb = canvas.convert("RGB")
        out = Path(output_path) if output_path else Path.cwd() / "thumbnail_from_frame.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        file_size = self._export_jpeg(canvas_rgb, out)

        return ThumbnailCompositionResult(
            output_path=str(out),
            file_size_bytes=file_size,
            width=width,
            height=height,
            headline=headline.text,
            word_count=len(headline.text.split()),
            subject_position=self.config.subject_position,
        )

    def compute_headline_layout(
        self,
        headline: HeadlineConfig,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
        subject_position: Optional[SubjectPosition] = None,
        subject_dest_x: Optional[int] = None,
        subject_width: Optional[int] = None,
    ) -> HeadlineLayout:
        """Calcula o layout deterministico da headline sem colisao com o apresentador e respeitando a Safe Area."""
        width = canvas_width or self.config.width
        height = canvas_height or self.config.height
        side = subject_position or self.config.subject_position

        text = headline.text.upper() if headline.all_caps else headline.text
        margin = 40

        if side == SubjectPosition.RIGHT:
            zone_left = margin
            if subject_dest_x is not None:
                max_right = subject_dest_x - margin
            else:
                max_right = int(width * (2 / 3)) - margin
            zone_right = min(SAFE_AREA_X, width - margin, max_right)
            zone_right = max(zone_right, zone_left + 1)
        else:
            if subject_dest_x is not None and subject_width is not None:
                min_left = subject_dest_x + subject_width + margin
            else:
                min_left = int(width * 0.34)
            zone_left = max(margin, min_left)
            zone_right = min(SAFE_AREA_X, width) - margin
            zone_right = max(zone_right, zone_left + 1)

        zone_width = max(zone_right - zone_left, 1)

        zone_top = int(height * 0.12)
        zone_bottom = int(height * 0.62)
        zone_height = max(zone_bottom - zone_top, 1)


        font, lines, font_size = self._fit_headline(
            text,
            zone_width=zone_width,
            zone_height=zone_height,
            max_font_size=headline.font_size,
        )

        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        spacing = max(1, int(line_height * 0.18))
        block_height = line_height * len(lines) + spacing * (len(lines) - 1)

        top = zone_top + (zone_height - block_height) // 2
        if top < 0:
            top = 0

        line_boxes: list[Tuple[int, int, int, int]] = []
        for index, line in enumerate(lines):
            line_width = int(math.ceil(_MEASURE.textlength(line, font=font)))
            line_left = zone_left + (zone_width - line_width) // 2
            line_top = top + index * (line_height + spacing)
            line_boxes.append((line_left, line_top, line_left + line_width, line_top + line_height))

        bbox = self._aggregate_bbox(line_boxes)

        # Garantia incondicional: nenhum texto pode invadir x > 1050 e y > 600.
        if bbox[2] > SAFE_AREA_X and bbox[3] > SAFE_AREA_Y:
            shift_up = bbox[3] - SAFE_AREA_Y
            line_boxes = [(left, t - shift_up, r, b - shift_up) for left, t, r, b in line_boxes]
            bbox = self._aggregate_bbox(line_boxes)

        return HeadlineLayout(
            lines=tuple(lines),
            font_size=font_size,
            bbox=bbox,
            line_boxes=tuple(line_boxes),
        )

    def _draw_headline(
        self,
        canvas: Image.Image,
        layout: HeadlineLayout,
        headline: HeadlineConfig,
    ) -> None:
        """Desenha a headline com drop shadow e contorno ultra-espesso."""
        draw = ImageDraw.Draw(canvas)
        font = _load_font(layout.font_size)

        for line, box in zip(layout.lines, layout.line_boxes):
            shadow_x = box[0] + headline.shadow_offset[0]
            shadow_y = box[1] + headline.shadow_offset[1]
            draw.text(
                (shadow_x, shadow_y),
                line,
                font=font,
                fill=headline.shadow_color,
                stroke_width=headline.stroke_width,
                stroke_fill=headline.shadow_color,
            )
            draw.text(
                (box[0], box[1]),
                line,
                font=font,
                fill=headline.text_color,
                stroke_width=headline.stroke_width,
                stroke_fill=headline.stroke_color,
            )

    def _fit_headline(
        self,
        text: str,
        zone_width: int,
        zone_height: int,
        max_font_size: int,
    ) -> Tuple[ImageFont.ImageFont, list[str], int]:
        """Encontra tamanho de fonte e quebra de linhas que cabem na zona."""
        size = max_font_size
        min_size = 36
        while size >= min_size:
            font = _load_font(size)
            lines = self._wrap_lines(text, font, zone_width)
            ascent, descent = font.getmetrics()
            line_height = ascent + descent
            spacing = max(1, int(line_height * 0.18))
            block_height = line_height * len(lines) + spacing * (len(lines) - 1)
            widest = max((_MEASURE.textlength(line, font=font) for line in lines), default=0)
            if widest <= zone_width and block_height <= zone_height:
                return font, lines, size
            size -= 2

        font = _load_font(min_size)
        return font, self._wrap_lines(text, font, zone_width), min_size

    @staticmethod
    def _wrap_lines(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        """Quebra de linhas gananciosa respeitando a largura maxima."""
        words = text.split()
        if not words:
            return [""]
        lines: list[str] = []
        current = ""
        for word in words:
            trial = word if not current else f"{current} {word}"
            if not current or _MEASURE.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines

    def _load_subject_image(self, subject: Union[SegmentationResult, np.ndarray, str, Path]) -> Image.Image:
        """Carrega o sujeito como PIL RGBA a partir dos formatos aceitos."""
        if isinstance(subject, SegmentationResult):
            array = subject.foreground_rgba
        elif isinstance(subject, np.ndarray):
            array = subject
        elif isinstance(subject, (str, Path)):
            path = Path(subject)
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo do sujeito nao encontrado: {path}")
            return Image.open(path).convert("RGBA")
        else:
            raise TypeError("subject deve ser SegmentationResult, ndarray, str ou Path")

        array = np.asarray(array)
        if array.ndim != 3:
            raise ValueError("Sujeito deve ser array HxWx3 (RGB) ou HxWx4 (RGBA)")
        if array.shape[2] == 4:
            return Image.fromarray(array.astype(np.uint8), mode="RGBA")
        if array.shape[2] == 3:
            rgb = array.astype(np.uint8)
            rgba = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
            return Image.fromarray(rgba, mode="RGBA")
        raise ValueError("Sujeito deve ter 3 ou 4 canais")

    @staticmethod
    def _load_frame_rgb(frame: Union[np.ndarray, str, Path]) -> Image.Image:
        """Carrega um frame/foto como PIL RGB a partir dos formatos aceitos."""
        if isinstance(frame, np.ndarray):
            array = np.asarray(frame)
            if array.ndim == 2:
                array = np.dstack([array, array, array])
            elif array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError("Frame deve ser array HxWx3 (RGB) ou HxWx4 (RGBA)")
            return Image.fromarray(array[:, :, :3].astype(np.uint8), mode="RGB")
        if isinstance(frame, (str, Path)):
            path = Path(frame)
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo do frame nao encontrado: {path}")
            return Image.open(path).convert("RGB")
        raise TypeError("frame deve ser ndarray, str ou Path")

    def _render_background(self, background: BackgroundConfig) -> Image.Image:
        """Renderiza a camada 1 (background) em 1280x720."""
        width, height = self.config.width, self.config.height
        if background.type == BackgroundType.GRADIENT:
            return self._render_gradient(background, width, height)
        if background.type == BackgroundType.SOLID:
            return Image.new("RGB", (width, height), tuple(background.color_start))
        if background.type == BackgroundType.IMAGE:
            if not background.image_path:
                raise ValueError("BackgroundConfig(type=IMAGE) requer image_path")
            path = Path(background.image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Imagem de background nao encontrada: {path}")
            image = Image.open(path).convert("RGB")
            image = self._cover_crop(image, width, height)
            if background.blur_radius > 0:
                image = image.filter(ImageFilter.GaussianBlur(radius=background.blur_radius))
            if background.darken_factor > 0:
                arr = np.asarray(image, dtype=np.float32) * (1.0 - background.darken_factor)
                image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
            return image
        raise ValueError(f"BackgroundType desconhecido: {background.type}")

    @staticmethod
    def _render_gradient(background: BackgroundConfig, width: int, height: int) -> Image.Image:
        """Gera gradiente cinematografico (horizontal, vertical ou diagonal)."""
        y, x = np.mgrid[0:height, 0:width].astype(np.float64)
        denom_x = max(width - 1, 1)
        denom_y = max(height - 1, 1)
        if background.direction == "horizontal":
            t = x / denom_x
        elif background.direction == "vertical":
            t = y / denom_y
        else:
            t = (x / denom_x + y / denom_y) / 2.0

        start = np.array(background.color_start, dtype=np.float64)[np.newaxis, np.newaxis, :]
        end = np.array(background.color_end, dtype=np.float64)[np.newaxis, np.newaxis, :]
        rgb = start * (1.0 - t[..., np.newaxis]) + end * t[..., np.newaxis]
        return Image.fromarray(np.clip(np.round(rgb), 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _cover_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
        """Redimensiona e recorta no centro cobrindo integralmente 16:9."""
        img_w, img_h = image.size
        scale = max(target_w / max(img_w, 1), target_h / max(img_h, 1))
        new_w = max(1, int(round(img_w * scale)))
        new_h = max(1, int(round(img_h * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))

    def _scale_subject(self, subject: Image.Image, canvas_height: int) -> Image.Image:
        """Escala proporcionalmente o sujeito para o percentual configurado."""
        sub_w, sub_h = subject.size
        target_h = max(1, int(round(self.config.subject_scale * canvas_height)))
        scale = target_h / max(sub_h, 1)
        new_w = max(1, int(round(sub_w * scale)))
        return subject.resize((new_w, target_h), Image.Resampling.LANCZOS)

    def _place_subject(self, subject_size: Tuple[int, int], width: int, height: int) -> Tuple[int, int]:
        """Ancora o sujeito ao terco lateral e a base da tela."""
        sub_w, sub_h = subject_size
        if self.config.subject_position == SubjectPosition.RIGHT:
            dest_x = max(0, width - sub_w - self.config.subject_margin_x)
        else:
            dest_x = min(max(0, self.config.subject_margin_x), max(0, width - sub_w))
        dest_y = height - sub_h
        return dest_x, dest_y

    @staticmethod
    def _aggregate_bbox(
        line_boxes: list[Tuple[int, int, int, int]],
    ) -> Tuple[int, int, int, int]:
        left = min(box[0] for box in line_boxes)
        top = min(box[1] for box in line_boxes)
        right = max(box[2] for box in line_boxes)
        bottom = max(box[3] for box in line_boxes)
        return left, top, right, bottom

    def _export_jpeg(self, image: Image.Image, output: Path) -> int:
        """Grava JPEG com qualidade adaptativa garantindo o teto configurado.

        Recalcula a compressao em patamares decrescentes (``quality -= 5``)
        ate o arquivo ficar estritamente abaixo de ``max_file_size_bytes``.
        """
        quality = self.config.jpeg_quality
        size = 0
        while True:
            image.save(output, format="JPEG", quality=quality, optimize=True)
            size = output.stat().st_size
            if size < self.config.max_file_size_bytes or quality <= 50:
                break
            quality -= 5
        if size >= self.config.max_file_size_bytes:
            raise ValueError(
                f"JPEG excedeu o teto de {self.config.max_file_size_bytes} bytes "
                f"mesmo na qualidade minima (50): {size} bytes em {output}"
            )
        return size


__all__ = ["HeadlineLayout", "ThumbnailComposer"]

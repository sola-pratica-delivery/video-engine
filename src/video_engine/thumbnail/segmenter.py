"""Motor de segmentacao semantica e remocao de fundo (Issue #12).

Implementa ``BackgroundSegmenter`` e ``OnnxBackgroundSegmenter``, capazes de
executar modelos como RMBG-1.4, BiRefNet e U2Net via ONNXRuntime, gerando
mascara alfa precisa sem halos ou bordas serrilhadas.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import av
import numpy as np

from video_engine.thumbnail.edge_enhancer import feather_mask
from video_engine.thumbnail.models import (
    GlowConfig,
    SegmentationResult,
    SegmenterConfig,
    StrokeConfig,
)

logger = logging.getLogger(__name__)


def _resize_frame_rgb(frame_rgb: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Redimensiona frame RGB usando libswscale do PyAV."""
    src_h, src_w = frame_rgb.shape[:2]
    if src_w == target_w and src_h == target_h:
        return frame_rgb.copy()
    av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
    scaled = av_frame.reformat(width=target_w, height=target_h, interpolation="BILINEAR")
    return scaled.to_ndarray(format="rgb24")


def _resize_mask_gray(mask: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Redimensiona mascara em escala de cinza usando libswscale do PyAV."""
    src_h, src_w = mask.shape[:2]
    if src_w == target_w and src_h == target_h:
        return mask.copy()
    av_frame = av.VideoFrame.from_ndarray(mask, format="gray")
    scaled = av_frame.reformat(width=target_w, height=target_h, interpolation="BILINEAR")
    return scaled.to_ndarray(format="gray")


def _read_image_file(image_path: Path) -> np.ndarray:
    """Carrega uma imagem em disco como array RGB (HxWx3, uint8) via PyAV."""
    with av.open(str(image_path)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        return frame.to_ndarray(format="rgb24")


def _save_rgba_png(rgba: np.ndarray, output_path: Path) -> Path:
    """Grava array RGBA como arquivo PNG com canal alfa transparente."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = rgba.shape[:2]
    with av.open(str(output_path), mode="w", format="image2") as container:
        stream = container.add_stream("png", rate=1)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "rgba"
        frame = av.VideoFrame.from_ndarray(rgba, format="rgba")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return output_path


class BackgroundSegmenter(ABC):
    """Interface abstrata para motores de segmentacao semantica de fundo."""

    @abstractmethod
    def segment(self, frame_rgb: np.ndarray) -> SegmentationResult:
        """Extrai o sujeito de um frame RGB gerando resultado com canal alfa."""


class OnnxBackgroundSegmenter(BackgroundSegmenter):
    """Segmentador semantico baseado em ONNXRuntime (ex: RMBG-1.4 / BiRefNet).

    Args:
        config: Configuracao de resolucao, limiar e feathering.
        onnx_session: Sessao de inferencia ONNX injetavel (determinismo em testes).
    """

    def __init__(
        self,
        config: Optional[SegmenterConfig] = None,
        onnx_session: Optional[Any] = None,
    ) -> None:
        self.config = config or SegmenterConfig()
        self._session = onnx_session

    def _get_session(self) -> Any:
        if self._session is None:
            if not self.config.model_path:
                raise RuntimeError(
                    "Caminho do modelo nao configurado em SegmenterConfig.model_path "
                    "e nenhuma sessao ONNX foi injetada."
                )
            import onnxruntime as ort

            self._session = ort.InferenceSession(self.config.model_path)
        return self._session

    def segment(self, frame_rgb: np.ndarray) -> SegmentationResult:
        """Executa segmentacao semantica em um frame RGB (HxWx3, uint8).

        Raises:
            ValueError: se ``frame_rgb`` for vazio ou nao tiver formato HxWx3.
        """
        array = np.asarray(frame_rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
            raise ValueError("Array de frame invalido: esperado HxWx3 com 3 canais RGB")

        orig_h, orig_w = array.shape[:2]
        target_w, target_h = self.config.target_size

        # Preprocessamento: redimensiona e normaliza
        resized_rgb = _resize_frame_rgb(array, target_w, target_h)
        tensor = resized_rgb.astype(np.float32) / 255.0
        # Normalizacao ImageNet padrao
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        # Formato (1, C, H, W)
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]

        # Inferencia ONNX
        session = self._get_session()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: tensor})
        pred = outputs[0]

        # Squeeze para (target_h, target_w) e aplica sigmoid se necessario
        pred_map = np.squeeze(pred)
        if pred_map.min() < 0.0 or pred_map.max() > 1.0:
            pred_map = 1.0 / (1.0 + np.exp(-pred_map))

        # Redimensiona a mascara de probabilidade de volta para a resolucao original
        mask_uint8 = np.clip(np.round(pred_map * 255.0), 0, 255).astype(np.uint8)
        mask_original = _resize_mask_gray(mask_uint8, orig_w, orig_h)

        # Binarizacao com limiar e suavizacao de borda (feathering / anti-aliasing)
        thresh_val = int(round(self.config.threshold * 255))
        binary_mask = (mask_original >= thresh_val).astype(np.uint8) * 255
        smooth_mask = feather_mask(binary_mask, radius=self.config.feather_radius)

        # Monta array RGBA (preserva RGB original com mascara alfa no canal 3)
        rgba = np.zeros((orig_h, orig_w, 4), dtype=np.uint8)
        rgba[:, :, :3] = array
        rgba[:, :, 3] = smooth_mask

        return SegmentationResult(
            width=orig_w,
            height=orig_h,
            foreground_rgba=rgba,
            alpha_mask=smooth_mask,
        )

    def remove_background(
        self,
        input_image: Union[str, Path, np.ndarray],
        output_path: Optional[Union[str, Path]] = None,
        stroke: Optional[StrokeConfig] = None,
        glow: Optional[GlowConfig] = None,
    ) -> SegmentationResult:
        """Segmenta o fundo de imagem em disco ou array, com efeitos e exportacao opcionais."""
        if isinstance(input_image, (str, Path)):
            path = Path(input_image)
            if not path.is_file():
                raise FileNotFoundError(f"Arquivo de imagem nao encontrado: {path}")
            frame_rgb = _read_image_file(path)
        else:
            frame_rgb = input_image

        result = self.segment(frame_rgb)

        # Se houver solicitacao de gravacao em disco
        if output_path is not None:
            dest = Path(output_path)
            if stroke or glow:
                enhanced = result.apply_enhancements(stroke=stroke, glow=glow)
                _save_rgba_png(enhanced, dest)
            else:
                _save_rgba_png(result.foreground_rgba, dest)

        return result


__all__ = [
    "BackgroundSegmenter",
    "OnnxBackgroundSegmenter",
]

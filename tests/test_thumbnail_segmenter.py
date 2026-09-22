"""Testes do motor de segmentacao semantica e remocao de fundo (Issue #12)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import av
import numpy as np
import pytest

from video_engine.thumbnail.models import GlowConfig, SegmenterConfig, StrokeConfig
from video_engine.thumbnail.segmenter import (
    OnnxBackgroundSegmenter,
    resolve_model_path,
)


class MockOnnxSession:
    """Mock deterministico de sessao ONNX que gera probabilidade circular no centro."""

    def __init__(self, target_size: tuple[int, int] = (1024, 1024)) -> None:
        self.target_size = target_size
        self._inputs = [type("Input", (), {"name": "input"})]

    def get_inputs(self) -> List[Any]:
        return self._inputs

    def run(self, output_names: Any, input_feed: dict) -> List[np.ndarray]:
        input_tensor = next(iter(input_feed.values()))
        batch, channels, height, width = input_tensor.shape

        # Gera probabilidade alta (1.0) num retangulo/circulo central e 0.0 no resto
        prob_map = np.zeros((batch, 1, height, width), dtype=np.float32)
        cy, cx = height // 2, width // 2
        ry, rx = height // 4, width // 4
        prob_map[:, :, cy - ry : cy + ry, cx - rx : cx + rx] = 1.0
        return [prob_map]


@pytest.fixture
def test_rgb_image() -> np.ndarray:
    """Cria imagem RGB 200x300 com cores distintas."""
    h, w = 200, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :150] = [200, 50, 50]   # metade esquerda avermelhada
    img[:, 150:] = [50, 200, 50]   # metade direita esverdeada
    return img


class TestOnnxBackgroundSegmenter:
    def test_segment_produces_valid_result(self, test_rgb_image: np.ndarray) -> None:
        mock_session = MockOnnxSession()
        segmenter = OnnxBackgroundSegmenter(
            config=SegmenterConfig(feather_radius=1),
            onnx_session=mock_session,
        )

        result = segmenter.segment(test_rgb_image)

        assert result.width == test_rgb_image.shape[1]
        assert result.height == test_rgb_image.shape[0]
        assert result.foreground_rgba.shape == (result.height, result.width, 4)
        assert result.foreground_rgba.dtype == np.uint8
        assert result.alpha_mask.shape == (result.height, result.width)
        assert result.alpha_mask.dtype == np.uint8

        # Regiao central deve ter canal alfa 255 (sujeito)
        cy, cx = result.height // 2, result.width // 2
        assert result.alpha_mask[cy, cx] == 255
        assert result.foreground_rgba[cy, cx, 3] == 255
        # Cores RGB centrais devem corresponder a imagem original
        assert np.array_equal(result.foreground_rgba[cy, cx, :3], test_rgb_image[cy, cx])

        # Cantos devem ter canal alfa 0 (fundo removido)
        assert result.alpha_mask[0, 0] == 0
        assert result.foreground_rgba[0, 0, 3] == 0

    def test_apply_enhancements_method(self, test_rgb_image: np.ndarray) -> None:
        mock_session = MockOnnxSession()
        segmenter = OnnxBackgroundSegmenter(onnx_session=mock_session)
        result = segmenter.segment(test_rgb_image)

        stroke_cfg = StrokeConfig(enabled=True, width=6, color=(255, 255, 255))
        glow_cfg = GlowConfig(enabled=True, radius=12, color=(0, 200, 255))

        enhanced = result.apply_enhancements(stroke=stroke_cfg, glow=glow_cfg)

        assert enhanced.shape == (result.height, result.width, 4)
        assert enhanced.dtype == np.uint8
        # Deve ter area transparente nao-nula reduzida devido a presenca de stroke/glow externo
        assert np.sum(enhanced[:, :, 3] > 0) > np.sum(result.foreground_rgba[:, :, 3] > 0)

    def test_remove_background_with_export(
        self,
        test_rgb_image: np.ndarray,
        tmp_path: Path,
    ) -> None:
        mock_session = MockOnnxSession()
        segmenter = OnnxBackgroundSegmenter(onnx_session=mock_session)

        output_png = tmp_path / "cutout.png"
        stroke = StrokeConfig(enabled=True, width=4)
        glow = GlowConfig(enabled=True, radius=8)

        result = segmenter.remove_background(
            input_image=test_rgb_image,
            output_path=output_png,
            stroke=stroke,
            glow=glow,
        )

        assert output_png.is_file()
        assert output_png.stat().st_size > 0

        # Valida que o arquivo gravado e um PNG valido com canal alfa
        with av.open(str(output_png)) as container:
            stream = container.streams.video[0]
            assert stream.width == test_rgb_image.shape[1]
            assert stream.height == test_rgb_image.shape[0]
            frame = next(container.decode(stream))
            rgba = frame.to_ndarray(format="rgba")
            assert rgba.shape == (result.height, result.width, 4)

    def test_invalid_inputs(self, tmp_path: Path) -> None:
        segmenter = OnnxBackgroundSegmenter(onnx_session=MockOnnxSession())

        with pytest.raises(FileNotFoundError):
            segmenter.remove_background(tmp_path / "nao_existe.jpg")

        with pytest.raises(ValueError, match="esperado HxWx3"):
            segmenter.segment(np.zeros((100, 100), dtype=np.uint8))  # 2D

        with pytest.raises(ValueError, match="esperado HxWx3"):
            segmenter.segment(np.empty((0, 0, 3), dtype=np.uint8))  # vazio

    def test_resolve_model_path_explicit(self, tmp_path: Path) -> None:
        model_file = tmp_path / "custom_model.onnx"
        model_file.write_bytes(b"dummy onnx content")
        assert resolve_model_path(str(model_file)) == model_file

    def test_resolve_model_path_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        model_file = tmp_path / "env_model.onnx"
        model_file.write_bytes(b"dummy onnx content")
        monkeypatch.setenv("VIDEO_ENGINE_SEGMENTER_MODEL_PATH", str(model_file))
        assert resolve_model_path() == model_file

    def test_resolve_model_path_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VIDEO_ENGINE_SEGMENTER_MODEL_PATH", raising=False)
        monkeypatch.delenv("SEGMENTER_MODEL_PATH", raising=False)
        # Caminho inexistente explícito deve retornar None
        assert resolve_model_path("caminho/completamente/inexistente.onnx") is None

    def test_get_session_raises_descriptive_error_when_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VIDEO_ENGINE_SEGMENTER_MODEL_PATH", raising=False)
        monkeypatch.delenv("SEGMENTER_MODEL_PATH", raising=False)
        monkeypatch.setattr("video_engine.thumbnail.segmenter.resolve_model_path", lambda *args, **kwargs: None)
        segmenter = OnnxBackgroundSegmenter(config=SegmenterConfig(model_path=None))
        # Sem modelo baixado e sem caminho, deve disparar RuntimeError claro
        with pytest.raises(RuntimeError, match="Caminho do modelo de segmentacao nao encontrado"):
            segmenter._get_session()

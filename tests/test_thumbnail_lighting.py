"""Testes de avaliacao de iluminacao e exposicao de frames (Issue #11)."""

from __future__ import annotations

import numpy as np
import pytest

from video_engine.thumbnail.lighting import (
    is_poor_lighting,
    lighting_score,
    luminance_channel,
    luminance_stats,
)


class TestLightingMetrics:
    def test_luminance_channel_bt601(self) -> None:
        # Puro branco -> Y = 255
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        luma_white = luminance_channel(white)
        assert np.allclose(luma_white, 255.0)

        # Puro preto -> Y = 0
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        luma_black = luminance_channel(black)
        assert np.allclose(luma_black, 0.0)

        # Puro verde: Y = 0.587 * 255 = 149.685
        green = np.zeros((5, 5, 3), dtype=np.uint8)
        green[:, :, 1] = 255
        luma_green = luminance_channel(green)
        assert np.allclose(luma_green, 0.587 * 255.0)

    def test_luminance_stats(self) -> None:
        frame = np.full((20, 20, 3), 100, dtype=np.uint8)
        mean, std = luminance_stats(frame)
        assert mean == pytest.approx(100.0, abs=1e-3)
        assert std == pytest.approx(0.0, abs=1e-3)

    def test_subexposed_poor_lighting(self) -> None:
        # Frame muito escuro (subexposto)
        dark_frame = np.full((50, 50, 3), 15, dtype=np.uint8)
        assert is_poor_lighting(dark_frame, min_luminance=35.0, max_luminance=225.0)
        score = lighting_score(dark_frame, min_luminance=35.0, max_luminance=225.0)
        assert score == 0.0

    def test_overexposed_poor_lighting(self) -> None:
        # Frame estourado (superexposto)
        blown_frame = np.full((50, 50, 3), 245, dtype=np.uint8)
        assert is_poor_lighting(blown_frame, min_luminance=35.0, max_luminance=225.0)
        score = lighting_score(blown_frame, min_luminance=35.0, max_luminance=225.0)
        assert score == 0.0

    def test_balanced_lighting_score(self) -> None:
        # Frame equilibrado no centro da janela (em torno de 130) com bom contraste
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:50, :] = 80
        frame[50:, :] = 180
        mean, std = luminance_stats(frame)
        assert mean == pytest.approx(130.0, abs=1.0)
        assert std > 40.0

        assert not is_poor_lighting(frame, min_luminance=35.0, max_luminance=225.0)
        score = lighting_score(frame, min_luminance=35.0, max_luminance=225.0)
        assert 0.6 <= score <= 1.0

    def test_invalid_frame_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="esperado HxWx3"):
            luminance_channel(np.ones((20, 20), dtype=np.uint8))

        with pytest.raises(ValueError, match="esperado HxWx3"):
            luminance_channel(np.empty((0, 0, 3), dtype=np.uint8))

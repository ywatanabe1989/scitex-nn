"""Tests for vendored _ensure_3d helper."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from scitex_nn._vendor_dsp_utils._ensure_3d import ensure_3d


class TestEnsure3D:
    """Tests for the ensure_3d function."""

    def test_ensure_3d_1d_tensor_becomes_3d(self):
        """ensure_3d expands a 1D tensor to 3D."""
        # Arrange
        x = torch.randn(10)
        # Act
        result = ensure_3d(x)
        # Assert
        assert result.ndim == 3

    def test_ensure_3d_1d_tensor_shape_correct(self):
        """ensure_3d reshapes 1D (seq_len,) to (1, 1, seq_len)."""
        # Arrange
        x = torch.randn(10)
        # Act
        result = ensure_3d(x)
        # Assert
        assert result.shape == (1, 1, 10)

    def test_ensure_3d_2d_tensor_becomes_3d(self):
        """ensure_3d expands a 2D tensor to 3D."""
        # Arrange
        x = torch.randn(4, 10)
        # Act
        result = ensure_3d(x)
        # Assert
        assert result.ndim == 3

    def test_ensure_3d_2d_tensor_shape_correct(self):
        """ensure_3d reshapes 2D (batch, seq_len) to (batch, 1, seq_len)."""
        # Arrange
        x = torch.randn(4, 10)
        # Act
        result = ensure_3d(x)
        # Assert
        assert result.shape == (4, 1, 10)

    def test_ensure_3d_3d_tensor_unchanged(self):
        """ensure_3d leaves a 3D tensor unchanged."""
        # Arrange
        x = torch.randn(4, 2, 10)
        # Act
        result = ensure_3d(x)
        # Assert
        assert result.shape == (4, 2, 10)

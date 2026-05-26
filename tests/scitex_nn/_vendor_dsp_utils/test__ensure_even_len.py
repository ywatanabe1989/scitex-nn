"""Tests for vendored _ensure_even_len helper."""

from __future__ import annotations

import torch

from scitex_nn._vendor_dsp_utils._ensure_even_len import ensure_even_len


class TestEnsureEvenLen:
    """Tests for the ensure_even_len function."""

    def test_ensure_even_len_even_tensor_unchanged(self):
        """ensure_even_len leaves even-length last dim unchanged."""
        # Arrange
        x = torch.randn(2, 4)
        # Act
        result = ensure_even_len(x)
        # Assert
        assert result.shape == (2, 4)

    def test_ensure_even_len_odd_tensor_truncated(self):
        """ensure_even_len truncates odd-length last dim by one."""
        # Arrange
        x = torch.randn(2, 5)
        # Act
        result = ensure_even_len(x)
        # Assert
        assert result.shape[-1] == 4

    def test_ensure_even_len_odd_3d_tensor_truncated(self):
        """ensure_even_len truncates odd last dim on 3D tensors."""
        # Arrange
        x = torch.randn(2, 3, 7)
        # Act
        result = ensure_even_len(x)
        # Assert
        assert result.shape == (2, 3, 6)

    def test_ensure_even_len_values_unchanged_for_even_slice(self):
        """ensure_even_len preserves all values for even-length tensors."""
        # Arrange
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        # Act
        result = ensure_even_len(x)
        # Assert
        assert bool(torch.allclose(result, x))

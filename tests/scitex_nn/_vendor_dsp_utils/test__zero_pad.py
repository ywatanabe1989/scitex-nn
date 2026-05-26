"""Tests for vendored _zero_pad helper."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import numpy as np
import torch

from scitex_nn._vendor_dsp_utils._zero_pad import zero_pad


class TestZeroPad:
    """Tests for the zero_pad function."""

    def test_zero_pad_stacks_equal_length_tensors(self):
        """zero_pad stacks equal-length tensors along dim 0."""
        # Arrange
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        # Act
        result = zero_pad([a, b])
        # Assert
        assert result.shape == (2, 3)

    def test_zero_pad_pads_shorter_tensor_to_longer(self):
        """zero_pad pads the shorter tensor with zeros to match length."""
        # Arrange
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        # Act
        result = zero_pad([a, b])
        # Assert
        assert result.shape == (2, 3)

    def test_zero_pad_accepts_numpy_arrays(self):
        """zero_pad converts numpy arrays to tensors."""
        # Arrange
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        # Act
        result = zero_pad([a, b])
        # Assert
        assert isinstance(result, torch.Tensor)

    def test_zero_pad_single_element_list_works(self):
        """zero_pad handles a single-element list."""
        # Arrange
        a = torch.tensor([1.0, 2.0, 3.0])
        # Act
        result = zero_pad([a])
        # Assert
        assert result.shape == (1, 3)

    def test_zero_pad_pads_shorter_tensor_left_with_zeros(self):
        """zero_pad pads the left side of shorter tensors."""
        # Arrange
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0])
        # Act
        result = zero_pad([a, b])
        # Assert
        assert result[1, 0] == 0.0

    def test_zero_pad_pads_shorter_tensor_right_with_zeros(self):
        """zero_pad pads the right side of shorter tensors."""
        # Arrange
        a = torch.tensor([1.0, 2.0, 3.0, 4.0])
        b = torch.tensor([5.0, 6.0])
        # Act
        result = zero_pad([a, b])
        # Assert
        assert result[1, 3] == 0.0

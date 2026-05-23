#!/usr/bin/env python3
# Time-stamp: "2025-01-06 (ywatanabe)"
# File: tests/scitex/nn/test__FreqGainChanger.py

"""Test suite for FreqGainChanger module.

FreqGainChanger depends on the external `julius` library for band-splitting.
These tests exercise the real layer end-to-end against the real `julius`
implementation. When `julius` is not installed the whole module is skipped
(no mocks — a mocked `split_bands` would only test the mock, not the layer).
"""

import pytest

# Required for this module
pytest.importorskip("torch")
# FreqGainChanger does `import julius` at module top, so the source itself is
# unimportable without julius. Skip the entire module when it is absent rather
# than mock the dependency.
pytest.importorskip("julius")

import torch
import torch.nn as nn

from scitex_nn import FreqGainChanger


class TestFreqGainChanger:
    """Test suite for FreqGainChanger layer."""

    @pytest.fixture
    def sample_rate(self):
        """Standard sample rate for testing."""
        return 1000

    @pytest.fixture
    def n_bands(self):
        """Number of frequency bands for testing."""
        return 10

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size, n_channels, seq_len = 4, 32, 1000
        return torch.randn(batch_size, n_channels, seq_len)

    def test_initialization_stores_n_bands(self, n_bands, sample_rate):
        """Test that n_bands is stored on the layer."""
        # Arrange
        # Act
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        # Assert
        assert layer.n_bands == n_bands

    def test_initialization_stores_samp_rate(self, n_bands, sample_rate):
        """Test that samp_rate is stored on the layer."""
        # Arrange
        # Act
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        # Assert
        assert layer.samp_rate == sample_rate

    def test_initialization_creates_dropout_module(self, n_bands, sample_rate):
        """Test that the dropout attribute is an nn.Dropout instance."""
        # Arrange
        # Act
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        # Assert
        assert isinstance(layer.dropout, nn.Dropout)

    def test_initialization_uses_fixed_dropout_probability(self, n_bands, sample_rate):
        """Test that the dropout probability is hardcoded to 0.5."""
        # Arrange
        # Act
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        # Assert
        assert layer.dropout.p == 0.5

    def test_forward_training_mode_preserves_input_shape(
        self, n_bands, sample_rate, sample_input
    ):
        """Test forward pass in training mode preserves the input shape."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.train()
        # Act
        output = layer(sample_input)
        # Assert
        assert output.shape == sample_input.shape

    def test_forward_eval_mode_is_identity(self, n_bands, sample_rate, sample_input):
        """Test forward pass in evaluation mode returns the input unchanged."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.eval()
        # Act
        output = layer(sample_input)
        # Assert
        assert torch.allclose(output, sample_input)

    def test_forward_training_mode_changes_input(self, n_bands, sample_rate):
        """Test that the layer alters the signal in training mode."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.train()
        x = torch.randn(2, 3, 256)
        # Act
        output = layer(x)
        # Assert
        assert not torch.allclose(output, x)

    def test_forward_gradient_flows_to_input(self, n_bands, sample_rate):
        """Test that gradients flow back to the input in training mode."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.train()
        x = torch.randn(2, 3, 256, requires_grad=True)
        # Act
        layer(x).sum().backward()
        # Assert
        assert x.grad is not None

    def test_forward_handles_zero_input(self, n_bands, sample_rate):
        """Test behavior with a zero input tensor."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.train()
        x = torch.zeros(2, 3, 256)
        # Act
        output = layer(x)
        # Assert
        assert torch.allclose(output, x)

    def test_forward_output_is_finite(self, n_bands, sample_rate):
        """Test that the output contains no NaN values."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.train()
        x = torch.randn(2, 3, 256)
        # Act
        output = layer(x)
        # Assert
        assert not torch.isnan(output).any()

    def test_forward_output_has_no_infinities(self, n_bands, sample_rate):
        """Test that the output contains no infinite values."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate)
        layer.train()
        x = torch.randn(2, 3, 256)
        # Act
        output = layer(x)
        # Assert
        assert not torch.isinf(output).any()

    def test_forward_runs_in_sequential_model(self, n_bands, sample_rate):
        """Test integration inside an nn.Sequential model."""
        # Arrange
        model = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
        )
        model.train()
        x = torch.randn(4, 32, 256)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (4, 32, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_preserves_device_on_cuda(self, n_bands, sample_rate):
        """Test that the output stays on the input's CUDA device."""
        # Arrange
        layer = FreqGainChanger(n_bands=n_bands, samp_rate=sample_rate).cuda()
        layer.train()
        x = torch.randn(2, 3, 256).cuda()
        # Act
        output = layer(x)
        # Assert
        assert output.is_cuda


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])

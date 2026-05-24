#!/usr/bin/env python3
# Time-stamp: "2025-01-06 (ywatanabe)"
# File: tests/scitex/nn/test__Spectrogram.py

"""Comprehensive test suite for Spectrogram module.

This module tests the spectrogram computation functionality for neural networks,
including STFT parameters, window functions, multi-channel support, and edge cases.
"""

import pytest

# Required for this module
pytest.importorskip("torch")
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from scitex_nn import Spectrogram


class TestSpectrogram:
    """Test suite for Spectrogram layer."""

    @pytest.fixture
    def sampling_rate(self):
        """Standard sampling rate for testing."""
        return 1000

    @pytest.fixture
    def n_fft(self):
        """Default FFT size for testing."""
        return 256

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size, n_channels, seq_len = 2, 3, 1000
        return torch.randn(batch_size, n_channels, seq_len)

    def test_initialization_default_params_sampling_rate(self, sampling_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Assert
        assert layer.sampling_rate == sampling_rate

    def test_initialization_default_params_n_fft(self, sampling_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Assert
        pass
        assert layer.n_fft == 256
        pass
        pass
        pass

    def test_initialization_default_params_hop_length(self, sampling_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Assert
        pass
        pass
        assert layer.hop_length == 256 // 4
        pass
        pass
        pass

    def test_initialization_default_params_win_length(self, sampling_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Assert
        pass
        pass
        pass
        assert layer.win_length == 256
        pass
        pass

    def test_initialization_default_params_isinstance(self, sampling_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Assert
        pass
        pass
        pass
        pass
        assert isinstance(layer.window, torch.Tensor)
        pass

    def test_initialization_default_params_shape(self, sampling_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Assert
        pass
        pass
        pass
        pass
        pass
        assert layer.window.shape == (256,)

    def test_initialization_custom_params_n_fft(self, sampling_rate):
        """Test initialization with custom parameters."""
        # Arrange
        n_fft = 512
        hop_length = 128
        win_length = 400
        # Act
        layer = Spectrogram(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        # Assert
        assert layer.n_fft == n_fft
        pass
        pass
        pass

    def test_initialization_custom_params_hop_length(self, sampling_rate):
        """Test initialization with custom parameters."""
        # Arrange
        n_fft = 512
        hop_length = 128
        win_length = 400
        # Act
        layer = Spectrogram(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        # Assert
        pass
        assert layer.hop_length == hop_length
        pass
        pass

    def test_initialization_custom_params_win_length(self, sampling_rate):
        """Test initialization with custom parameters."""
        # Arrange
        n_fft = 512
        hop_length = 128
        win_length = 400
        # Act
        layer = Spectrogram(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        # Assert
        pass
        pass
        assert layer.win_length == win_length
        pass

    def test_initialization_custom_params_shape(self, sampling_rate):
        """Test initialization with custom parameters."""
        # Arrange
        n_fft = 512
        hop_length = 128
        win_length = 400
        # Act
        layer = Spectrogram(
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
        )
        # Assert
        pass
        pass
        pass
        assert layer.window.shape == (win_length,)

    def test_initialization_invalid_window(self, sampling_rate):
        """Test initialization with invalid window type."""
        # Arrange
        # Act
        # Assert
        with pytest.raises(ValueError, match="Unsupported window type"):
            Spectrogram(sampling_rate=sampling_rate, window="invalid")

    def test_forward_basic_spectrogram_behaves_correctly_check1(
        self, sampling_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        expected_freq_bins = layer.n_fft // 2 + 1
        # Assert
        assert spectrograms.shape[0] == batch_size
        pass
        pass
        pass
        pass

    def test_forward_basic_spectrogram_behaves_correctly_check2(
        self, sampling_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        expected_freq_bins = layer.n_fft // 2 + 1
        # Assert
        pass
        assert spectrograms.shape[1] == n_channels
        pass
        pass
        pass

    def test_forward_basic_spectrogram_behaves_correctly_check3(
        self, sampling_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        expected_freq_bins = layer.n_fft // 2 + 1
        # Assert
        pass
        pass
        assert spectrograms.shape[2] == expected_freq_bins
        pass
        pass

    def test_forward_basic_spectrogram_behaves_correctly_shape(
        self, sampling_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        expected_freq_bins = layer.n_fft // 2 + 1
        # Assert
        pass
        pass
        pass
        assert freqs.shape == (expected_freq_bins,)
        pass

    def test_forward_basic_spectrogram_behaves_correctly_check5(
        self, sampling_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        expected_freq_bins = layer.n_fft // 2 + 1
        # Assert
        pass
        pass
        pass
        pass
        assert times.shape[0] == spectrograms.shape[3]

    def test_forward_single_channel(self, sampling_rate):
        """Test forward pass with single channel input."""
        # Arrange
        x = torch.randn(2, 1, 1000)
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        assert spectrograms.shape[1] == 1

    def test_forward_multi_channel(self, sampling_rate):
        """Test forward pass with multi-channel input."""
        # Arrange
        n_channels = 8
        x = torch.randn(2, n_channels, 1000)
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        assert spectrograms.shape[1] == n_channels

    def test_frequency_range_spectrogram_behaves_correctly_check1(self, sampling_rate):
        """Test that frequency range is correct (0 to Nyquist)."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(1, 1, 1000)
        # Act
        _, freqs, _ = layer(x)
        # Assert
        assert freqs[0] == 0
        pass
        pass

    def test_frequency_range_spectrogram_behaves_correctly_check2(self, sampling_rate):
        """Test that frequency range is correct (0 to Nyquist)."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(1, 1, 1000)
        # Act
        _, freqs, _ = layer(x)
        # Assert
        pass
        assert freqs[-1] == sampling_rate / 2
        pass

    def test_frequency_range_spectrogram_behaves_correctly_all(self, sampling_rate):
        """Test that frequency range is correct (0 to Nyquist)."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(1, 1, 1000)
        # Act
        _, freqs, _ = layer(x)
        # Assert
        pass
        pass
        assert torch.all(torch.diff(freqs) > 0)

    def test_time_values_spectrogram_behaves_correctly(self, sampling_rate):
        """Test that time values are correctly computed."""
        # Arrange
        hop_length = 100
        layer = Spectrogram(sampling_rate=sampling_rate, hop_length=hop_length)
        x = torch.randn(1, 1, 1000)
        _, _, times = layer(x)
        expected_time_step = hop_length / sampling_rate
        # Act
        time_diffs = torch.diff(times)
        # Assert
        assert torch.allclose(time_diffs, torch.tensor(expected_time_step))

    def test_different_fft_sizes_check1(self, sampling_rate):
        """Test with various FFT sizes."""
        # Arrange
        fft_sizes = [128, 256, 512, 1024]
        # Act
        x = torch.randn(2, 3, 2000)
        # Assert
        for n_fft in fft_sizes:
            layer = Spectrogram(sampling_rate=sampling_rate, n_fft=n_fft)
            spectrograms, freqs, times = layer(x)
            expected_freq_bins = n_fft // 2 + 1
            assert spectrograms.shape[2] == expected_freq_bins
            pass

    def test_different_fft_sizes_check2(self, sampling_rate):
        """Test with various FFT sizes."""
        # Arrange
        fft_sizes = [128, 256, 512, 1024]
        # Act
        x = torch.randn(2, 3, 2000)
        # Assert
        for n_fft in fft_sizes:
            layer = Spectrogram(sampling_rate=sampling_rate, n_fft=n_fft)
            spectrograms, freqs, times = layer(x)
            expected_freq_bins = n_fft // 2 + 1
            pass
            assert freqs.shape[0] == expected_freq_bins

    def test_different_hop_lengths(self, sampling_rate):
        """Test with various hop lengths."""
        # Arrange
        n_fft = 256
        hop_lengths = [32, 64, 128, 256]
        # Act
        x = torch.randn(2, 3, 2000)
        prev_n_frames = None
        # Assert
        for hop_length in hop_lengths:
            layer = Spectrogram(
                sampling_rate=sampling_rate, n_fft=n_fft, hop_length=hop_length
            )
            spectrograms, _, times = layer(x)
            n_frames = spectrograms.shape[3]
            if prev_n_frames is not None:
                assert n_frames < prev_n_frames
            prev_n_frames = n_frames

    def test_gradient_flow_spectrogram_behaves_correctly_grad(
        self, sampling_rate, sample_input
    ):
        """Test that gradients flow properly through the layer."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        sample_input.requires_grad = True
        spectrograms, _, _ = layer(sample_input)
        loss = spectrograms.sum()
        # Act
        loss.backward()
        # Assert
        assert sample_input.grad is not None
        pass

    def test_gradient_flow_spectrogram_behaves_correctly_allclose(
        self, sampling_rate, sample_input
    ):
        """Test that gradients flow properly through the layer."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        sample_input.requires_grad = True
        spectrograms, _, _ = layer(sample_input)
        loss = spectrograms.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.allclose(
            sample_input.grad, torch.zeros_like(sample_input.grad)
        )

    def test_device_compatibility_cpu_device(self, sampling_rate):
        """Test layer works on CPU."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(2, 3, 1000)
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        assert spectrograms.device == x.device
        pass

    def test_device_compatibility_cpu_is_cuda(self, sampling_rate):
        """Test layer works on CPU."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(2, 3, 1000)
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        pass
        assert not spectrograms.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_device(self, sampling_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        assert spectrograms.device == x.device
        pass
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_is_cuda(self, sampling_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        pass
        assert spectrograms.is_cuda
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_is_cuda_v2(self, sampling_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        spectrograms, freqs, times = layer(x)
        # Assert
        pass
        pass
        assert layer.window.is_cuda

    def test_window_length_variations_any(self, sampling_rate):
        """Test with different window lengths."""
        # Arrange
        n_fft = 512
        win_lengths = [128, 256, 384, 512]
        # Act
        x = torch.randn(2, 3, 2000)
        # Assert
        for win_length in win_lengths:
            layer = Spectrogram(
                sampling_rate=sampling_rate, n_fft=n_fft, win_length=win_length
            )
            spectrograms, _, _ = layer(x)
            assert not torch.isnan(spectrograms).any()
            pass

    def test_window_length_variations_any_v2(self, sampling_rate):
        """Test with different window lengths."""
        # Arrange
        n_fft = 512
        win_lengths = [128, 256, 384, 512]
        # Act
        x = torch.randn(2, 3, 2000)
        # Assert
        for win_length in win_lengths:
            layer = Spectrogram(
                sampling_rate=sampling_rate, n_fft=n_fft, win_length=win_length
            )
            spectrograms, _, _ = layer(x)
            pass
            assert not torch.isinf(spectrograms).any()

    def test_short_signal_handling_check1(self, sampling_rate):
        """Test with signals that are at minimum length for STFT.

        Note: PyTorch STFT requires signal length >= n_fft for center=True.
        Signals too short will raise RuntimeError.
        """
        # Arrange
        n_fft = 256
        layer = Spectrogram(sampling_rate=sampling_rate, n_fft=n_fft)
        x_min = torch.randn(2, 3, n_fft)
        # Act
        spectrograms, _, _ = layer(x_min)
        # Assert
        assert spectrograms.shape[0] == 2
        pass
        pass

    def test_short_signal_handling_check2(self, sampling_rate):
        """Test with signals that are at minimum length for STFT.

        Note: PyTorch STFT requires signal length >= n_fft for center=True.
        Signals too short will raise RuntimeError.
        """
        # Arrange
        n_fft = 256
        layer = Spectrogram(sampling_rate=sampling_rate, n_fft=n_fft)
        x_min = torch.randn(2, 3, n_fft)
        # Act
        spectrograms, _, _ = layer(x_min)
        # Assert
        pass
        assert spectrograms.shape[1] == 3
        pass

    def test_short_signal_handling_any(self, sampling_rate):
        """Test with signals that are at minimum length for STFT.

        Note: PyTorch STFT requires signal length >= n_fft for center=True.
        Signals too short will raise RuntimeError.
        """
        # Arrange
        n_fft = 256
        layer = Spectrogram(sampling_rate=sampling_rate, n_fft=n_fft)
        x_min = torch.randn(2, 3, n_fft)
        # Act
        spectrograms, _, _ = layer(x_min)
        # Assert
        pass
        pass
        assert not torch.isnan(spectrograms).any()

    def test_zero_input_handling(self, sampling_rate):
        """Test behavior with zero input."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.zeros(2, 3, 1000)
        # Act
        spectrograms, _, _ = layer(x)
        # Assert
        assert torch.allclose(spectrograms, torch.zeros_like(spectrograms), atol=1e-10)

    def test_single_tone_input(self, sampling_rate):
        """Test with single frequency tone."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate, n_fft=512)
        t = torch.arange(0, 2, 1 / sampling_rate)
        freq = 100
        x = torch.sin(2 * np.pi * freq * t).unsqueeze(0).unsqueeze(0)
        spectrograms, freqs, _ = layer(x)
        avg_spectrum = spectrograms[0, 0].mean(dim=1)
        # Act
        peak_idx = torch.argmax(avg_spectrum)
        peak_freq = freqs[peak_idx]
        # Assert
        assert abs(peak_freq - freq) < 10

    def test_magnitude_only_output_dtype(self, sampling_rate, sample_input):
        """Test that output is magnitude (not complex)."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, _, _ = layer(sample_input)
        # Assert
        assert spectrograms.dtype in [torch.float32, torch.float64]
        pass

    def test_magnitude_only_output_all(self, sampling_rate, sample_input):
        """Test that output is magnitude (not complex)."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        # Act
        spectrograms, _, _ = layer(sample_input)
        # Assert
        pass
        assert (spectrograms >= 0).all()

    def test_numerical_stability_spectrogram_behaves_correctly_any(self, sampling_rate):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        spectrograms_large, _, _ = layer(x_large)
        # Assert
        assert not torch.isnan(spectrograms_large).any()
        pass
        x_small = torch.randn(2, 3, 1000) * 1e-06
        spectrograms_small, _, _ = layer(x_small)
        pass

    def test_numerical_stability_spectrogram_behaves_correctly_any_v2(
        self, sampling_rate
    ):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        spectrograms_large, _, _ = layer(x_large)
        # Assert
        pass
        assert not torch.isinf(spectrograms_large).any()
        x_small = torch.randn(2, 3, 1000) * 1e-06
        spectrograms_small, _, _ = layer(x_small)
        pass

    def test_numerical_stability_spectrogram_behaves_correctly_any_v3(
        self, sampling_rate
    ):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        spectrograms_large, _, _ = layer(x_large)
        # Assert
        pass
        pass
        x_small = torch.randn(2, 3, 1000) * 1e-06
        spectrograms_small, _, _ = layer(x_small)
        assert not torch.isnan(spectrograms_small).any()

    def test_batch_consistency_spectrogram_behaves_correctly(self, sampling_rate):
        """Test that batched processing gives consistent results."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x_single = torch.randn(1, 3, 1000)
        spec_single, _, _ = layer(x_single)
        x_batch = x_single.repeat(4, 1, 1)
        # Act
        spec_batch, _, _ = layer(x_batch)
        # Assert
        for i in range(4):
            assert torch.allclose(spec_batch[i], spec_single[0])

    def test_integration_with_sequential(self, sampling_rate):
        """Test integration in nn.Sequential."""
        # Arrange

        class SpectrogramWrapper(nn.Module):
            def __init__(self, sampling_rate):
                super().__init__()
                self.spec = Spectrogram(sampling_rate)

            def forward(self, x):
                spec, _, _ = self.spec(x)
                return spec

        model = nn.Sequential(
            SpectrogramWrapper(sampling_rate),
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        x = torch.randn(4, 3, 1000)
        # Act
        output = model(x)
        # Assert
        assert output.shape == (4, 16, 8, 8)

    def test_memory_efficiency_spectrogram_behaves_correctly_check1(
        self, sampling_rate
    ):
        """Test memory usage with large inputs."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(8, 16, 4000)
        # Act
        spectrograms, _, _ = layer(x)
        # Assert
        assert spectrograms.shape[0] == 8
        pass

    def test_memory_efficiency_spectrogram_behaves_correctly_check2(
        self, sampling_rate
    ):
        """Test memory usage with large inputs."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.randn(8, 16, 4000)
        # Act
        spectrograms, _, _ = layer(x)
        # Assert
        pass
        assert spectrograms.shape[1] == 16

    def test_padding_mode_effect_any(self, sampling_rate):
        """Test that reflect padding is working."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.ones(1, 1, 1000)
        x[:, :, :100] = -1
        # Act
        spectrograms, _, _ = layer(x)
        # Assert
        assert not torch.isnan(spectrograms).any()
        pass

    def test_padding_mode_effect_any_v2(self, sampling_rate):
        """Test that reflect padding is working."""
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate)
        x = torch.ones(1, 1, 1000)
        x[:, :, :100] = -1
        # Act
        spectrograms, _, _ = layer(x)
        # Assert
        pass
        assert not torch.isinf(spectrograms).any()

    def test_energy_preservation_spectrogram_behaves_correctly_ratio(
        self, sampling_rate
    ):
        """Test approximate energy preservation (Parseval's theorem).

        Note: With non-normalized STFT and overlapping windows, the frequency
        domain energy will be larger than time domain. The ratio depends on
        window type, overlap, and n_fft. This test just verifies the ratio
        is consistent and finite.
        """
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate, n_fft=512)
        x = torch.randn(1, 1, 2000)
        time_energy = (x**2).sum()
        spectrograms, _, _ = layer(x)
        # Act
        freq_energy = (spectrograms**2).sum()
        ratio = freq_energy / time_energy
        # Assert
        assert ratio > 0 and (not torch.isinf(ratio))
        pass

    def test_energy_preservation_spectrogram_behaves_correctly_check2(
        self, sampling_rate
    ):
        """Test approximate energy preservation (Parseval's theorem).

        Note: With non-normalized STFT and overlapping windows, the frequency
        domain energy will be larger than time domain. The ratio depends on
        window type, overlap, and n_fft. This test just verifies the ratio
        is consistent and finite.
        """
        # Arrange
        layer = Spectrogram(sampling_rate=sampling_rate, n_fft=512)
        x = torch.randn(1, 1, 2000)
        time_energy = (x**2).sum()
        spectrograms, _, _ = layer(x)
        # Act
        freq_energy = (spectrograms**2).sum()
        ratio = freq_energy / time_energy
        # Assert
        pass
        assert 100 < ratio < 1000


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_Spectrogram.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-02 09:21:12 (ywatanabe)"
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import scitex
# from scitex.decorators import numpy_fn, torch_fn
#
#
# class Spectrogram(nn.Module):
#     def __init__(
#         self,
#         sampling_rate,
#         n_fft=256,
#         hop_length=None,
#         win_length=None,
#         window="hann",
#     ):
#         super().__init__()
#         self.sampling_rate = sampling_rate
#         self.n_fft = n_fft
#         self.hop_length = hop_length if hop_length is not None else n_fft // 4
#         self.win_length = win_length if win_length is not None else n_fft
#         if window == "hann":
#             self.window = torch.hann_window(window_length=self.win_length)
#         else:
#             raise ValueError(
#                 "Unsupported window type. Extend this to support more window types."
#             )
#
#     def forward(self, x):
#         """
#         Computes the spectrogram for each channel in the input signal.
#
#         Parameters:
#         - signal (torch.Tensor): Input signal of shape (batch_size, n_chs, seq_len).
#
#         Returns:
#         - spectrograms (torch.Tensor): The computed spectrograms for each channel.
#         """
#
#         x = scitex.dsp.ensure_3d(x)
#
#         batch_size, n_chs, seq_len = x.shape
#         spectrograms = []
#
#         for ch in range(n_chs):
#             x_ch = x[:, ch, :].unsqueeze(1)  # Maintain expected input shape for stft
#             spec = torch.stft(
#                 x_ch.squeeze(1),
#                 n_fft=self.n_fft,
#                 hop_length=self.hop_length,
#                 win_length=self.win_length,
#                 window=self.window.to(x.device),
#                 center=True,
#                 pad_mode="reflect",
#                 normalized=False,
#                 return_complex=True,
#             )
#             magnitude = torch.abs(spec).unsqueeze(1)  # Keep channel dimension
#             spectrograms.append(magnitude)
#
#         # Concatenate spectrograms along channel dimension
#         spectrograms = torch.cat(spectrograms, dim=1)
#
#         # Calculate frequencies (y-axis)
#         freqs = torch.linspace(0, self.sampling_rate / 2, steps=self.n_fft // 2 + 1)
#
#         # Calculate times (x-axis)
#         # The number of frames can be computed from the size of the last dimension of the spectrogram
#         n_frames = spectrograms.shape[-1]
#         # Time of each frame in seconds, considering the hop length and sampling rate
#         times_sec = torch.arange(0, n_frames) * (self.hop_length / self.sampling_rate)
#
#         return spectrograms, freqs, times_sec
#
#
# @torch_fn
# def spectrograms(x, fs, cuda=False):
#     return Spectrogram(fs)(x)
#
#
# @torch_fn
# def my_softmax(x, dim=-1):
#     return F.softmax(x, dim=dim)
#
#
# @torch_fn
# def unbias(x, func="min", dim=-1, cuda=False):
#     if func == "min":
#         return x - x.min(dim=dim, keepdims=True)[0]
#     if func == "mean":
#         return x - x.mean(dim=dim, keepdims=True)[0]
#
#
# @torch_fn
# def normalize(x, axis=-1, amp=1.0, cuda=False):
#     high = torch.abs(x.max(axis=axis, keepdims=True)[0])
#     low = torch.abs(x.min(axis=axis, keepdims=True)[0])
#     return amp * x / torch.maximum(high, low)
#
#
# @torch_fn
# def spectrograms(x, fs, dj=0.125, cuda=False):
#     try:
#         from wavelets_pytorch.transform import (
#             WaveletTransformTorch,
#         )  # PyTorch version
#     except ImportError:
#         raise ImportError(
#             "The spectrograms function requires the wavelets-pytorch package. "
#             "Install it with: pip install wavelets-pytorch"
#         )
#
#     dt = 1 / fs
#     # dj = 0.125
#     batch_size, n_chs, seq_len = x.shape
#
#     x = x.cpu().numpy()
#
#     # # Batch of signals to process
#     # batch = np.array([batch_size * seq_len])
#
#     # Initialize wavelet filter banks (scipy and torch implementation)
#     # wa_scipy = WaveletTransform(dt, dj)
#     wa_torch = WaveletTransformTorch(dt, dj, cuda=True)
#
#     # Performing wavelet transform (and compute scalogram)
#     # cwt_scipy = wa_scipy.cwt(batch)
#     x = x[:, 0][:, np.newaxis]
#     cwt_torch = wa_torch.cwt(x)
#
#     return cwt_torch
#
#
# if __name__ == "__main__":
#     import scitex
#     import seaborn as sns
#     import torchaudio
#
#     fs = 1024  # 128
#     t_sec = 10
#     x = scitex.dsp.np.demo_sig(t_sec=t_sec, fs=fs, type="ripple")
#
#     normalize(unbias(x, cuda=True), cuda=True)
#
#     # My implementtion
#     ss = spectrograms(x, fs, cuda=True)
#     fig, axes = plt.subplots(nrows=2)
#     axes[0].plot(np.arange(x[0, 0]) / fs, x[0, 0])
#     sns.heatmap(ss[0], ax=axes[1])
#     plt.show()
#
#     ss, ff, tt = spectrograms(x, fs, cuda=True)
#     fig, axes = plt.subplots(nrows=2)
#     axes[0].plot(np.arange(x[0, 0]) / fs, x[0, 0])
#     sns.heatmap(ss[0], ax=axes[1])
#     plt.show()
#
#     # Torch Audio
#     transform = torchaudio.transforms.Spectrogram(n_fft=16, normalized=True).cuda()
#     xx = torch.tensor(x).float().cuda()[0, 0]
#     ss = transform(xx)
#     sns.heatmap(ss.detach().cpu().numpy())
#
#     plt.show()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_Spectrogram.py
# --------------------------------------------------------------------------------

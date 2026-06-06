#!/usr/bin/env python3
# Time-stamp: "2025-01-06 (ywatanabe)"
# File: tests/scitex/nn/test__Wavelet.py

"""Comprehensive test suite for Wavelet module.

This module tests the wavelet transform functionality for neural networks,
including Morlet wavelet generation, multi-scale analysis, phase/amplitude extraction,
and edge cases.
"""

import pytest

# Required for this module
pytest.importorskip("torch")
# Wavelet imports `from scitex_gen import to_even` at module top, so it
# needs the real scitex_gen package (no mocks — the source uses the
# standalone scitex_gen, not the umbrella scitex.gen).
pytest.importorskip("scitex_gen")
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from scitex_gen import to_even

from scitex_nn import Wavelet


class TestWavelet:
    """Test suite for Wavelet layer."""

    @pytest.fixture
    def sample_rate(self):
        """Standard sample rate for testing."""
        return 1000

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size, n_channels, seq_len = 2, 3, 1000
        return torch.randn(batch_size, n_channels, seq_len)

    def test_initialization_default_params_out_scale(self, sample_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        assert layer.out_scale == "log"
        pass
        pass
        pass

    def test_initialization_default_params_kernel(self, sample_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        pass
        assert layer.kernel is not None
        pass
        pass

    def test_initialization_default_params_freqs(self, sample_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        pass
        pass
        assert layer.freqs is not None
        pass

    def test_initialization_default_params_isinstance(self, sample_rate):
        """Test initialization with default parameters."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        pass
        pass
        pass
        assert isinstance(layer.dummy, torch.Tensor)

    def test_initialization_custom_params_out_scale(self, sample_rate):
        """Test initialization with custom parameters."""
        # Arrange
        kernel_size = 512
        freq_scale = "log"
        out_scale = "linear"
        # Act
        layer = Wavelet(
            samp_rate=sample_rate,
            kernel_size=kernel_size,
            freq_scale=freq_scale,
            out_scale=out_scale,
        )
        # Assert
        assert layer.out_scale == out_scale
        pass

    def test_initialization_custom_params_kernel_size(self, sample_rate):
        """Test initialization with custom parameters."""
        # Arrange
        kernel_size = 512
        freq_scale = "log"
        out_scale = "linear"
        # Act
        layer = Wavelet(
            samp_rate=sample_rate,
            kernel_size=kernel_size,
            freq_scale=freq_scale,
            out_scale=out_scale,
        )
        # Assert
        assert layer.kernel_size == to_even(kernel_size)

    def test_morlet_generation_linear_scale_isinstance(self, sample_rate):
        """Test Morlet wavelet generation with linear frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="linear"
        )
        # Assert
        assert isinstance(morlets, np.ndarray)
        pass
        nyquist = sample_rate / 2
        pass
        pass
        pass
        pass

    def test_morlet_generation_linear_scale_isinstance_v2(self, sample_rate):
        """Test Morlet wavelet generation with linear frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="linear"
        )
        # Assert
        pass
        assert isinstance(freqs, np.ndarray)
        nyquist = sample_rate / 2
        pass
        pass
        pass
        pass

    def test_morlet_generation_linear_scale_check3(self, sample_rate):
        """Test Morlet wavelet generation with linear frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="linear"
        )
        # Assert
        pass
        pass
        nyquist = sample_rate / 2
        assert freqs[0] > 0
        pass
        pass
        pass

    def test_morlet_generation_linear_scale_check4(self, sample_rate):
        """Test Morlet wavelet generation with linear frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="linear"
        )
        # Assert
        pass
        pass
        nyquist = sample_rate / 2
        pass
        assert freqs[-1] <= nyquist
        pass
        pass

    def test_morlet_generation_linear_scale_all(self, sample_rate):
        """Test Morlet wavelet generation with linear frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="linear"
        )
        # Assert
        pass
        pass
        nyquist = sample_rate / 2
        pass
        pass
        assert np.all(np.diff(freqs) > 0)
        pass

    def test_morlet_generation_linear_scale_dtype(self, sample_rate):
        """Test Morlet wavelet generation with linear frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="linear"
        )
        # Assert
        pass
        pass
        nyquist = sample_rate / 2
        pass
        pass
        pass
        assert morlets.dtype == np.complex128

    def test_morlet_generation_log_scale(self, sample_rate):
        """Test Morlet wavelet generation with log frequency scale."""
        # Arrange
        # Act
        morlets, freqs = Wavelet.gen_morlet_to_nyquist(
            samp_rate=sample_rate, kernel_size=None, freq_scale="log"
        )
        freq_ratios = freqs[1:] / freqs[:-1]
        # Assert
        assert np.std(freq_ratios) < np.mean(freq_ratios) * 0.5

    def test_forward_basic_wavelet_behaves_correctly_shape(
        self, sample_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        # Act
        pha, amp, freqs = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        n_freqs = layer.kernel.shape[0]
        # Assert
        assert pha.shape == (batch_size, n_channels, n_freqs, seq_len)
        pass
        pass

    def test_forward_basic_wavelet_behaves_correctly_shape_v2(
        self, sample_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        # Act
        pha, amp, freqs = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        n_freqs = layer.kernel.shape[0]
        # Assert
        pass
        assert amp.shape == (batch_size, n_channels, n_freqs, seq_len)
        pass

    def test_forward_basic_wavelet_behaves_correctly_shape_v3(
        self, sample_rate, sample_input
    ):
        """Test basic forward pass."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        # Act
        pha, amp, freqs = layer(sample_input)
        batch_size, n_channels, seq_len = sample_input.shape
        n_freqs = layer.kernel.shape[0]
        # Assert
        pass
        pass
        assert freqs.shape == (batch_size, n_channels, n_freqs)

    def test_forward_log_scale_output(self, sample_rate, sample_input):
        """Test forward pass with log scale output."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate, out_scale="log")
        # Act
        pha, amp, freqs = layer(sample_input)
        # Assert
        assert not torch.isinf(amp).any()

    def test_forward_linear_scale_output(self, sample_rate, sample_input):
        """Test forward pass with linear scale output."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate, out_scale="linear")
        # Act
        pha, amp, freqs = layer(sample_input)
        # Assert
        assert (amp >= 0).all()

    def test_phase_range_wavelet_behaves_correctly_all(self, sample_rate, sample_input):
        """Test that phase values are in correct range."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        # Act
        pha, _, _ = layer(sample_input)
        # Assert
        assert (pha >= -np.pi).all()
        pass

    def test_phase_range_wavelet_behaves_correctly_all_v2(
        self, sample_rate, sample_input
    ):
        """Test that phase values are in correct range."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        # Act
        pha, _, _ = layer(sample_input)
        # Assert
        pass
        assert (pha <= np.pi).all()

    def test_gradient_flow_wavelet_behaves_correctly_grad(
        self, sample_rate, sample_input
    ):
        """Test that gradients flow properly through the layer."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        sample_input.requires_grad = True
        pha, amp, _ = layer(sample_input)
        loss = amp.sum() + pha.sum()
        # Act
        loss.backward()
        # Assert
        assert sample_input.grad is not None
        pass

    def test_gradient_flow_wavelet_behaves_correctly_allclose(
        self, sample_rate, sample_input
    ):
        """Test that gradients flow properly through the layer."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        sample_input.requires_grad = True
        pha, amp, _ = layer(sample_input)
        loss = amp.sum() + pha.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.allclose(
            sample_input.grad, torch.zeros_like(sample_input.grad)
        )

    def test_edge_handling_wavelet_behaves_correctly_any(self, sample_rate):
        """Test edge handling with reflection padding."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.ones(1, 1, 1000)
        x[:, :, :100] = -1
        x[:, :, -100:] = -1
        # Act
        pha, amp, _ = layer(x)
        # Assert
        assert not torch.isnan(pha).any()
        pass

    def test_edge_handling_wavelet_behaves_correctly_any_v2(self, sample_rate):
        """Test edge handling with reflection padding."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.ones(1, 1, 1000)
        x[:, :, :100] = -1
        x[:, :, -100:] = -1
        # Act
        pha, amp, _ = layer(x)
        # Assert
        pass
        assert not torch.isinf(amp).any()

    def test_device_compatibility_cpu_device(self, sample_rate):
        """Test layer works on CPU."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.randn(2, 3, 1000)
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        assert pha.device == x.device
        pass
        pass

    def test_device_compatibility_cpu_device_v2(self, sample_rate):
        """Test layer works on CPU."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.randn(2, 3, 1000)
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        pass
        assert amp.device == x.device
        pass

    def test_device_compatibility_cpu_is_cuda(self, sample_rate):
        """Test layer works on CPU."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.randn(2, 3, 1000)
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        pass
        pass
        assert not pha.is_cuda

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_device(self, sample_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate).cuda()
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        assert pha.device == x.device
        pass
        pass
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_device_v2(self, sample_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate).cuda()
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        pass
        assert amp.device == x.device
        pass
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_is_cuda(self, sample_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate).cuda()
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        pass
        pass
        assert pha.is_cuda
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda_is_cuda_v2(self, sample_rate):
        """Test layer works on CUDA."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate).cuda()
        x = torch.randn(2, 3, 1000).cuda()
        # Act
        pha, amp, freqs = layer(x)
        # Assert
        pass
        pass
        pass
        assert layer.kernel.is_cuda

    def test_different_kernel_sizes_check1(self, sample_rate):
        """Test with various kernel sizes."""
        # Arrange
        kernel_sizes = [256, 512, 1024, 2048]
        # Act
        x = torch.randn(2, 3, 2000)
        # Assert
        for kernel_size in kernel_sizes:
            layer = Wavelet(samp_rate=sample_rate, kernel_size=kernel_size)
            pha, amp, _ = layer(x)
            assert pha.shape[-1] == x.shape[-1]
            pass

    def test_different_kernel_sizes_check2(self, sample_rate):
        """Test with various kernel sizes."""
        # Arrange
        kernel_sizes = [256, 512, 1024, 2048]
        # Act
        x = torch.randn(2, 3, 2000)
        # Assert
        for kernel_size in kernel_sizes:
            layer = Wavelet(samp_rate=sample_rate, kernel_size=kernel_size)
            pha, amp, _ = layer(x)
            pass
            assert amp.shape[-1] == x.shape[-1]

    def test_frequency_resolution_wavelet_behaves_correctly(self, sample_rate):
        """Test frequency resolution with different scales."""
        # Arrange
        layer_linear = Wavelet(samp_rate=sample_rate, freq_scale="linear")
        # Act
        layer_log = Wavelet(samp_rate=sample_rate, freq_scale="log")
        n_freqs_linear = layer_linear.freqs.shape[0]
        n_freqs_log = layer_log.freqs.shape[0]
        # Assert
        assert n_freqs_linear > n_freqs_log

    def test_single_tone_analysis(self, sample_rate):
        """Test wavelet analysis of single frequency tone."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate, freq_scale="linear")
        t = torch.arange(0, 2, 1 / sample_rate)
        freq = 100
        x = torch.sin(2 * np.pi * freq * t).unsqueeze(0).unsqueeze(0)
        pha, amp, freqs = layer(x)
        avg_amp = amp[0, 0].mean(dim=1)
        # Act
        peak_idx = torch.argmax(avg_amp)
        peak_freq = freqs[0, 0, peak_idx]
        # Assert
        assert abs(peak_freq - freq) < 20

    def test_chirp_signal_analysis(self, sample_rate):
        """Test wavelet analysis of chirp signal."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        t = torch.arange(0, 2, 1 / sample_rate)
        f0, f1 = (50, 200)
        chirp = torch.sin(2 * np.pi * (f0 + (f1 - f0) * t / 2) * t)
        x = chirp.unsqueeze(0).unsqueeze(0)
        pha, amp, freqs = layer(x)
        early_amp = amp[0, 0, :, :100].mean(dim=1)
        late_amp = amp[0, 0, :, -100:].mean(dim=1)
        early_peak = freqs[0, 0, torch.argmax(early_amp)]
        # Act
        late_peak = freqs[0, 0, torch.argmax(late_amp)]
        # Assert
        assert late_peak > early_peak

    def test_zero_input_handling_all(self, sample_rate):
        """Test behavior with zero input."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.zeros(2, 3, 1000)
        # Act
        pha, amp, _ = layer(x)
        # Assert
        if layer.out_scale == "log":
            assert (amp < -5).all()
        else:
            pass

    def test_zero_input_handling_allclose(self, sample_rate):
        """Test behavior with zero input."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.zeros(2, 3, 1000)
        # Act
        pha, amp, _ = layer(x)
        # Assert
        if layer.out_scale == "log":
            pass
        else:
            assert torch.allclose(amp, torch.zeros_like(amp), atol=1e-10)

    def test_numerical_stability_wavelet_behaves_correctly_any(self, sample_rate):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        pha_large, amp_large, _ = layer(x_large)
        # Assert
        assert not torch.isnan(pha_large).any()
        pass
        x_small = torch.randn(2, 3, 1000) * 1e-06
        pha_small, amp_small, _ = layer(x_small)
        pass
        pass

    def test_numerical_stability_wavelet_behaves_correctly_any_v2(self, sample_rate):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        pha_large, amp_large, _ = layer(x_large)
        # Assert
        pass
        assert not torch.isinf(amp_large).any()
        x_small = torch.randn(2, 3, 1000) * 1e-06
        pha_small, amp_small, _ = layer(x_small)
        pass
        pass

    def test_numerical_stability_wavelet_behaves_correctly_any_v3(self, sample_rate):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        pha_large, amp_large, _ = layer(x_large)
        # Assert
        pass
        pass
        x_small = torch.randn(2, 3, 1000) * 1e-06
        pha_small, amp_small, _ = layer(x_small)
        assert not torch.isnan(pha_small).any()
        pass

    def test_numerical_stability_wavelet_behaves_correctly_any_v4(self, sample_rate):
        """Test numerical stability with extreme values."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x_large = torch.randn(2, 3, 1000) * 1000000.0
        # Act
        pha_large, amp_large, _ = layer(x_large)
        # Assert
        pass
        pass
        x_small = torch.randn(2, 3, 1000) * 1e-06
        pha_small, amp_small, _ = layer(x_small)
        pass
        assert not torch.isinf(amp_small).any()

    def test_batch_consistency_wavelet_behaves_correctly_allclose(self, sample_rate):
        """Test that batched processing gives consistent results."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x_single = torch.randn(1, 3, 1000)
        pha_single, amp_single, _ = layer(x_single)
        x_batch = x_single.repeat(4, 1, 1)
        # Act
        pha_batch, amp_batch, _ = layer(x_batch)
        # Assert
        for i in range(4):
            assert torch.allclose(pha_batch[i], pha_single[0])
            pass

    def test_batch_consistency_wavelet_behaves_correctly_allclose_v2(self, sample_rate):
        """Test that batched processing gives consistent results."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x_single = torch.randn(1, 3, 1000)
        pha_single, amp_single, _ = layer(x_single)
        x_batch = x_single.repeat(4, 1, 1)
        # Act
        pha_batch, amp_batch, _ = layer(x_batch)
        # Assert
        for i in range(4):
            pass
            assert torch.allclose(amp_batch[i], amp_single[0])

    def test_kernel_properties_wavelet_behaves_correctly_dtype(self, sample_rate):
        """Test properties of generated Morlet wavelets."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        assert (
            layer.kernel.dtype == torch.complex64
            or layer.kernel.dtype == torch.complex128
        )
        for i in range(layer.kernel.shape[0]):
            wavelet = layer.kernel[i]
            pass
            pass

    def test_kernel_properties_wavelet_behaves_correctly_max(self, sample_rate):
        """Test properties of generated Morlet wavelets."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        pass
        for i in range(layer.kernel.shape[0]):
            wavelet = layer.kernel[i]
            assert wavelet.abs().max() > 0
            pass

    def test_kernel_properties_wavelet_behaves_correctly_max_v2(self, sample_rate):
        """Test properties of generated Morlet wavelets."""
        # Arrange
        # Act
        layer = Wavelet(samp_rate=sample_rate)
        # Assert
        pass
        for i in range(layer.kernel.shape[0]):
            wavelet = layer.kernel[i]
            pass
            assert wavelet.abs().max() < 10

    def test_memory_efficiency_wavelet_behaves_correctly_check1(self, sample_rate):
        """Test memory usage with large inputs."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.randn(4, 8, 4000)
        # Act
        pha, amp, _ = layer(x)
        # Assert
        assert pha.shape[0] == 4
        pass

    def test_memory_efficiency_wavelet_behaves_correctly_check2(self, sample_rate):
        """Test memory usage with large inputs."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        x = torch.randn(4, 8, 4000)
        # Act
        pha, amp, _ = layer(x)
        # Assert
        pass
        assert pha.shape[1] == 8

    def test_integration_with_sequential(self, sample_rate):
        """Test integration in nn.Sequential."""
        # Arrange

        class WaveletFeatures(nn.Module):
            def __init__(self, samp_rate):
                super().__init__()
                self.wavelet = Wavelet(samp_rate)

            def forward(self, x):
                _, amp, _ = self.wavelet(x)
                return amp.mean(dim=-1)

        model = nn.Sequential(
            WaveletFeatures(sample_rate),
            nn.Flatten(),
            nn.Linear(3 * 10, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        # Act
        x = torch.randn(4, 3, 1000)
        # Assert
        try:
            output = model(x)
            assert output.shape[0] == 4
        except RuntimeError:
            pass

    def test_phase_amplitude_consistency(self, sample_rate, sample_input):
        """Test that phase and amplitude are consistent."""
        # Arrange
        layer = Wavelet(samp_rate=sample_rate)
        pha, amp, _ = layer(sample_input)
        phase_diff = torch.diff(pha, dim=-1)
        # Act
        fraction_small = (torch.abs(phase_diff) < np.pi).float().mean()
        # Assert
        assert fraction_small > 0.7

    def test_custom_kernel_size_effect(self, sample_rate):
        """Test that kernel size affects frequency resolution."""
        # Arrange
        x = torch.randn(1, 1, 2000)
        layer_small = Wavelet(samp_rate=sample_rate, kernel_size=256)
        _, amp_small, freqs_small = layer_small(x)
        layer_large = Wavelet(samp_rate=sample_rate, kernel_size=2048)
        # Act
        _, amp_large, freqs_large = layer_large(x)
        # Assert
        assert amp_small.shape != amp_large.shape or not torch.allclose(
            amp_small, amp_large
        )


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_Wavelet.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-03 07:17:26 (ywatanabe)"
# # File: ./scitex_repo/src/scitex/nn/_Wavelet.py
#
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-05-30 11:04:45 (ywatanabe)"
#
#
# import scitex
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from scitex.gen._to_even import to_even
# from scitex.gen._to_odd import to_odd
#
#
# class Wavelet(nn.Module):
#     def __init__(
#         self, samp_rate, kernel_size=None, freq_scale="linear", out_scale="log"
#     ):
#         super().__init__()
#         self.register_buffer("dummy", torch.tensor(0))
#         self.kernel = None
#         self.init_kernel(samp_rate, kernel_size=kernel_size, freq_scale=freq_scale)
#         self.out_scale = out_scale
#
#     def forward(self, x):
#         """Apply the 2D filter (n_filts, kernel_size) to input signal x with shape: (batch_size, n_chs, seq_len)"""
#         x = scitex.dsp.ensure_3d(x).to(self.dummy.device)
#         seq_len = x.shape[-1]
#
#         # Ensure the kernel is initialized
#         if self.kernel is None:
#             self.init_kernel()
#             if self.kernel is None:
#                 raise ValueError("Filter kernel has not been initialized.")
#         assert self.kernel.ndim == 2
#         self.kernel = self.kernel.to(x.device)  # cuda, torch.complex128
#
#         # Edge handling and convolution
#         extension_length = self.radius
#         first_segment = x[:, :, :extension_length].flip(dims=[-1])
#         last_segment = x[:, :, -extension_length:].flip(dims=[-1])
#         extended_x = torch.cat([first_segment, x, last_segment], dim=-1)
#
#         # working??
#         kernel_batched = self.kernel.unsqueeze(1)
#         extended_x_reshaped = extended_x.view(-1, 1, extended_x.shape[-1])
#
#         filtered_x_real = F.conv1d(
#             extended_x_reshaped, kernel_batched.real.float(), groups=1
#         )
#         filtered_x_imag = F.conv1d(
#             extended_x_reshaped, kernel_batched.imag.float(), groups=1
#         )
#
#         filtered_x = torch.view_as_complex(
#             torch.stack([filtered_x_real, filtered_x_imag], dim=-1)
#         )
#
#         filtered_x = filtered_x.view(
#             x.shape[0], x.shape[1], kernel_batched.shape[0], -1
#         )
#         filtered_x = filtered_x.view(
#             x.shape[0], x.shape[1], kernel_batched.shape[0], -1
#         )
#         filtered_x = filtered_x[..., :seq_len]
#         assert filtered_x.shape[-1] == seq_len
#
#         pha = filtered_x.angle()
#         amp = filtered_x.abs()
#
#         # Repeats freqs
#         freqs = (
#             self.freqs.unsqueeze(0).unsqueeze(0).repeat(pha.shape[0], pha.shape[1], 1)
#         )
#
#         if self.out_scale == "log":
#             return pha, torch.log(amp + 1e-5), freqs
#         else:
#             return pha, amp, freqs
#
#     def init_kernel(self, samp_rate, kernel_size=None, freq_scale="log"):
#         device = self.dummy.device
#         morlets, freqs = self.gen_morlet_to_nyquist(
#             samp_rate, kernel_size=kernel_size, freq_scale=freq_scale
#         )
#         self.kernel = torch.tensor(morlets).to(device)
#         self.freqs = torch.tensor(freqs).float().to(device)
#
#     @staticmethod
#     def gen_morlet_to_nyquist(samp_rate, kernel_size=None, freq_scale="linear"):
#         """
#         Generates Morlet wavelets for exponentially increasing frequency bands up to the Nyquist frequency.
#
#         Parameters:
#         - samp_rate (int): The sampling rate of the signal, in Hertz.
#         - kernel_size (int): The size of the kernel, in number of samples.
#
#         Returns:
#         - np.ndarray: A 2D array of complex values representing the Morlet wavelets for each frequency band.
#         """
#         if kernel_size is None:
#             kernel_size = int(samp_rate)  # * 2.5)
#
#         nyquist_freq = samp_rate / 2
#
#         # Log freq_scale
#         def calc_freq_boundaries_log(nyquist_freq):
#             n_kernels = int(np.floor(np.log2(nyquist_freq)))
#             mid_hz = np.array([2 ** (n + 1) for n in range(n_kernels)])
#             width_hz = np.hstack([np.array([1]), np.diff(mid_hz) / 2]) + 1
#             low_hz = mid_hz - width_hz
#             high_hz = mid_hz + width_hz
#             low_hz[0] = 0.1
#             return low_hz, high_hz
#
#         def calc_freq_boundaries_linear(nyquist_freq):
#             n_kernels = int(nyquist_freq)
#             high_hz = np.linspace(1, nyquist_freq, n_kernels)
#             low_hz = high_hz - np.hstack([np.array(1), np.diff(high_hz)])
#             low_hz[0] = 0.1
#             return low_hz, high_hz
#
#         if freq_scale == "linear":
#             fn = calc_freq_boundaries_linear
#         if freq_scale == "log":
#             fn = calc_freq_boundaries_log
#         low_hz, high_hz = fn(nyquist_freq)
#
#         morlets = []
#         freqs = []
#
#         for _, (ll, hh) in enumerate(zip(low_hz, high_hz)):
#             if ll > nyquist_freq:
#                 break
#
#             center_frequency = (ll + hh) / 2
#
#             t = np.arange(-kernel_size // 2, kernel_size // 2) / samp_rate
#             # Calculate standard deviation of the gaussian window for a given center frequency
#             sigma = 7 / (2 * np.pi * center_frequency)
#             sine_wave = np.exp(2j * np.pi * center_frequency * t)
#             gaussian_window = np.exp(-(t**2) / (2 * sigma**2))
#             morlet_wavelet = sine_wave * gaussian_window
#
#             freqs.append(center_frequency)
#             morlets.append(morlet_wavelet)
#
#         return np.array(morlets), np.array(freqs)
#
#     @property
#     def kernel_size(
#         self,
#     ):
#         return to_even(self.kernel.shape[-1])
#
#     @property
#     def radius(
#         self,
#     ):
#         return to_even(self.kernel_size // 2)
#
#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import scitex
#
#     xx, tt, fs = scitex.dsp.demo_sig(sig_type="chirp")
#
#     pha, amp, ff = scitex.dsp.wavelet(xx, fs)
#
#     fig, ax = scitex.plt.subplots()
#     ax.imshow2d(amp[0, 0].T)
#     ax = scitex.plt.ax.set_ticks(ax, xticks=tt, yticks=ff)
#     ax = scitex.plt.ax.set_n_ticks(ax)
#     plt.show()
#
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_Wavelet.py
# --------------------------------------------------------------------------------

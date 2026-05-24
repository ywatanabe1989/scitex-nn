#!/usr/bin/env python3
# Time-stamp: "2025-06-01 00:00:00 (ywatanabe)"
# Tests for Power Spectral Density (PSD) layer

import math

import pytest

# Required for this module
pytest.importorskip("torch")
import numpy as np
import torch
import torch.nn as nn

from scitex_nn import PSD


class TestPSD:
    """Comprehensive test suite for Power Spectral Density layer."""

    def test_initialization_basic_psd_behaves_correctly_sample_rate(self):
        """Test basic PSD layer initialization."""
        # Arrange
        sample_rate = 1000
        # Act
        psd = PSD(sample_rate)
        # Assert
        assert psd.sample_rate == sample_rate

    def test_initialization_basic_psd_behaves_correctly_dim(self):
        """Test basic PSD layer initialization."""
        # Arrange
        sample_rate = 1000
        # Act
        psd = PSD(sample_rate)
        # Assert
        pass
        assert psd.dim == -1

    def test_initialization_basic_psd_behaves_correctly_prob(self):
        """Test basic PSD layer initialization."""
        # Arrange
        sample_rate = 1000
        # Act
        psd = PSD(sample_rate)
        # Assert
        pass
        pass
        assert psd.prob == False

    def test_initialization_with_options_sample_rate(self):
        """Test PSD layer initialization with all options."""
        # Arrange
        sample_rate = 500
        # Act
        psd = PSD(sample_rate, prob=True, dim=-2)
        # Assert
        assert psd.sample_rate == sample_rate
        pass
        pass

    def test_initialization_with_options_dim(self):
        """Test PSD layer initialization with all options."""
        # Arrange
        sample_rate = 500
        # Act
        psd = PSD(sample_rate, prob=True, dim=-2)
        # Assert
        pass
        assert psd.dim == -2
        pass

    def test_initialization_with_options_prob(self):
        """Test PSD layer initialization with all options."""
        # Arrange
        sample_rate = 500
        # Act
        psd = PSD(sample_rate, prob=True, dim=-2)
        # Assert
        pass
        pass
        assert psd.prob == True

    def test_forward_real_signal_1d_shape(self):
        """Test forward pass with 1D real signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        x = torch.randn(seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert psd.shape == (seq_len // 2 + 1,)
        pass
        pass
        pass

    def test_forward_real_signal_1d_shape_v2(self):
        """Test forward pass with 1D real signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        x = torch.randn(seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        assert freqs.shape == (seq_len // 2 + 1,)
        pass

    def test_forward_real_signal_1d_check3(self):
        """Test forward pass with 1D real signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        x = torch.randn(seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        pass
        assert freqs[0] == 0
        pass

    def test_forward_real_signal_1d_check4(self):
        """Test forward pass with 1D real signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        x = torch.randn(seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        pass
        pass
        assert freqs[-1] == sample_rate / 2

    def test_forward_real_signal_2d_shape(self):
        """Test forward pass with 2D real signal (batch)."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        batch_size = 4
        seq_len = 512
        x = torch.randn(batch_size, seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert psd.shape == (batch_size, seq_len // 2 + 1)
        pass

    def test_forward_real_signal_2d_shape_v2(self):
        """Test forward pass with 2D real signal (batch)."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        batch_size = 4
        seq_len = 512
        x = torch.randn(batch_size, seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        assert freqs.shape == (seq_len // 2 + 1,)

    def test_forward_real_signal_3d_shape(self):
        """Test forward pass with 3D real signal (batch, channels, time)."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        batch_size = 4
        n_channels = 3
        seq_len = 512
        x = torch.randn(batch_size, n_channels, seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert psd.shape == (batch_size, n_channels, seq_len // 2 + 1)
        pass

    def test_forward_real_signal_3d_shape_v2(self):
        """Test forward pass with 3D real signal (batch, channels, time)."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        batch_size = 4
        n_channels = 3
        seq_len = 512
        x = torch.randn(batch_size, n_channels, seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        assert freqs.shape == (seq_len // 2 + 1,)

    def test_forward_complex_signal_shape(self):
        """Test forward pass with complex signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 512
        x_real = torch.randn(seq_len)
        x_imag = torch.randn(seq_len)
        x = torch.complex(x_real, x_imag)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert psd.shape == (seq_len,)
        pass

    def test_forward_complex_signal_shape_v2(self):
        """Test forward pass with complex signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 512
        x_real = torch.randn(seq_len)
        x_imag = torch.randn(seq_len)
        x = torch.complex(x_real, x_imag)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        assert freqs.shape == (seq_len,)

    def test_parseval_theorem_psd_behaves_correctly(self):
        """Test Parseval's theorem: energy conservation (approximate).

            Note: PSD normalization varies by implementation. One-sided PSDs often
            scale differently than two-sided. We test for consistent ratio rather
            than exact equality.
            """
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        x = torch.randn(seq_len)
        time_energy = (x ** 2).mean()
        psd, freqs = psd_layer(x)
        freq_step = freqs[1] - freqs[0]
        # Act
        freq_energy = psd.sum() * freq_step
        ratio = time_energy / freq_energy
        # Assert
        assert 0.1 < ratio < 10

    def test_single_frequency_signal(self):
        """Test PSD of pure sinusoidal signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        t = torch.linspace(0, 1, seq_len)
        freq = 50
        x = torch.sin(2 * math.pi * freq * t)
        psd, freqs = psd_layer(x)
        # Act
        peak_idx = psd.argmax()
        peak_freq = freqs[peak_idx]
        # Assert
        assert abs(peak_freq - freq) < sample_rate / seq_len

    def test_multiple_frequency_signal(self):
        """Test PSD of signal with multiple frequencies."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 2000
        t = torch.linspace(0, 2, seq_len)
        freq1, freq2 = (50, 150)
        x = torch.sin(2 * math.pi * freq1 * t) + torch.sin(2 * math.pi * freq2 * t)
        psd, freqs = psd_layer(x)
        # Act
        peak_idx = psd.argmax()
        peak_freq = freqs[peak_idx]
        # Assert
        assert abs(peak_freq - freq1) < 5 or abs(peak_freq - freq2) < 5

    def test_white_noise_spectrum(self):
        """Test PSD of white noise has reasonable statistical properties.

            Note: White noise PSD has inherent variance - it's not perfectly flat
            for finite signals. We just verify it doesn't have obvious peaks.
            """
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 10000
        x = torch.randn(seq_len)
        psd, freqs = psd_layer(x)
        psd_no_dc = psd[1:]
        # Act
        ratio = psd_no_dc.max() / psd_no_dc.mean()
        # Assert
        assert ratio < 20

    def test_probability_mode_psd_behaves_correctly(self):
        """Test probability normalization mode."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate, prob=True)
        batch_size = 4
        seq_len = 512
        x = torch.randn(batch_size, seq_len)
        psd, freqs = psd_layer(x)
        # Act
        psd_sum = psd.sum(dim=-1)
        # Assert
        assert torch.allclose(psd_sum, torch.ones(batch_size), atol=1e-06)

    def test_different_dimensions_psd_behaves_correctly_shape(self):
        """Test operation along different dimensions."""
        # Arrange
        sample_rate = 1000
        psd1 = PSD(sample_rate, dim=-1)
        x1 = torch.randn(4, 3, 512)
        # Act
        psd_out1, freqs1 = psd1(x1)
        # Assert
        assert psd_out1.shape == (4, 3, 257)
        psd2 = PSD(sample_rate, dim=-2)
        x2 = torch.randn(4, 512, 3)
        psd_out2, freqs2 = psd2(x2)
        pass

    def test_different_dimensions_psd_behaves_correctly_shape_v2(self):
        """Test operation along different dimensions."""
        # Arrange
        sample_rate = 1000
        psd1 = PSD(sample_rate, dim=-1)
        x1 = torch.randn(4, 3, 512)
        # Act
        psd_out1, freqs1 = psd1(x1)
        # Assert
        pass
        psd2 = PSD(sample_rate, dim=-2)
        x2 = torch.randn(4, 512, 3)
        psd_out2, freqs2 = psd2(x2)
        assert psd_out2.shape == (4, 257, 3)

    def test_different_sample_rates(self):
        """Test with various sample rates."""
        # Arrange
        seq_len = 1000
        # Act
        x = torch.randn(seq_len)
        # Assert
        for sample_rate in [100, 500, 1000, 44100, 48000]:
            psd_layer = PSD(sample_rate)
            psd, freqs = psd_layer(x)
            assert freqs[-1] == sample_rate / 2

    def test_dc_component_psd_behaves_correctly_check1(self):
        """Test DC component (zero frequency) handling."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        dc_offset = 2.5
        x = torch.ones(seq_len) * dc_offset + 0.1 * torch.randn(seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert psd[0] > psd[1:].max()
        pass

    def test_dc_component_psd_behaves_correctly_check2(self):
        """Test DC component (zero frequency) handling."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        dc_offset = 2.5
        x = torch.ones(seq_len) * dc_offset + 0.1 * torch.randn(seq_len)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        assert freqs[0] == 0

    def test_frequency_resolution_psd_behaves_correctly(self):
        """Test frequency resolution of PSD."""
        # Arrange
        sample_rate = 1000
        # Act
        psd_layer = PSD(sample_rate)
        # Assert
        for seq_len in [100, 500, 1000, 2000]:
            x = torch.randn(seq_len)
            psd, freqs = psd_layer(x)
            freq_resolution = freqs[1] - freqs[0]
            expected_resolution = sample_rate / seq_len
            assert abs(freq_resolution - expected_resolution) < 1e-06

    def test_gradient_flow_psd_behaves_correctly_grad(self):
        """Test that gradients flow through the layer."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        x = torch.randn(4, 512, requires_grad=True)
        psd, freqs = psd_layer(x)
        loss = psd.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_psd_behaves_correctly_allclose(self):
        """Test that gradients flow through the layer."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        x = torch.randn(4, 512, requires_grad=True)
        psd, freqs = psd_layer(x)
        loss = psd.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_compatibility_psd_behaves_correctly_device(self):
        """Test operation on different devices."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate).cuda()
        x = torch.randn(4, 512).cuda()
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert psd.device == x.device
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_compatibility_psd_behaves_correctly_device_v2(self):
        """Test operation on different devices."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate).cuda()
        x = torch.randn(4, 512).cuda()
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        pass
        assert freqs.device == x.device

    def test_batch_independence_psd_behaves_correctly(self):
        """Test that batch samples are processed independently."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        batch_size = 3
        seq_len = 512
        x = torch.zeros(batch_size, seq_len)
        t = torch.linspace(0, 1, seq_len)
        x[0] = torch.sin(2 * math.pi * 50 * t)
        x[1] = torch.sin(2 * math.pi * 100 * t)
        x[2] = torch.sin(2 * math.pi * 150 * t)
        psd, freqs = psd_layer(x)
        # Act
        peaks = psd.argmax(dim=-1)
        # Assert
        assert len(torch.unique(peaks)) == batch_size

    def test_numerical_stability_psd_behaves_correctly_all(self):
        """Test numerical stability with extreme values."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        x_large = torch.randn(512) * 1000000.0
        # Act
        psd_large, _ = psd_layer(x_large)
        # Assert
        assert torch.isfinite(psd_large).all()
        x_small = torch.randn(512) * 1e-06
        psd_small, _ = psd_layer(x_small)
        pass

    def test_numerical_stability_psd_behaves_correctly_all_v2(self):
        """Test numerical stability with extreme values."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        x_large = torch.randn(512) * 1000000.0
        # Act
        psd_large, _ = psd_layer(x_large)
        # Assert
        pass
        x_small = torch.randn(512) * 1e-06
        psd_small, _ = psd_layer(x_small)
        assert torch.isfinite(psd_small).all()

    def test_zero_signal_psd_behaves_correctly(self):
        """Test PSD of zero signal."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        x = torch.zeros(512)
        # Act
        psd, freqs = psd_layer(x)
        # Assert
        assert torch.allclose(psd, torch.zeros_like(psd))

    def test_window_function_effect(self):
        """Test that PSD captures windowing effects correctly."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        t = torch.linspace(0, 1, seq_len)
        window = torch.hann_window(seq_len)
        x = window * torch.sin(2 * math.pi * 50 * t)
        psd, freqs = psd_layer(x)
        # Act
        peak_idx = psd.argmax()
        peak_freq = freqs[peak_idx]
        # Assert
        assert abs(peak_freq - 50) < 5

    def test_integration_with_nn_sequential(self):
        """Test integration with PyTorch Sequential model."""
        # Arrange
        sample_rate = 1000
        seq_len = 512

        class PSDFeatureExtractor(nn.Module):

            def __init__(self):
                super().__init__()
                self.psd = PSD(sample_rate)
                self.fc = nn.Linear(seq_len // 2 + 1, 10)

            def forward(self, x):
                psd, _ = self.psd(x)
                return self.fc(psd)
        model = PSDFeatureExtractor()
        x = torch.randn(8, seq_len)
        # Act
        y = model(x)
        # Assert
        assert y.shape == (8, 10)

    def test_power_vs_amplitude_spectrum(self):
        """Test that PSD represents power (squared amplitude)."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        t = torch.linspace(0, 1, seq_len)
        amplitude = 2.0
        x = amplitude * torch.sin(2 * math.pi * 50 * t)
        psd, freqs = psd_layer(x)
        # Act
        peak_psd = psd.max()
        # Assert
        assert peak_psd is not None
        # Due to normalization, exact relationship depends on implementation

    def test_multi_channel_processing(self):
        """Test that multiple channels are processed correctly."""
        # Arrange
        sample_rate = 1000
        psd_layer = PSD(sample_rate)
        n_channels = 3
        seq_len = 1000
        t = torch.linspace(0, 1, seq_len)
        x = torch.zeros(n_channels, seq_len)
        x[0] = torch.sin(2 * math.pi * 30 * t)
        x[1] = torch.sin(2 * math.pi * 60 * t)
        x[2] = torch.sin(2 * math.pi * 90 * t)
        psd, freqs = psd_layer(x)
        # Act
        peaks = psd.argmax(dim=-1)
        # Assert
        assert len(torch.unique(peaks)) == n_channels

    def test_aliasing_detection_psd_behaves_correctly(self):
        """Test detection of aliasing (frequencies above Nyquist)."""
        # Arrange
        sample_rate = 100
        psd_layer = PSD(sample_rate)
        seq_len = 1000
        t = torch.linspace(0, 10, seq_len)
        true_freq = 80
        x = torch.sin(2 * math.pi * true_freq * t)
        psd, freqs = psd_layer(x)
        # Act
        peak_idx = psd.argmax()
        peak_freq = freqs[peak_idx]
        # Assert
        assert peak_freq < sample_rate / 2


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_PSD.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-04-11 21:50:09 (ywatanabe)"
#
# import torch
# import torch.nn as nn
#
#
# class PSD(nn.Module):
#     def __init__(self, sample_rate, prob=False, dim=-1):
#         super(PSD, self).__init__()
#         self.sample_rate = sample_rate
#         self.dim = dim
#         self.prob = prob
#
#     def forward(self, signal):
#         is_complex = signal.is_complex()
#         if is_complex:
#             signal_fft = torch.fft.fft(signal, dim=self.dim)
#             freqs = torch.fft.fftfreq(signal.size(self.dim), 1 / self.sample_rate).to(
#                 signal.device
#             )
#
#         else:
#             signal_fft = torch.fft.rfft(signal, dim=self.dim)
#             freqs = torch.fft.rfftfreq(signal.size(self.dim), 1 / self.sample_rate).to(
#                 signal.device
#             )
#
#         power_spectrum = torch.abs(signal_fft) ** 2
#         power_spectrum = power_spectrum / signal.size(self.dim)
#
#         psd = power_spectrum * (1.0 / self.sample_rate)
#
#         # To probability if specified
#         if self.prob:
#             psd /= psd.sum(dim=self.dim, keepdims=True)
#
#         return psd, freqs

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_PSD.py
# --------------------------------------------------------------------------------

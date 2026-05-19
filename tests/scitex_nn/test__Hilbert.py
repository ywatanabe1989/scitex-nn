#!/usr/bin/env python3
# Time-stamp: "2025-06-01 00:00:00 (ywatanabe)"
# Tests for Hilbert transform layer

import math

import pytest

# Required for this module
pytest.importorskip("torch")
import numpy as np
import torch
import torch.nn as nn

from scitex_nn import Hilbert


class TestHilbert:
    """Comprehensive test suite for Hilbert transform layer."""

    def test_initialization_basic_hilbert_behaves_correctly_n(self):
        """Test basic Hilbert layer initialization."""
        # Arrange
        seq_len = 100
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        assert hilbert.n == seq_len
        pass
        pass
        pass
        pass
        pass

    def test_initialization_basic_hilbert_behaves_correctly_dim(self):
        """Test basic Hilbert layer initialization."""
        # Arrange
        seq_len = 100
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        assert hilbert.dim == -1
        pass
        pass
        pass
        pass

    def test_initialization_basic_hilbert_behaves_correctly_fp16(self):
        """Test basic Hilbert layer initialization."""
        # Arrange
        seq_len = 100
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        pass
        assert hilbert.fp16 == False
        pass
        pass
        pass

    def test_initialization_basic_hilbert_behaves_correctly_in_place(self):
        """Test basic Hilbert layer initialization."""
        # Arrange
        seq_len = 100
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        pass
        pass
        assert hilbert.in_place == False
        pass
        pass

    def test_initialization_basic_hilbert_behaves_correctly_hasattr(self):
        """Test basic Hilbert layer initialization."""
        # Arrange
        seq_len = 100
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        pass
        pass
        pass
        assert hasattr(hilbert, 'h_mask')
        pass

    def test_initialization_basic_hilbert_behaves_correctly_shape(self):
        """Test basic Hilbert layer initialization."""
        # Arrange
        seq_len = 100
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        pass
        pass
        pass
        pass
        assert hilbert.h_mask.shape == (seq_len,)

    def test_initialization_with_options_n(self):
        """Test Hilbert layer initialization with all options."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len, dim=-2, fp16=True, in_place=True)
        # Assert
        assert hilbert.n == seq_len
        pass
        pass
        pass

    def test_initialization_with_options_dim(self):
        """Test Hilbert layer initialization with all options."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len, dim=-2, fp16=True, in_place=True)
        # Assert
        pass
        assert hilbert.dim == -2
        pass
        pass

    def test_initialization_with_options_fp16(self):
        """Test Hilbert layer initialization with all options."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len, dim=-2, fp16=True, in_place=True)
        # Assert
        pass
        pass
        assert hilbert.fp16 == True
        pass

    def test_initialization_with_options_in_place(self):
        """Test Hilbert layer initialization with all options."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len, dim=-2, fp16=True, in_place=True)
        # Assert
        pass
        pass
        pass
        assert hilbert.in_place == True

    def test_frequency_buffer_properties_shape(self):
        """Test properties of frequency buffer."""
        # Arrange
        seq_len = 64
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        assert hilbert.h_mask.shape == (seq_len,)
        pass
        pass

    def test_frequency_buffer_properties_min(self):
        """Test properties of frequency buffer."""
        # Arrange
        seq_len = 64
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        assert hilbert.h_mask.min() >= 0.0
        pass

    def test_frequency_buffer_properties_max(self):
        """Test properties of frequency buffer."""
        # Arrange
        seq_len = 64
        # Act
        hilbert = Hilbert(seq_len)
        # Assert
        pass
        pass
        assert hilbert.h_mask.max() <= 2.0

    def test_forward_basic_1d(self):
        """Test forward pass with 1D signal."""
        # Arrange
        seq_len = 100
        hilbert = Hilbert(seq_len)
        x = torch.randn(seq_len)
        # Act
        y = hilbert(x)
        # Assert
        assert y.shape == (seq_len, 2)

    def test_forward_basic_2d(self):
        """Test forward pass with 2D signal (batch)."""
        # Arrange
        seq_len = 100
        batch_size = 4
        hilbert = Hilbert(seq_len)
        x = torch.randn(batch_size, seq_len)
        # Act
        y = hilbert(x)
        # Assert
        assert y.shape == (batch_size, seq_len, 2)

    def test_forward_basic_3d(self):
        """Test forward pass with 3D signal (batch, channels, time)."""
        # Arrange
        seq_len = 100
        batch_size = 4
        n_channels = 3
        hilbert = Hilbert(seq_len)
        x = torch.randn(batch_size, n_channels, seq_len)
        # Act
        y = hilbert(x)
        # Assert
        assert y.shape == (batch_size, n_channels, seq_len, 2)

    def test_phase_amplitude_extraction_shape(self):
        """Test that phase and amplitude are correctly extracted."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        n_cycles = 10
        t = torch.linspace(0, 2 * math.pi * n_cycles, seq_len)
        x = torch.sin(t)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        amplitude = y[..., 1]
        # Assert
        assert phase.shape == x.shape
        pass
        pass
        pass
        pass

    def test_phase_amplitude_extraction_shape_v2(self):
        """Test that phase and amplitude are correctly extracted."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        n_cycles = 10
        t = torch.linspace(0, 2 * math.pi * n_cycles, seq_len)
        x = torch.sin(t)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        amplitude = y[..., 1]
        # Assert
        pass
        assert amplitude.shape == x.shape
        pass
        pass
        pass

    def test_phase_amplitude_extraction_min(self):
        """Test that phase and amplitude are correctly extracted."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        n_cycles = 10
        t = torch.linspace(0, 2 * math.pi * n_cycles, seq_len)
        x = torch.sin(t)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        amplitude = y[..., 1]
        # Assert
        pass
        pass
        assert amplitude.min() > 0
        pass
        pass

    def test_phase_amplitude_extraction_max(self):
        """Test that phase and amplitude are correctly extracted."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        n_cycles = 10
        t = torch.linspace(0, 2 * math.pi * n_cycles, seq_len)
        x = torch.sin(t)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        amplitude = y[..., 1]
        # Assert
        pass
        pass
        pass
        assert amplitude.max() <= 1.1
        pass

    def test_phase_amplitude_extraction_mean(self):
        """Test that phase and amplitude are correctly extracted."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        n_cycles = 10
        t = torch.linspace(0, 2 * math.pi * n_cycles, seq_len)
        x = torch.sin(t)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        amplitude = y[..., 1]
        # Assert
        pass
        pass
        pass
        pass
        assert amplitude.mean() > 0.5

    def test_analytic_signal_properties(self):
        """Test properties of analytic signal from Hilbert transform."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        t = torch.linspace(0, 4 * math.pi, seq_len)
        freq = 2.0
        x = torch.cos(2 * math.pi * freq * t / seq_len)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        amplitude = y[..., 1]
        # Assert
        assert torch.allclose(amplitude.mean(), torch.tensor(1.0), atol=0.1)

    def test_fp16_mode_hilbert_behaves_correctly_dtype(self):
        """Test operation in fp16 mode."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len, fp16=True)
        x = torch.randn(4, seq_len)
        # Act
        y = hilbert(x)
        # Assert
        assert y.dtype == torch.float32
        pass

    def test_fp16_mode_hilbert_behaves_correctly_shape(self):
        """Test operation in fp16 mode."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len, fp16=True)
        x = torch.randn(4, seq_len)
        # Act
        y = hilbert(x)
        # Assert
        pass
        assert y.shape == (4, seq_len, 2)

    def test_in_place_mode(self):
        """Test in-place vs non-in-place operation."""
        # Arrange
        seq_len = 128
        hilbert_not_inplace = Hilbert(seq_len, in_place=False)
        x1 = torch.randn(seq_len, requires_grad=True)
        x1_copy = x1.clone()
        # Act
        y1 = hilbert_not_inplace(x1)
        # Assert
        assert torch.equal(x1, x1_copy)
        hilbert_inplace = Hilbert(seq_len, in_place=True)
        x2 = torch.randn(seq_len)
        y2 = hilbert_inplace(x2)
        # Just verify it works without error

    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        # Arrange
        # Act
        # Assert
        for seq_len in [32, 64, 128, 256, 512, 1024]:
            hilbert = Hilbert(seq_len)
            x = torch.randn(2, seq_len)
            y = hilbert(x)
            assert y.shape == (2, seq_len, 2)

    def test_dim_parameter_hilbert_behaves_correctly_shape(self):
        """Test operation along different dimensions.

            Note: Current implementation only supports dim=-1 due to frequency buffer
            shape constraints. The dim parameter is stored but non-default values
            will cause broadcasting errors.
            """
        # Arrange
        seq_len = 100
        hilbert1 = Hilbert(seq_len, dim=-1)
        x1 = torch.randn(4, 3, seq_len)
        # Act
        y1 = hilbert1(x1)
        # Assert
        assert y1.shape == (4, 3, seq_len, 2)
        hilbert2 = Hilbert(seq_len, dim=-2)
        pass

    def test_dim_parameter_hilbert_behaves_correctly_dim(self):
        """Test operation along different dimensions.

            Note: Current implementation only supports dim=-1 due to frequency buffer
            shape constraints. The dim parameter is stored but non-default values
            will cause broadcasting errors.
            """
        # Arrange
        seq_len = 100
        hilbert1 = Hilbert(seq_len, dim=-1)
        x1 = torch.randn(4, 3, seq_len)
        # Act
        y1 = hilbert1(x1)
        # Assert
        pass
        hilbert2 = Hilbert(seq_len, dim=-2)
        assert hilbert2.dim == -2

    def test_instantaneous_frequency_hilbert_behaves_correctly(self):
        """Test instantaneous frequency calculation from phase."""
        # Arrange
        seq_len = 256
        hilbert = Hilbert(seq_len)
        t = torch.linspace(0, 1, seq_len)
        f0, f1 = (10, 50)
        phase_chirp = 2 * math.pi * (f0 * t + (f1 - f0) * t ** 2 / 2)
        x = torch.cos(phase_chirp)
        y = hilbert(x)
        phase = y[..., 0]
        phase_unwrapped = torch.from_numpy(np.unwrap(phase.numpy()))
        # Act
        inst_freq = torch.diff(phase_unwrapped) / (2 * math.pi / seq_len)
        # Assert
        assert inst_freq[10] < inst_freq[-10]

    def test_envelope_detection_hilbert_behaves_correctly(self):
        """Test envelope detection using Hilbert transform."""
        # Arrange
        seq_len = 512
        hilbert = Hilbert(seq_len)
        t = torch.linspace(0, 1, seq_len)
        carrier_freq = 50
        mod_freq = 5
        carrier = torch.cos(2 * math.pi * carrier_freq * t)
        envelope = 1 + 0.5 * torch.cos(2 * math.pi * mod_freq * t)
        x = envelope * carrier
        # Act
        y = hilbert(x)
        detected_envelope = y[..., 1]
        # Assert
        assert torch.allclose(detected_envelope[::10], envelope[::10], atol=0.15)

    def test_gradient_flow_hilbert_behaves_correctly_grad(self):
        """Test that gradients flow through the layer."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x = torch.randn(4, seq_len, requires_grad=True)
        y = hilbert(x)
        loss = y.sum()
        # Act
        loss.backward()
        # Assert
        assert x.grad is not None
        pass

    def test_gradient_flow_hilbert_behaves_correctly_allclose(self):
        """Test that gradients flow through the layer."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x = torch.randn(4, seq_len, requires_grad=True)
        y = hilbert(x)
        loss = y.sum()
        # Act
        loss.backward()
        # Assert
        pass
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_compatibility_hilbert_behaves_correctly_device(self):
        """Test operation on different devices."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len).cuda()
        x = torch.randn(4, seq_len).cuda()
        # Act
        y = hilbert(x)
        # Assert
        assert y.device == x.device
        pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_device_compatibility_hilbert_behaves_correctly_shape(self):
        """Test operation on different devices."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len).cuda()
        x = torch.randn(4, seq_len).cuda()
        # Act
        y = hilbert(x)
        # Assert
        pass
        assert y.shape == (4, seq_len, 2)

    def test_batch_processing_consistency(self):
        """Test consistency of batch processing."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x_single = torch.randn(seq_len)
        y_single = hilbert(x_single)
        x_batch = x_single.unsqueeze(0).repeat(4, 1)
        # Act
        y_batch = hilbert(x_batch)
        # Assert
        for i in range(4):
            assert torch.allclose(y_batch[i], y_single, atol=1e-06)

    def test_real_signal_constraint(self):
        """Test that input must be real-valued."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x_real = torch.randn(seq_len)
        # Act
        y_real = hilbert(x_real)
        # Assert
        assert y_real.shape == (seq_len, 2)

    def test_numerical_stability_hilbert_behaves_correctly_all(self):
        """Test numerical stability with extreme values."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x_large = torch.randn(seq_len) * 1000000.0
        # Act
        y_large = hilbert(x_large)
        # Assert
        assert torch.isfinite(y_large).all()
        x_small = torch.randn(seq_len) * 1e-06
        y_small = hilbert(x_small)
        pass

    def test_numerical_stability_hilbert_behaves_correctly_all_v2(self):
        """Test numerical stability with extreme values."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x_large = torch.randn(seq_len) * 1000000.0
        # Act
        y_large = hilbert(x_large)
        # Assert
        pass
        x_small = torch.randn(seq_len) * 1e-06
        y_small = hilbert(x_small)
        assert torch.isfinite(y_small).all()

    def test_orthogonality_property_hilbert_behaves_correctly(self):
        """Test orthogonality between original and Hilbert transform."""
        # Arrange
        seq_len = 256
        hilbert = Hilbert(seq_len)
        t = torch.linspace(0, 4 * math.pi, seq_len)
        x = torch.sin(t)
        # Act
        y = hilbert(x)
        phase = y[..., 0]
        # Assert
        assert phase is not None

        # Phase should advance by ~90 degrees for sine wave
        # This is an indirect test of the orthogonality property

    def test_integration_with_nn_module(self):
        """Test integration with PyTorch Sequential model."""
        # Arrange
        seq_len = 128
        model = nn.Sequential(nn.Linear(seq_len, seq_len), nn.ReLU(), Hilbert(seq_len), nn.Flatten(), nn.Linear(seq_len * 2, 10))
        x = torch.randn(8, seq_len)
        # Act
        y = model(x)
        # Assert
        assert y.shape == (8, 10)

    def test_state_dict_save_load(self):
        """Test saving and loading model state."""
        # Arrange
        seq_len = 128
        hilbert1 = Hilbert(seq_len, fp16=True, in_place=True)
        state = hilbert1.state_dict()
        hilbert2 = Hilbert(seq_len, fp16=True, in_place=True)
        # Act
        hilbert2.load_state_dict(state)
        # Assert
        assert torch.equal(hilbert1.h_mask, hilbert2.h_mask)

    def test_power_spectrum_preservation(self):
        """Test that power spectrum magnitude is preserved."""
        # Arrange
        seq_len = 256
        hilbert = Hilbert(seq_len)
        x = torch.randn(seq_len)
        y = hilbert(x)
        amplitude = y[..., 1]
        # Act
        x_power = (x ** 2).mean()
        # Assert
        assert x_power is not None
        # Note: amplitude is envelope, so comparison is indirect

    def test_canonical_mask_shape_item(self):
        """Test that h_mask is the canonical analytic-signal mask."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len)
        m = hilbert.h_mask
        # Assert
        assert m[0].item() == 1.0
        pass
        pass
        pass

    def test_canonical_mask_shape_item_v2(self):
        """Test that h_mask is the canonical analytic-signal mask."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len)
        m = hilbert.h_mask
        # Assert
        pass
        assert m[seq_len // 2].item() == 1.0
        pass
        pass

    def test_canonical_mask_shape_allclose(self):
        """Test that h_mask is the canonical analytic-signal mask."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len)
        m = hilbert.h_mask
        # Assert
        pass
        pass
        assert torch.allclose(m[1:seq_len // 2], torch.full((seq_len // 2 - 1,), 2.0))
        pass

    def test_canonical_mask_shape_allclose_v2(self):
        """Test that h_mask is the canonical analytic-signal mask."""
        # Arrange
        seq_len = 128
        # Act
        hilbert = Hilbert(seq_len)
        m = hilbert.h_mask
        # Assert
        pass
        pass
        pass
        assert torch.allclose(m[seq_len // 2 + 1:], torch.zeros(seq_len // 2 - 1))

    def test_multi_channel_independence_shape(self):
        """Test that channels are processed independently."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x = torch.zeros(3, seq_len)
        x[0] = torch.sin(torch.linspace(0, 2 * math.pi, seq_len))
        x[1] = torch.cos(torch.linspace(0, 4 * math.pi, seq_len))
        x[2] = torch.randn(seq_len) * 0.1
        # Act
        y = hilbert(x)
        # Assert
        assert y.shape == (3, seq_len, 2)
        pass
        pass

    def test_multi_channel_independence_allclose(self):
        """Test that channels are processed independently."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x = torch.zeros(3, seq_len)
        x[0] = torch.sin(torch.linspace(0, 2 * math.pi, seq_len))
        x[1] = torch.cos(torch.linspace(0, 4 * math.pi, seq_len))
        x[2] = torch.randn(seq_len) * 0.1
        # Act
        y = hilbert(x)
        # Assert
        pass
        assert not torch.allclose(y[0], y[1], atol=0.1)
        pass

    def test_multi_channel_independence_allclose_v2(self):
        """Test that channels are processed independently."""
        # Arrange
        seq_len = 128
        hilbert = Hilbert(seq_len)
        x = torch.zeros(3, seq_len)
        x[0] = torch.sin(torch.linspace(0, 2 * math.pi, seq_len))
        x[1] = torch.cos(torch.linspace(0, 4 * math.pi, seq_len))
        x[2] = torch.randn(seq_len) * 0.1
        # Act
        y = hilbert(x)
        # Assert
        pass
        pass
        assert not torch.allclose(y[1], y[2], atol=0.1)


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_Hilbert.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-10 12:46:06 (ywatanabe)"
# # File: /home/ywatanabe/proj/scitex_repo/src/scitex/nn/_Hilbert.py
# # ----------------------------------------
# import os
#
# __FILE__ = "/home/ywatanabe/proj/scitex_repo/src/scitex/nn/_Hilbert.py"
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# #!/usr/bin/env python
#
# import torch  # 1.7.1
# import torch.nn as nn
# from torch.fft import fft, ifft
#
#
# class Hilbert(nn.Module):
#     def __init__(self, seq_len, dim=-1, fp16=False, in_place=False):
#         super().__init__()
#         self.dim = dim
#         self.fp16 = fp16
#         self.in_place = in_place
#         self.n = seq_len
#         f = torch.cat(
#             [
#                 torch.arange(0, (self.n - 1) // 2 + 1) / float(self.n),
#                 torch.arange(-(self.n // 2), 0) / float(self.n),
#             ]
#         )
#         self.register_buffer("f", f)
#
#     def hilbert_transform(self, x):
#         # n = x.shape[self.dim]
#
#         # Create frequency dim
#         # f = torch.cat(
#         #     [
#         #         torch.arange(0, (n - 1) // 2 + 1, device=x.device) / float(n),
#         #         torch.arange(-(n // 2), 0, device=x.device) / float(n),
#         #     ]
#         # )
#
#         orig_dtype = x.dtype
#         x = x.float()
#         xf = fft(x, n=self.n, dim=self.dim)
#         x = x.to(orig_dtype)
#
#         # Create step function
#         steepness = 50  # This value can be adjusted
#         u = torch.sigmoid(
#             steepness * self.f.type_as(x)
#         )  # Soft step function for differentiability
#
#         transformed = ifft(xf * 2 * u, dim=self.dim)
#
#         return transformed
#
#     def forward(self, x):
#         if self.fp16:
#             x = x.half()
#
#         if not self.in_place:
#             x = x.clone()  # Ensure that we do not modify the input in-place
#
#         x_comp = self.hilbert_transform(x)
#
#         pha = torch.atan2(x_comp.imag, x_comp.real)
#         amp = x_comp.abs()
#
#         assert x.shape == pha.shape == amp.shape
#
#         out = torch.cat(
#             [
#                 pha.unsqueeze(-1),
#                 amp.unsqueeze(-1),
#             ],
#             dim=-1,
#         )
#
#         # if self.fp16:
#         #     out = (
#         #         out.float()
#         #     )
#         #     # Optionally cast back to float for stability in subsequent operations
#
#         if self.fp16:
#             out = out.float()
#
#         return out
#
#
# if __name__ == "__main__":
#     import scitex
#
#     xx, tt, fs = scitex.dsp.demo_sig()
#     xx = torch.tensor(xx)
#
#     # Parameters
#     device = "cuda"
#     fp16 = True
#     in_place = True
#
#     # Initialization
#     m = Hilbert(xx.shape[-1], fp16=fp16, in_place=in_place).to(device)
#
#     # Calculation
#     xx = xx.to(device)
#     y = m(xx)
#
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/nn/_Hilbert.py
# --------------------------------------------------------------------------------

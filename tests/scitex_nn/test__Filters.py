#!/usr/bin/env python3
# Time-stamp: "2025-01-01 00:00:00 (ywatanabe)"
# File: test__Filters.py

"""Test suite for the neural-network filter layers.

These tests exercise the real filter implementations (no mocks). The
underlying scipy/firwin filter-design helpers (`design_filter`,
`init_bandpass_filters`, `build_bandpass_filters`) run for real, so kernel
shapes and forward outputs are the genuine ones.
"""

import pytest

# Required for this module
pytest.importorskip("torch")
import numpy as np
import torch


class TestBaseFilter1D:
    """The abstract BaseFilter1D base class."""

    def test_base_filter_stores_fp16_flag(self):
        """Test BaseFilter1D records the fp16 flag."""
        # Arrange
        from scitex_nn import BaseFilter1D

        class ConcreteFilter(BaseFilter1D):
            def init_kernels(self):
                return torch.randn(5, 64)

        # Act
        filter_layer = ConcreteFilter(fp16=False, in_place=False)
        # Assert
        assert filter_layer.fp16 is False

    def test_base_filter_stores_in_place_flag(self):
        """Test BaseFilter1D records the in_place flag."""
        # Arrange
        from scitex_nn import BaseFilter1D

        class ConcreteFilter(BaseFilter1D):
            def init_kernels(self):
                return torch.randn(5, 64)

        # Act
        filter_layer = ConcreteFilter(fp16=False, in_place=False)
        # Assert
        assert filter_layer.in_place is False

    def test_forward_without_kernels_raises_valueerror(self):
        """Test forward fails when kernels are not initialized."""
        # Arrange
        from scitex_nn import BaseFilter1D

        class ConcreteFilter(BaseFilter1D):
            def init_kernels(self):
                pass

        filter_layer = ConcreteFilter()
        filter_layer.kernels = None
        x = torch.randn(2, 4, 100)
        # Act
        ctx = pytest.raises(ValueError, match="Filter kernels has not been initialized")
        # Assert
        with ctx:
            filter_layer(x)

    def test_flip_extend_mirrors_signal_edges(self):
        """Test flip_extend mirrors the signal at both edges."""
        # Arrange
        from scitex_nn import BaseFilter1D

        x = torch.tensor([[[1, 2, 3, 4, 5]]])
        expected = torch.tensor([[[2, 1, 1, 2, 3, 4, 5, 5, 4]]])
        # Act
        extended = BaseFilter1D.flip_extend(x, 2)
        # Assert
        assert torch.allclose(extended, expected)

    def test_batch_conv_adds_kernel_dimension(self):
        """Test batch_conv produces a (batch, chs, n_kernels, seq) output."""
        # Arrange
        from scitex_nn import BaseFilter1D

        x = torch.randn(2, 3, 10)
        kernels = torch.randn(4, 3)
        # Act
        output = BaseFilter1D.batch_conv(x, kernels, padding=1)
        # Assert
        assert output.shape == (2, 3, 4, 10)

    def test_remove_edges_with_explicit_length(self):
        """Test remove_edges with an explicit edge length trims both ends."""
        # Arrange
        from scitex_nn import BaseFilter1D

        x = torch.randn(2, 3, 100)
        # Act
        trimmed = BaseFilter1D.remove_edges(x, edge_len=10)
        # Assert
        assert trimmed.shape[-1] == 80

    def test_remove_edges_with_auto_length(self):
        """Test remove_edges('auto') trims seq_len // 8 from each end."""
        # Arrange
        from scitex_nn import BaseFilter1D

        x = torch.randn(2, 3, 100)
        # Act
        trimmed_auto = BaseFilter1D.remove_edges(x, edge_len="auto")
        # Assert
        assert trimmed_auto.shape[-1] == 100 - 2 * (100 // 8)

    def test_remove_edges_with_zero_length_is_identity(self):
        """Test remove_edges(0) leaves the signal shape unchanged."""
        # Arrange
        from scitex_nn import BaseFilter1D

        x = torch.randn(2, 3, 100)
        # Act
        no_trim = BaseFilter1D.remove_edges(x, edge_len=0)
        # Assert
        assert no_trim.shape == x.shape


class TestBandPassFilter:
    """The BandPassFilter (real firwin kernels)."""

    def test_bandpass_has_kernels_attribute(self):
        """Test BandPassFilter exposes a kernels buffer."""
        # Arrange
        from scitex_nn import BandPassFilter

        # Act
        bp_filter = BandPassFilter(
            torch.tensor([[10, 20], [20, 40], [40, 80]]), 256, 1024
        )
        # Assert
        assert hasattr(bp_filter, "kernels")

    def test_bandpass_builds_one_kernel_per_band(self):
        """Test BandPassFilter builds one kernel row per requested band."""
        # Arrange
        from scitex_nn import BandPassFilter

        # Act
        bp_filter = BandPassFilter(
            torch.tensor([[10, 20], [20, 40], [40, 80]]), 256, 1024
        )
        # Assert
        assert bp_filter.kernels.shape[0] == 3

    def test_bandpass_accepts_numpy_bands(self):
        """Test BandPassFilter accepts numpy-array bands."""
        # Arrange
        from scitex_nn import BandPassFilter

        # Act
        bp_filter = BandPassFilter(np.array([[10, 20], [20, 40]]), 256, 512)
        # Assert
        assert bp_filter.kernels is not None

    def test_bandpass_builds_with_bands_above_nyquist(self):
        """Test BandPassFilter constructs valid kernels when bands exceed Nyquist."""
        # Arrange
        from scitex_nn import BandPassFilter

        # Act — high_hz of 60/70 exceed the Nyquist of 50; source clips internally
        bp_filter = BandPassFilter(torch.tensor([[10, 60], [30, 70]]), 100, 256)
        # Assert
        assert bp_filter.kernels.shape[0] == 2

    def test_bandpass_forward_adds_band_dimension(self):
        """Test BandPassFilter forward yields (batch, chs, n_bands, seq)."""
        # Arrange
        from scitex_nn import BandPassFilter

        bp_filter = BandPassFilter(torch.tensor([[5, 15], [15, 30]]), 128, 512)
        x = torch.randn(2, 3, 512)
        # Act
        output = bp_filter(x)
        # Assert
        assert output.shape == (2, 3, 2, 512)

    def test_bandpass_fp16_sets_flag(self):
        """Test BandPassFilter with fp16=True records the flag."""
        # Arrange
        from scitex_nn import BandPassFilter

        # Act
        bp_filter = BandPassFilter(torch.tensor([[10, 20]]), 256, 256, fp16=True)
        # Assert
        assert bp_filter.fp16 is True

    def test_bandpass_fp16_produces_half_precision_kernels(self):
        """Test BandPassFilter with fp16=True yields float16 kernels."""
        # Arrange
        from scitex_nn import BandPassFilter

        # Act
        bp_filter = BandPassFilter(torch.tensor([[10, 20]]), 256, 256, fp16=True)
        # Assert
        assert bp_filter.kernels.dtype == torch.float16


class TestBandStopFilter:
    """The BandStopFilter (real firwin kernels)."""

    def test_bandstop_has_kernels_attribute(self):
        """Test BandStopFilter exposes a kernels buffer."""
        # Arrange
        from scitex_nn import BandStopFilter

        # Act
        bs_filter = BandStopFilter(np.array([[45, 55], [95, 105]]), 500, 1000)
        # Assert
        assert hasattr(bs_filter, "kernels")

    def test_bandstop_builds_one_kernel_per_band(self):
        """Test BandStopFilter builds one kernel row per requested band."""
        # Arrange
        from scitex_nn import BandStopFilter

        # Act
        bs_filter = BandStopFilter(np.array([[45, 55], [95, 105]]), 500, 1000)
        # Assert
        assert bs_filter.kernels.shape[0] == 2


class TestLowPassFilter:
    """The LowPassFilter (real firwin kernels)."""

    def test_lowpass_has_kernels_attribute(self):
        """Test LowPassFilter exposes a kernels buffer."""
        # Arrange
        from scitex_nn import LowPassFilter

        # Act
        lp_filter = LowPassFilter(np.array([10, 20, 30]), 100, 256)
        # Assert
        assert hasattr(lp_filter, "kernels")

    def test_lowpass_builds_one_kernel_per_cutoff(self):
        """Test LowPassFilter builds one kernel row per cutoff."""
        # Arrange
        from scitex_nn import LowPassFilter

        # Act
        lp_filter = LowPassFilter(np.array([10, 20, 30]), 100, 256)
        # Assert
        assert lp_filter.kernels.shape[0] == 3

    def test_lowpass_rejects_cutoff_above_nyquist(self):
        """Test LowPassFilter asserts on a cutoff above the Nyquist limit."""
        # Arrange
        from scitex_nn import LowPassFilter

        # Act
        ctx = pytest.raises(AssertionError)
        # Assert
        with ctx:
            LowPassFilter(np.array([60]), 100, 256)

    def test_lowpass_forward_adds_band_dimension(self):
        """Test LowPassFilter forward yields (batch, chs, n_cutoffs, seq)."""
        # Arrange
        from scitex_nn import LowPassFilter

        lp_filter = LowPassFilter(np.array([15, 25]), 100, 200)
        x = torch.randn(3, 4, 200)
        # Act
        output = lp_filter(x)
        # Assert
        assert output.shape == (3, 4, 2, 200)


class TestHighPassFilter:
    """The HighPassFilter (real firwin kernels)."""

    def test_highpass_has_kernels_attribute(self):
        """Test HighPassFilter exposes a kernels buffer."""
        # Arrange
        from scitex_nn import HighPassFilter

        # Act
        hp_filter = HighPassFilter(np.array([1, 5, 10]), 100, 256)
        # Assert
        assert hasattr(hp_filter, "kernels")

    def test_highpass_builds_one_kernel_per_cutoff(self):
        """Test HighPassFilter builds one kernel row per cutoff."""
        # Arrange
        from scitex_nn import HighPassFilter

        # Act
        hp_filter = HighPassFilter(np.array([1, 5, 10]), 100, 256)
        # Assert
        assert hp_filter.kernels.shape[0] == 3

    def test_highpass_forward_adds_band_dimension(self):
        """Test HighPassFilter forward yields (batch, chs, n_cutoffs, seq)."""
        # Arrange
        from scitex_nn import HighPassFilter

        hp_filter = HighPassFilter(np.array([0.5, 1.0]), 50, 400)
        x = torch.randn(2, 3, 400)
        # Act
        output = hp_filter(x)
        # Assert
        assert output.shape == (2, 3, 2, 400)


class TestGaussianFilter:
    """The GaussianFilter (real to_even kernel sizing)."""

    def test_gaussian_has_kernels_attribute(self):
        """Test GaussianFilter exposes a kernels buffer."""
        # Arrange
        from scitex_nn import GaussianFilter

        # Act
        gauss_filter = GaussianFilter(4)
        # Assert
        assert hasattr(gauss_filter, "kernels")

    def test_gaussian_stores_sigma(self):
        """Test GaussianFilter records the sigma value."""
        # Arrange
        from scitex_nn import GaussianFilter

        # Act
        gauss_filter = GaussianFilter(4)
        # Assert
        assert gauss_filter.sigma == 4

    def test_gaussian_kernel_has_single_row(self):
        """Test the Gaussian kernel tensor has a single row."""
        # Arrange
        from scitex_nn import GaussianFilter

        # Act
        kernels = GaussianFilter.init_kernels(4)
        # Assert
        assert kernels.shape[0] == 1

    def test_gaussian_kernel_spans_at_least_six_sigma(self):
        """Test the Gaussian kernel spans at least 6*sigma samples."""
        # Arrange
        from scitex_nn import GaussianFilter

        sigma = 4
        # Act
        kernels = GaussianFilter.init_kernels(sigma)
        # Assert
        assert kernels.shape[1] >= sigma * 6

    def test_gaussian_kernel_sums_to_one(self):
        """Test the Gaussian kernel is normalized to sum to one."""
        # Arrange
        from scitex_nn import GaussianFilter

        # Act
        kernels = GaussianFilter.init_kernels(4)
        # Assert
        assert torch.allclose(kernels.sum(), torch.tensor(1.0), atol=1e-06)

    def test_gaussian_forward_adds_singleton_band_dimension(self):
        """Test GaussianFilter forward yields (batch, chs, 1, seq)."""
        # Arrange
        from scitex_nn import GaussianFilter

        gauss_filter = GaussianFilter(3)
        x = torch.randn(2, 3, 100)
        # Act
        output = gauss_filter(x)
        # Assert
        assert output.shape == (2, 3, 1, 100)

    def test_gaussian_reduces_signal_variance(self):
        """Test the Gaussian filter smooths (reduces variance of) a noisy signal."""
        # Arrange
        from scitex_nn import GaussianFilter

        t = torch.linspace(0, 1, 200)
        noisy_signal = torch.sin(2 * np.pi * 5 * t) + torch.randn_like(t) * 0.5
        noisy_signal = noisy_signal.unsqueeze(0).unsqueeze(0)
        gauss_filter = GaussianFilter(8)
        # Act
        smoothed = gauss_filter(noisy_signal)
        # Assert
        assert smoothed.var() < noisy_signal.var()


class TestDifferentiableBandPassFilter:
    """The DifferentiableBandPassFilter (real learnable bandpass)."""

    def test_differentiable_stores_sig_len(self):
        """Test DifferentiableBandPassFilter records sig_len."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        # Act
        dbp_filter = DifferentiableBandPassFilter(512, 256)
        # Assert
        assert dbp_filter.sig_len == 512

    def test_differentiable_stores_fs(self):
        """Test DifferentiableBandPassFilter records fs."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        # Act
        dbp_filter = DifferentiableBandPassFilter(512, 256)
        # Assert
        assert dbp_filter.fs == 256

    def test_differentiable_has_phase_mids(self):
        """Test DifferentiableBandPassFilter exposes learnable pha_mids."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        # Act
        dbp_filter = DifferentiableBandPassFilter(512, 256)
        # Assert
        assert hasattr(dbp_filter, "pha_mids")

    def test_differentiable_has_amplitude_mids(self):
        """Test DifferentiableBandPassFilter exposes learnable amp_mids."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        # Act
        dbp_filter = DifferentiableBandPassFilter(512, 256)
        # Assert
        assert hasattr(dbp_filter, "amp_mids")

    def test_differentiable_stores_phase_high_hz(self):
        """Test DifferentiableBandPassFilter records the requested pha_high_hz."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        # Act
        dbp_filter = DifferentiableBandPassFilter(
            1024, 200, pha_high_hz=150, amp_high_hz=200
        )
        # Assert
        assert dbp_filter.pha_high_hz == 150

    def test_differentiable_stores_amplitude_high_hz(self):
        """Test DifferentiableBandPassFilter records the requested amp_high_hz."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        # Act
        dbp_filter = DifferentiableBandPassFilter(
            1024, 200, pha_high_hz=150, amp_high_hz=200
        )
        # Assert
        assert dbp_filter.amp_high_hz == 200

    def test_differentiable_forward_adds_total_band_dimension(self):
        """Test forward yields (batch, chs, n_pha*n_amp, seq)."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        dbp_filter = DifferentiableBandPassFilter(
            256, 128, pha_n_bands=10, amp_n_bands=10
        )
        x = torch.randn(2, 3, 256)
        # Act
        output = dbp_filter(x)
        # Assert
        assert output.shape == (2, 3, 20, 256)

    def test_differentiable_gradient_flows_to_input(self):
        """Test gradients flow back to the input through the learnable filter."""
        # Arrange
        from scitex_nn import DifferentiableBandPassFilter

        dbp_filter = DifferentiableBandPassFilter(128, 64, pha_n_bands=5, amp_n_bands=5)
        x = torch.randn(1, 1, 128, requires_grad=True)
        # Act
        dbp_filter(x).sum().backward()
        # Assert
        assert x.grad is not None


class TestEdgeCases:
    """Edge cases and the optional time-parameter path."""

    def test_short_sequence_output_does_not_exceed_input_length(self):
        """Test a very short sequence yields an output no longer than the input."""
        # Arrange
        from scitex_nn import BandPassFilter

        bp_filter = BandPassFilter(torch.tensor([[5, 10]]), 50, 32)
        x = torch.randn(1, 1, 32)
        # Act
        output = bp_filter(x)
        # Assert
        assert output.shape[-1] <= 32

    def test_time_parameter_output_matches_signal_length(self):
        """Test the returned time vector length matches the trimmed signal length."""
        # Arrange
        from scitex_nn import HighPassFilter

        hp_filter = HighPassFilter(np.array([1.0]), 20, 200)
        x = torch.randn(2, 3, 200)
        t = torch.linspace(0, 10, 200)
        # Act
        x_out, t_out = hp_filter(x, t=t, edge_len=10)
        # Assert
        assert x_out.shape[-1] == t_out.shape[-1]

    def test_time_parameter_trims_edges_from_time_vector(self):
        """Test the returned time vector is trimmed by 2*edge_len."""
        # Arrange
        from scitex_nn import HighPassFilter

        hp_filter = HighPassFilter(np.array([1.0]), 20, 200)
        x = torch.randn(2, 3, 200)
        t = torch.linspace(0, 10, 200)
        # Act
        _, t_out = hp_filter(x, t=t, edge_len=10)
        # Assert
        assert t_out.shape[-1] == 200 - 20


class TestMultiFilterProcessing:
    """Processing across multiple bands and cascaded filters."""

    def test_multi_band_forward_produces_one_output_per_band(self):
        """Test a five-band bandpass yields five output bands."""
        # Arrange
        from scitex_nn import BandPassFilter

        bands = torch.tensor([[1, 4], [4, 8], [8, 13], [13, 30], [30, 100]])
        bp_filter = BandPassFilter(bands, 256, 1024)
        x = torch.randn(4, 8, 1024)
        # Act
        output = bp_filter(x)
        # Assert
        assert output.shape == (4, 8, 5, 1024)

    def test_cascaded_filters_preserve_batch_and_channel_dims(self):
        """Test high-pass then low-pass keeps the leading (batch, chs) dims."""
        # Arrange
        from scitex_nn import HighPassFilter, LowPassFilter

        hp_filter = HighPassFilter(np.array([5]), 100, 500)
        lp_filter = LowPassFilter(np.array([20]), 100, 500)
        x = torch.randn(2, 3, 500)
        # Act
        x_hp = hp_filter(x)
        x_bp = lp_filter(x_hp[:, :, 0, :])
        # Assert
        assert x_bp.shape[0:2] == x.shape[0:2]


class TestDeviceCompatibility:
    """Filter operations across devices."""

    def test_cpu_filtering_stays_on_cpu(self):
        """Test a GaussianFilter on CPU input produces a CPU output."""
        # Arrange
        from scitex_nn import GaussianFilter

        gauss_filter = GaussianFilter(4)
        x = torch.randn(2, 3, 100)
        # Act
        output = gauss_filter(x)
        # Assert
        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_filtering_stays_on_cuda(self):
        """Test a BandPassFilter on CUDA input produces a CUDA output."""
        # Arrange
        from scitex_nn import BandPassFilter

        bp_filter = BandPassFilter(torch.tensor([[10, 20]]), 100, 256).cuda()
        x = torch.randn(2, 3, 256).cuda()
        # Act
        output = bp_filter(x)
        # Assert
        assert output.is_cuda


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__)])

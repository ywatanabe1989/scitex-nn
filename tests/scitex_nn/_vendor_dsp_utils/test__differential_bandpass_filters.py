"""Tests for vendored differentiable bandpass filters."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch
import torch.nn as nn

from scitex_nn._vendor_dsp_utils._differential_bandpass_filters import (
    build_bandpass_filters,
)


class TestBuildBandpassFilters:
    """Tests for the build_bandpass_filters function."""

    def test_build_bandpass_filters_returns_tensor(self):
        """build_bandpass_filters returns a torch Tensor."""
        # Arrange
        sig_len, fs = 512, 1024
        pha_mids = nn.Parameter(torch.linspace(4, 12, 5))
        amp_mids = nn.Parameter(torch.linspace(60, 120, 3))
        # Act
        filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle=3)
        # Assert
        assert isinstance(filters, torch.Tensor)

    def test_build_bandpass_filters_expected_number_of_filters(self):
        """build_bandpass_filters returns pha_n_bands + amp_n_bands filters."""
        # Arrange
        sig_len, fs = 512, 1024
        pha_n, amp_n = 5, 3
        pha_mids = nn.Parameter(torch.linspace(4, 12, pha_n))
        amp_mids = nn.Parameter(torch.linspace(60, 120, amp_n))
        # Act
        filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle=3)
        # Assert
        assert filters.shape[0] == pha_n + amp_n

    def test_build_bandpass_filters_supports_backward(self):
        """build_bandpass_filters output supports gradient computation."""
        # Arrange
        sig_len, fs = 512, 1024
        pha_mids = nn.Parameter(torch.linspace(4, 12, 5))
        amp_mids = nn.Parameter(torch.linspace(60, 120, 3))
        # Act
        filters = build_bandpass_filters(sig_len, fs, pha_mids, amp_mids, cycle=3)
        loss = filters.sum()
        loss.backward()
        # Assert
        assert pha_mids.grad is not None

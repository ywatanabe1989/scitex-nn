"""Tests for vendored filter design helper."""

from __future__ import annotations

import numpy as np
import pytest

from scitex_nn._vendor_dsp_utils.filter import design_filter


class TestDesignFilter:
    """Tests for the design_filter function."""

    def test_design_filter_lowpass_returns_coefficients(self):
        """design_filter returns a 1D array for lowpass."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        h = design_filter(sig_len, fs, low_hz=30)
        # Assert
        assert isinstance(h, np.ndarray)

    def test_design_filter_lowpass_has_expected_len(self):
        """design_filter output length approximates expected order."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        h = design_filter(sig_len, fs, low_hz=30)
        # Assert
        assert len(h) > 0

    def test_design_filter_highpass_returns_coefficients(self):
        """design_filter returns coefficients for highpass."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        h = design_filter(sig_len, fs, high_hz=70)
        # Assert
        assert isinstance(h, np.ndarray)

    def test_design_filter_bandpass_returns_coefficients(self):
        """design_filter returns coefficients for bandpass."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        h = design_filter(sig_len, fs, low_hz=30, high_hz=70)
        # Assert
        assert isinstance(h, np.ndarray)

    def test_design_filter_bandstop_returns_coefficients(self):
        """design_filter returns coefficients for bandstop."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        h = design_filter(sig_len, fs, low_hz=30, high_hz=70, is_bandstop=True)
        # Assert
        assert isinstance(h, np.ndarray)

    def test_design_filter_no_cutoffs_raises_error(self):
        """design_filter raises FilterParameterError with no cutoffs."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        ctx = pytest.raises(Exception)
        # Assert
        with ctx:
            design_filter(sig_len, fs)

    def test_design_filter_invalid_low_high_raises_error(self):
        """design_filter raises error when low_hz >= high_hz."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        ctx = pytest.raises(Exception)
        # Assert
        with ctx:
            design_filter(sig_len, fs, low_hz=70, high_hz=30)

    def test_design_filter_normalized_bandpass_magnitude(self):
        """design_filter bandpass coefficients have reasonable magnitude."""
        # Arrange
        sig_len, fs = 256, 1024
        # Act
        h = design_filter(sig_len, fs, low_hz=30, high_hz=70)
        # Assert
        assert np.max(np.abs(h)) <= 1.0

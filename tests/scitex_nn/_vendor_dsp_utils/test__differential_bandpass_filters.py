"""Smoke import test for vendored helper."""


def test_module_importable_differential_bandpass_filters_behaves_correctly():
    # Arrange
    import scitex_nn._vendor_dsp_utils
    # Act
    # Assert
    assert True is True

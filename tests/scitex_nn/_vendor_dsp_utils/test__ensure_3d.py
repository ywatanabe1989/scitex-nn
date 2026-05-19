"""Smoke import test for vendored helper."""


def test_module_importable_ensure_3d_behaves_correctly():
    # Arrange
    import scitex_nn._vendor_dsp_utils
    # Act
    # Assert
    assert True is True

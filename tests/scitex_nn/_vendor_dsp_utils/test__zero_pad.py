"""Smoke import test for vendored helper."""


def test_module_importable():
    import scitex_nn._vendor_dsp_utils  # noqa: F401

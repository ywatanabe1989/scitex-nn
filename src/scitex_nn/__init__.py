#!/usr/bin/env python3
"""Scitex nn module.

Top-level imports are PEP 562 lazy — ``import scitex_nn`` is cheap and does
NOT pull in ``torch``. Public symbols (layers, filters, transforms) load on
first attribute access, at which point ``torch`` is imported. This mirrors
``scitex_io``'s lazy-loading pattern (``_LAZY_ATTRS`` + ``__getattr__`` +
``__dir__``).
"""

from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _v

    try:
        __version__ = _v("scitex-nn")
    except PackageNotFoundError:
        __version__ = "0.0.0+local"
    del _v, PackageNotFoundError
except ImportError:  # pragma: no cover — only on ancient Pythons
    __version__ = "0.0.0+local"


# ---------------------------------------------------------------------------
# PEP 562 lazy attribute map.
#
# Most public-name → submodule pairs (LHS == public symbol, RHS == submodule
# that defines it under the same name) live in ``_LAZY_ATTRS``.
#
# Names whose public alias differs from the in-module symbol (the BNet /
# BNet_Res pair) use ``_ALIASED_ATTRS`` with the explicit
# ``(relative_module, attr_name_in_module)`` form.
# ---------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, str] = {
    # Augmentation layers
    "AxiswiseDropout": "._aug",
    "ChannelGainChanger": "._aug",
    "DropoutChannels": "._aug",
    "FreqGainChanger": "._aug",
    "SwapChannels": "._aug",
    # Filters
    "BandPassFilter": "._Filters",
    "BandStopFilter": "._Filters",
    "BaseFilter1D": "._Filters",
    "DifferentiableBandPassFilter": "._Filters",
    "GaussianFilter": "._Filters",
    "HighPassFilter": "._Filters",
    "LowPassFilter": "._Filters",
    # Transforms / signal layers
    "DimHandler": "._DimHandler",
    "Hilbert": "._Hilbert",
    "ModulationIndex": "._ModulationIndex",
    "PAC": "._PAC",
    "PSD": "._PSD",
    "SpatialAttention": "._SpatialAttention",
    "TransposeLayer": "._TransposeLayer",
    "Wavelet": "._Wavelet",
    # MNet
    "MNet1000": "._MNet_1000",
    "MNet_1000": "._MNet_1000",
    "MNet_config": "._MNet_1000",
    "ReshapeLayer": "._MNet_1000",
    "SwapLayer": "._MNet_1000",
    # ResNet
    "ResNet1D": "._ResNet1D",
    "ResNetBasicBlock": "._ResNet1D",
    # Spectrogram
    "Spectrogram": "._Spectrogram",
    "my_softmax": "._Spectrogram",
    "normalize": "._Spectrogram",
    "spectrograms": "._Spectrogram",
    "unbias": "._Spectrogram",
}

# Public names whose alias differs from the in-module symbol name.
# name: (relative_module, attr_name_in_module)
_ALIASED_ATTRS: dict[str, tuple[str, str]] = {
    "BHead_v1": ("._BNet", "BHead"),
    "BNet_v1": ("._BNet", "BNet"),
    "BNet_config_v1": ("._BNet", "BNet_config"),
    "BHead_Res": ("._BNet_Res", "BHead"),
    "BNet_Res": ("._BNet_Res", "BNet"),
    "BNet_config_Res": ("._BNet_Res", "BNet_config"),
}


def _load_lazy_attr(name: str):
    """Resolve a ``_LAZY_ATTRS`` name and cache it."""
    from importlib import import_module

    mod = import_module(_LAZY_ATTRS[name], __name__)
    attr = getattr(mod, name)
    globals()[name] = attr
    return attr


def _load_aliased_attr(name: str):
    """Resolve an ``_ALIASED_ATTRS`` name and cache it."""
    from importlib import import_module

    mod_name, attr_name = _ALIASED_ATTRS[name]
    mod = import_module(mod_name, __name__)
    attr = getattr(mod, attr_name)
    globals()[name] = attr
    return attr


def __getattr__(name: str):
    """PEP 562 lazy-loader: import on first access, cache, return."""
    # Reference the dispatch tables directly so static analysers (e.g.
    # scitex-dev's PA-102 audit) recognise the PEP 562 lazy keys as bound.
    if _LAZY_ATTRS.get(name) is not None:
        return _load_lazy_attr(name)
    if _ALIASED_ATTRS.get(name) is not None:
        return _load_aliased_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(_LAZY_ATTRS) | set(_ALIASED_ATTRS) | set(globals()))


__all__ = [
    "__version__",
    "AxiswiseDropout",
    "BHead_v1",
    "BHead_Res",
    "BNet_v1",
    "BNet_Res",
    "BNet_config_v1",
    "BNet_config_Res",
    "BandPassFilter",
    "BandStopFilter",
    "BaseFilter1D",
    "ChannelGainChanger",
    "DifferentiableBandPassFilter",
    "DimHandler",
    "DropoutChannels",
    "FreqGainChanger",
    "GaussianFilter",
    "HighPassFilter",
    "Hilbert",
    "LowPassFilter",
    "MNet1000",
    "MNet_1000",
    "MNet_config",
    "ModulationIndex",
    "PAC",
    "PSD",
    "ResNet1D",
    "ResNetBasicBlock",
    "ReshapeLayer",
    "SpatialAttention",
    "Spectrogram",
    "SwapChannels",
    "SwapLayer",
    "TransposeLayer",
    "Wavelet",
    "my_softmax",
    "normalize",
    "spectrograms",
    "unbias",
]

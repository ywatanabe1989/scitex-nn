"""Vendored subset of scitex.dsp.utils needed by scitex_nn.

Prefers the real scitex.dsp.utils when scitex umbrella is installed
(so behaviour stays in lockstep), falls back to the vendored copy when
scitex_nn is used standalone without scitex.dsp.
"""

# Always use the vendored copy by default to keep scitex_nn standalone.
# The umbrella ``scitex.dsp.utils`` import was unreliable across partial
# installs (ImportError vs AttributeError) and is no longer a runtime dep.
from ._differential_bandpass_filters import (
    build_bandpass_filters,
    init_bandpass_filters,
)
from ._ensure_3d import ensure_3d
from ._ensure_even_len import ensure_even_len
from ._zero_pad import zero_pad
from .filter import design_filter

__all__ = [
    "build_bandpass_filters",
    "design_filter",
    "ensure_3d",
    "ensure_even_len",
    "init_bandpass_filters",
    "zero_pad",
]

# scitex-nn

<!-- scitex-badges:start -->
[![PyPI](https://img.shields.io/pypi/v/scitex-nn.svg)](https://pypi.org/project/scitex-nn/)
[![Python](https://img.shields.io/pypi/pyversions/scitex-nn.svg)](https://pypi.org/project/scitex-nn/)
[![Tests](https://github.com/ywatanabe1989/scitex-nn/actions/workflows/test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-nn/actions/workflows/test.yml)
[![Install Test](https://github.com/ywatanabe1989/scitex-nn/actions/workflows/install-test.yml/badge.svg)](https://github.com/ywatanabe1989/scitex-nn/actions/workflows/install-test.yml)
[![Coverage](https://codecov.io/gh/ywatanabe1989/scitex-nn/graph/badge.svg)](https://codecov.io/gh/ywatanabe1989/scitex-nn)
[![Docs](https://readthedocs.org/projects/scitex-nn/badge/?version=latest)](https://scitex-nn.readthedocs.io/en/latest/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- scitex-badges:end -->

PyTorch neural-network building blocks (BNet, Hilbert, PAC, Wavelet, Filters, AxiswiseDropout, …) extracted from the [SciTeX](https://github.com/ywatanabe1989/scitex-python) ecosystem as a standalone package.

## Install

```bash
pip install scitex-nn
```

## API

```python
import scitex_nn as nn

m = nn.BNet(...)
m = nn.Filters(...)
m = nn.Hilbert(...)
m = nn.Wavelet(...)
m = nn.PAC(...)
```

## Status

Standalone fork of `scitex.nn`. The umbrella package's `scitex.nn` import path
is preserved via a `sys.modules`-alias bridge.

Decoupling notes:
- `scitex.{decorators,gen}` → `scitex_decorators` / `scitex_gen` direct imports.
- `scitex.dsp.utils` (build_bandpass_filters, init_bandpass_filters,
  ensure_3d, ensure_even_len, zero_pad, design_filter) → vendored under
  `_vendor_dsp_utils/`. Vendor prefers the real `scitex.dsp.utils` when
  the umbrella is installed (lockstep behaviour) and falls back to the
  vendored copy when scitex_nn runs standalone.
- `scitex.nn.X` self-references rewritten to `scitex_nn.X`.
- Example `if __name__ == "__main__":` blocks still reference
  `scitex.{io,plt,session,dsp,ai}` — only run when the umbrella is
  installed; module-level imports do not depend on those.

## License

AGPL-3.0-only.

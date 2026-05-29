---
description: |
  [TOPIC] Quick Start
  [DETAILS] Smallest useful example demonstrating the primary use case in
  under 30 seconds.
tags: [scitex-nn-quick-start]
---

# Quick Start

```python
import torch
import scitex_nn

x = torch.randn(8, 19, 1024)                 # (batch, channels, samples)

# Differentiable Hilbert transform
hilbert = scitex_nn.Hilbert(seq_len=1024)(x)  # analytic signal
phase, amplitude = hilbert[..., 0], hilbert[..., 1]

# Differentiable bandpass filter bank
bp = scitex_nn.BandPassFilter(low=4.0, high=8.0, fs=1000.0, order=4)
y = bp(x)                                     # (8, 19, 1, 1024)

# Phase-amplitude coupling
pac = scitex_nn.PAC(fs=1000.0)(y, y)
```

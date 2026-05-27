---
description: |
  [TOPIC] Python API
  [DETAILS] Public Python API of scitex-nn — exported functions, signatures,
  return types, and minimal usage examples per function.
tags: [scitex-nn-python-api]
---

# Python API

```python
import scitex_nn

# All public classes are importable directly:
#   Filters, Hilbert, Wavelet, PSD, Spectrogram,
#   ModulationIndex, PAC,
#   AxiswiseDropout, DropoutChannels, ChannelGainChanger,
#   FreqGainChanger, SwapChannels,
#   SpatialAttention, TransposeLayer,
#   ResNet1D, ResNetBasicBlock,
#   MNet1000, MNet_1000, MNet_config,
#   BNet_v1, BNet_config_v1, BHead_v1,
#   BNet_Res, BNet_config_Res, BHead_Res
```

## scitex_nn.BandPassFilter(low, high, fs, order=4, fp16=False) -> nn.Module

FIR-init bandpass filter. Operates on the last (time) axis of
`(batch, channels, samples)` tensors.

```python
bp = scitex_nn.BandPassFilter(low=4.0, high=8.0, fs=1000.0, order=4)
y = bp(x)  # (B, C, 1, T)
```

## scitex_nn.Hilbert(seq_len, dim=-1, fp16=False) -> nn.Module

Differentiable Hilbert transform returning analytic signal
(real + imaginary along `dim`, stacked as a trailing dim).

```python
h = scitex_nn.Hilbert(seq_len=1024, dim=-1)
analytic = h(x)  # (..., 2) — last dim: 0=real, 1=imag
phase, amp = analytic[..., 0], analytic[..., 1]
```

## scitex_nn.AxiswiseDropout(dropout_prob, dim) -> nn.Module

Dropout that zeros entire slices along a chosen axis during training.
Useful for channel-wise or time-segment dropout in SSL pipelines.

```python
drop = scitex_nn.AxiswiseDropout(dropout_prob=0.5, dim=1).train()
y = drop(x)  # random channels zeroed
```

## scitex_nn.PAC(fs, ...) -> nn.Module

Phase-amplitude coupling end-to-end pipeline.
Wraps bandpass filters + Hilbert + ModulationIndex.

```python
pac = scitex_nn.PAC(fs=1000.0)
comod = pac(pha, amp)  # (B, C, n_pha_bins, n_amp_bins)
```

## scitex_nn.ResNet1D(block, layers, ...) -> nn.Module

1D ResNet backbone with configurable blocks.

```python
model = scitex_nn.ResNet1D(
    block=scitex_nn.ResNetBasicBlock,
    layers=[2, 2, 2, 2],
    in_channels=19,
    out_channels=10,
)
```

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.fft import fft, ifft


class Hilbert(nn.Module):
    """Differentiable Hilbert transform → analytic signal.

    Matches ``scipy.signal.hilbert`` to float-precision. The analytic-
    signal mask is a constant tensor (1 at DC + Nyquist, 2 at positive
    frequencies, 0 at negative frequencies); multiplying by it is fully
    differentiable, so gradients flow through ``fft → mul → ifft →
    abs/atan2`` cleanly.

    Parameters
    ----------
    seq_len : int
        Length of the signal along the transform axis.
    dim : int
        Axis along which the Hilbert transform is applied. Default ``-1``.
    fp16 : bool
        Cast input to half precision (output is cast back to float32).
    in_place : bool
        If False, the input is cloned before processing.
    """

    def __init__(self, seq_len, dim=-1, fp16=False, in_place=False):
        super().__init__()
        self.dim = dim
        self.fp16 = fp16
        self.in_place = in_place
        self.n = seq_len

        # Canonical analytic-signal multiplier. 1 at DC, 2 at positive
        # bins, 0 at negative bins; for even N also 1 at Nyquist.
        h = torch.zeros(self.n)
        if self.n % 2 == 0:
            h[0] = 1.0
            h[self.n // 2] = 1.0
            h[1 : self.n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (self.n + 1) // 2] = 2.0
        self.register_buffer("h_mask", h)

    def hilbert_transform(self, x):
        orig_dtype = x.dtype
        # FFT does not support float16/bfloat16/int dtypes; promote those
        # to float32. float32/float64 pass through.
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()
        xf = fft(x, n=self.n, dim=self.dim)
        x = x.to(orig_dtype)

        # Broadcast the 1-D mask along self.dim.
        mult = self.h_mask.type_as(x)
        if x.ndim > 1:
            shape = [1] * x.ndim
            shape[self.dim] = self.n
            mult = mult.view(*shape)

        return ifft(xf * mult, dim=self.dim)

    def forward(self, x):
        if self.fp16:
            x = x.half()
        if not self.in_place:
            x = x.clone()

        z = self.hilbert_transform(x)
        pha = torch.atan2(z.imag, z.real)
        amp = z.abs()
        assert x.shape == pha.shape == amp.shape

        out = torch.cat([pha.unsqueeze(-1), amp.unsqueeze(-1)], dim=-1)
        if self.fp16:
            out = out.float()
        return out


# EOF

"""scitex-nn quickstart: differentiable Hilbert transform + axis-wise dropout.

scitex-nn provides drop-in PyTorch modules tailored for signal-processing.
"""

import torch

import scitex_nn


def main():
    torch.manual_seed(0)

    # 1. Hilbert: differentiable Hilbert transform that returns the analytic
    # signal as (amplitude, phase) along the last axis.
    fs = 256
    seq_len = 512
    t = torch.linspace(0, seq_len / fs, seq_len)
    # Two-channel signal: a 5 Hz sine + a 10 Hz sine.
    signal = torch.stack(
        [torch.sin(2 * torch.pi * 5 * t), torch.sin(2 * torch.pi * 10 * t)],
        dim=0,
    ).unsqueeze(0)  # shape: (batch=1, channels=2, seq_len)
    print("signal shape:", tuple(signal.shape))

    hilbert = scitex_nn.Hilbert(seq_len=seq_len, dim=-1)
    out = hilbert(signal)
    print("hilbert output shape:", tuple(out.shape))
    # Output stacks (amplitude, phase) along a new last dim.
    assert out.shape[:-1] == signal.shape
    assert out.shape[-1] == 2

    phase = out[..., 0]
    amplitude = out[..., 1]
    # For a unit-amplitude sine, the analytic-signal envelope ≈ 1.
    print("amplitude mean (5 Hz channel):", amplitude[0, 0].mean().item())
    print("amplitude mean (10 Hz channel):", amplitude[0, 1].mean().item())
    assert 0.5 < amplitude[0, 0].mean().item() < 1.5
    assert torch.isfinite(phase).all()

    # 2. AxiswiseDropout: zero out entire features along a chosen axis.
    drop = scitex_nn.AxiswiseDropout(dropout_prob=0.5, dim=1)
    drop.train()
    x = torch.ones(4, 8, 16)
    y = drop(x)
    print("\naxiswise dropout output shape:", tuple(y.shape))
    assert y.shape == x.shape


if __name__ == "__main__":
    main()

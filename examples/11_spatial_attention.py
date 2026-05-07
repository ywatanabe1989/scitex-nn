#!/usr/bin/env python3
"""SpatialAttention — channel-wise gain from a 1×1 conv.

``scitex_nn.SpatialAttention`` is a tiny block: average-pool over time,
then a 1×1 conv ``(n_chs_in → 1)`` produces one scalar per batch which
is broadcast back across all channels and time-points to gate the
input. The intuition: the network learns *how much* of each channel
contributes overall — not where in time it matters.

This example feeds an 8-channel input where channels 0/3/6 carry a
strong signal and the rest carry weak noise, then plots the input,
the post-attention output, and the per-channel gain (the residual
gain after the broadcast multiplication).

Run:
    python 11_spatial_attention.py
"""

from pathlib import Path

import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    n_chs: int = 8,
    seq_len: int = 256,
    fs: int = 256,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Apply SpatialAttention to a structured 8-channel input."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    t = torch.linspace(0, seq_len / fs, seq_len)
    base = torch.zeros(n_chs, seq_len)
    strong = {0, 3, 6}
    for c in range(n_chs):
        if c in strong:
            base[c] = torch.sin(2 * torch.pi * (5 + c) * t)
        else:
            base[c] = 0.1 * torch.randn(seq_len)
    x = base.unsqueeze(0)

    sa = scitex_nn.SpatialAttention(n_chs_in=n_chs)
    y = sa(x).detach()
    gain = (y / (x + 1e-12)).detach().squeeze(0).mean(dim=-1)
    logger.info(f"per-channel gain: {gain.tolist()}")
    logger.info(f"output shape: {tuple(y.shape)}")

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=False,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=32,
    )
    for ch in range(n_chs):
        axes[0].plot(t.numpy(), x[0, ch].numpy() + ch * 2.5, lw=0.6)
    axes[0].set_xyt(None, "channel (offset)", "input — channels 0/3/6 carry signal")

    for ch in range(n_chs):
        axes[1].plot(t.numpy(), y[0, ch].numpy() + ch * 2.5, lw=0.6)
    axes[1].set_xyt(None, "channel (offset)", "after SpatialAttention")

    axes[2].bar(range(n_chs), gain.numpy())
    axes[2].set_xyt("channel", "mean gain", "Per-channel gain learned by attention")

    stx.io.save(fig, OUT / "spatial_attention.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

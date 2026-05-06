#!/usr/bin/env python3
"""Channel augmentations — DropoutChannels, SwapChannels, ChannelGainChanger.

Three SSL-style augmentations that perturb the channel axis of a
``(batch, n_chs, seq_len)`` tensor:

- ``DropoutChannels(p)`` — replaces a fraction of channels with i.i.d. noise
- ``SwapChannels(p)`` — randomly permutes a fraction of channels
- ``ChannelGainChanger(n_chs)`` — applies a softmax-normalized random gain

The figure overlays a small EEG-shape signal (single trial, first 4
channels) before vs after each augmentation so the effect is visible
trace-by-trace.

Run:
    python 03_channel_aug.py
    python 03_channel_aug.py --p 0.5 --n-chs 8 --seed 1
"""

from pathlib import Path

import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    p: float = 0.4,
    n_chs: int = 8,
    seq_len: int = 256,
    n_show: int = 4,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Compare three channel augmentations on a synthetic multi-channel signal."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    t = torch.linspace(0, 1, seq_len)
    x = torch.stack(
        [torch.sin(2 * torch.pi * (3 + i) * t) for i in range(n_chs)], dim=0
    ).unsqueeze(0)

    drop = scitex_nn.DropoutChannels(dropout=p).train()
    swap = scitex_nn.SwapChannels(dropout=p).train()
    gain = scitex_nn.ChannelGainChanger(n_chs=n_chs).train()

    y_drop = drop(x.clone()).detach()
    y_swap = swap(x.clone()).detach()
    y_gain = gain(x.clone()).detach()

    logger.info(f"input range: [{x.min():.3f}, {x.max():.3f}]")
    logger.info(f"after DropoutChannels  : [{y_drop.min():.3f}, {y_drop.max():.3f}]")
    logger.info(f"after ChannelGainChanger: [{y_gain.min():.4f}, {y_gain.max():.4f}]")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=30,
    )
    rows = [
        ("input", x[0]),
        (f"DropoutChannels(p={p}) — random channels → noise", y_drop[0]),
        (f"SwapChannels(p={p}) — random channels permuted", y_swap[0]),
        ("ChannelGainChanger — per-channel softmax gain", y_gain[0]),
    ]
    for ax, (title, mat) in zip(axes, rows):
        for ch in range(n_show):
            ax.plot(t.numpy(), mat[ch].numpy() + ch * 2.5, lw=0.7)
        ax.set_xyt(None, "channel (offset)", title)
    axes[-1].set_xyt("time [s]", "channel (offset)", rows[-1][0])

    stx.io.save(fig, OUT / "channel_aug.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

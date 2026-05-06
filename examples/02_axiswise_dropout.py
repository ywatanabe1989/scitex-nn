#!/usr/bin/env python3
"""AxiswiseDropout — drop entire slices along a chosen axis.

Standard ``nn.Dropout`` is element-wise. ``AxiswiseDropout`` zeros out
entire slices along one axis (e.g. whole channels, whole time-points,
whole feature columns) — useful when neighbouring elements within a
slice are correlated and dropping individual elements does not
decorrelate the unit.

The figure shows the same ``(B, C, T)`` tensor under three settings:
``dim=1`` (channel-wise), ``dim=2`` (time-wise), and standard
element-wise dropout for contrast. Eval-mode is the identity.

Run:
    python 02_axiswise_dropout.py
    python 02_axiswise_dropout.py --p 0.7 --seed 1
"""

from pathlib import Path

import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    p: float = 0.5,
    n_chs: int = 8,
    seq_len: int = 24,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Visualize axis-wise vs element-wise dropout masks."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    x = torch.ones(1, n_chs, seq_len)
    drop_chan = scitex_nn.AxiswiseDropout(dropout_prob=p, dim=1).train()
    drop_time = scitex_nn.AxiswiseDropout(dropout_prob=p, dim=2).train()
    drop_elem = torch.nn.Dropout(p=p).train()

    y_chan = drop_chan(x)[0].detach().numpy()
    y_time = drop_time(x)[0].detach().numpy()
    y_elem = drop_elem(x)[0].detach().numpy()

    drop_chan.eval()
    y_eval = drop_chan(x)[0].detach().numpy()
    assert (y_eval == 1.0).all(), "AxiswiseDropout must be identity in eval mode"
    logger.info(f"channel-drop kept {(y_chan.any(axis=1) > 0).sum()}/{n_chs} channels")
    logger.info(f"time-drop kept {(y_time.any(axis=0) > 0).sum()}/{seq_len} timepoints")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        sharex=True,
        sharey=True,
        axes_width_mm=140,
        axes_height_mm=22,
    )
    titles = [
        "input (all ones)",
        f"AxiswiseDropout(dim=1, p={p}) — entire channels dropped",
        f"AxiswiseDropout(dim=2, p={p}) — entire timepoints dropped",
        f"nn.Dropout(p={p}) — element-wise (for contrast)",
    ]
    for ax, mat, title in zip(axes, [x[0].numpy(), y_chan, y_time, y_elem], titles):
        ax.imshow(mat, aspect="auto", cmap="gray_r", vmin=0, vmax=2.5)
        ax.set_xyt(None, "channel", title)
    axes[-1].set_xyt("time", "channel", titles[-1])

    stx.io.save(fig, OUT / "axiswise_dropout.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

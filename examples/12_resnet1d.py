#!/usr/bin/env python3
"""ResNet1D — small training-loss curve on synthetic 1D data.

``scitex_nn.ResNet1D`` is a backbone of stacked
``ResNetBasicBlock`` (three Conv1d→BN→ReLU layers with a residual skip
and 1×1 expansion). It returns a feature map ``(B, n_filts, T)`` — no
classifier head; users plug their own.

This example wraps it in a tiny classifier (mean-pool + Linear) and
trains a few hundred steps on a 3-class synthetic problem (sine,
chirp, noise). The training-loss curve drops monotonically — a
sanity check that the residual blocks differentiate properly.

Run:
    python 12_resnet1d.py
    python 12_resnet1d.py --n-blks 3 --n-steps 400
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch
import torch.nn as nn
from scipy.signal import chirp

import scitex_nn


def _synth(n_per_class, fs, seq_len, rng):
    t = np.arange(seq_len) / fs
    x_list, y_list = [], []
    for cls in range(3):
        for _ in range(n_per_class):
            if cls == 0:
                f = rng.uniform(5, 10)
                sig = np.sin(2 * np.pi * f * t)
            elif cls == 1:
                sig = chirp(t, f0=4, t1=t[-1], f1=40)
            else:
                sig = rng.standard_normal(seq_len) * 0.5
            sig = sig + 0.05 * rng.standard_normal(seq_len)
            x_list.append(sig.astype(np.float32))
            y_list.append(cls)
    return np.stack(x_list), np.array(y_list)


@stx.session
def main(
    n_per_class: int = 64,
    fs: int = 200,
    seq_len: int = 256,
    n_chs: int = 1,
    n_blks: int = 2,
    n_steps: int = 200,
    batch: int = 16,
    lr: float = 1e-3,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Train a tiny ResNet1D classifier on 3-class synthetic 1D data."""
    OUT = Path(CONFIG.SDIR_RUN)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X, y = _synth(n_per_class, fs, seq_len, rng)
    X = torch.from_numpy(X).unsqueeze(1)
    y = torch.from_numpy(y).long()
    n = len(X)
    logger.info(f"dataset: {n} examples, balanced 3-class, shape {tuple(X.shape)}")

    backbone = scitex_nn.ResNet1D(n_chs=n_chs, n_out=3, n_blks=n_blks)
    n_filts = n_chs * 4
    head = nn.Linear(n_filts, 3)
    model = nn.Sequential(backbone, nn.AdaptiveAvgPool1d(1), nn.Flatten(), head)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ResNet1D + head: {n_params:,} parameters")

    # Manual SGD step — torch.optim.* triggers torch._dynamo import that hits
    # a torch._inductor circular-import bug on some CPU torch builds. The
    # loop below is equivalent to vanilla SGD with momentum.
    momentum = 0.9
    velocities = [torch.zeros_like(p) for p in model.parameters()]
    losses, accs = [], []
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch,))
        xb, yb = X[idx], y[idx]
        logits = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss.backward()
        with torch.no_grad():
            for p, v in zip(model.parameters(), velocities):
                if p.grad is None:
                    continue
                v.mul_(momentum).add_(p.grad)
                p.add_(v, alpha=-lr * 10)
        losses.append(float(loss.item()))
        accs.append(float((logits.argmax(dim=-1) == yb).float().mean()))

    final_acc = float(accs[-1])
    logger.info(f"final train batch loss = {losses[-1]:.4f} | acc = {final_acc:.3f}")
    assert final_acc > 0.6, f"ResNet1D failed to learn: final acc {final_acc:.2f}"

    fig, axes = plt.subplots(
        nrows=2, ncols=1, sharex=True, axes_width_mm=160, axes_height_mm=42
    )
    axes[0].plot(losses, lw=0.8)
    axes[0].set_xyt(None, "train loss", "ResNet1D — cross-entropy")
    axes[1].plot(accs, lw=0.8)
    axes[1].set_xyt(
        "step", "batch acc", f"Train batch accuracy (final = {final_acc:.2f})"
    )

    stx.io.save(fig, OUT / "resnet1d.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

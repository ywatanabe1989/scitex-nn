#!/usr/bin/env python3
"""MNet1000 — forward + backward + per-parameter gradient norms.

``scitex_nn.MNet1000`` is a 4-stage Conv2d backbone for multi-channel
1-second EEG/MEG (≈ 19-270 channels at 1 kHz). The forward path
unfolds time, applies depth-wise then point-wise 2D conv stacks with
Mish activations, and projects through two fully-connected layers to
``len(classes)`` logits.

This example builds a small MNet1000 (capped batch ≤ 8 to fit in
typical CPU/GPU memory — see HANDOFF "Known traps"), runs one forward
pass through synthetic EEG-shape data, computes a cross-entropy loss
with random labels, and plots the resulting per-parameter gradient
norm so it is visible at a glance which layers receive the most
gradient signal.

Run:
    python 15_mnet_classifier.py
    python 15_mnet_classifier.py --batch 4 --n-chs 19 --seq-len 1000
"""

from pathlib import Path

import scitex as stx
import torch
import torch.nn as nn

import scitex_nn


@stx.session
def main(
    batch: int = 4,
    n_chs: int = 19,
    seq_len: int = 1000,
    n_fc1: int = 256,
    n_fc2: int = 128,
    n_classes: int = 4,
    seed: int = 42,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Forward + backward pass through MNet1000 on synthetic EEG."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    cfg = {
        "n_chs": n_chs,
        "n_fc1": n_fc1,
        "n_fc2": n_fc2,
        "d_ratio1": 0.5,
        "d_ratio2": 0.5,
        "classes": list(range(n_classes)),
    }
    model = scitex_nn.MNet1000(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"MNet1000 — {n_params:,} parameters")

    x = torch.randn(batch, n_chs, seq_len)
    y_true = torch.randint(0, n_classes, (batch,))

    logits = model(x)
    assert logits.shape == (batch, n_classes), f"got {logits.shape}"
    logger.info(f"logits.shape = {tuple(logits.shape)}")

    loss = nn.functional.cross_entropy(logits, y_true)
    loss.backward()
    grad_norms = [
        (name, p.grad.norm().item())
        for name, p in model.named_parameters()
        if p.grad is not None
    ]
    logger.info(f"loss = {loss.item():.4f}")
    logger.info(f"params with grad: {len(grad_norms)} / {n_params}")
    assert all(g > 0 for _, g in grad_norms), "dead gradient detected"

    fig, ax = plt.subplots(axes_width_mm=160, axes_height_mm=110)
    names = [n.split(".")[-2:] for n, _ in grad_norms]
    labels = [".".join(n) for n in names]
    values = [g for _, g in grad_norms]
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xyt(
        "‖grad‖",
        "parameter",
        f"MNet1000 per-parameter gradient norms ({n_params:,} params)",
    )
    stx.io.save(fig, OUT / "mnet_gradient_norms.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

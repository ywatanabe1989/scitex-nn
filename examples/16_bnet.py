#!/usr/bin/env python3
"""BNet — multi-modality B-shaped backbone built on top of MNet1000.

``scitex_nn.BNet_v1`` is the multi-modality wrapper around
``MNet1000``: each modality (e.g. EEG vs MEG) gets its own
``BHead`` (channel attention + 1×1 → ``N_VIRTUAL_CHS`` virtual
channels) and its own classifier head; the shared MNet backbone is
called via ``forward(x, i_head)``.

This example builds a two-modality BNet (19-channel "EEG" classifier
with 4 classes; 32-channel "MEG" classifier with 2 classes), runs a
forward pass through each head, and bar-charts the parameter count
of the shared backbone vs each per-head tower.

The example caps the inner MNet1000 to its canonical
``seq_len=1000``; deviations break the hardcoded ``N_FC_IN = 15950``
post-conv flatten size in ``_MNet_1000.py``.

Run:
    python 16_bnet.py
    python 16_bnet.py --batch 2
"""

from pathlib import Path

import scitex as stx
import torch
import torch.nn as nn

import scitex_nn


@stx.session
def main(
    batch: int = 1,
    seq_len: int = 1000,
    n_chs_eeg: int = 19,
    n_chs_meg: int = 32,
    n_classes_eeg: int = 4,
    n_classes_meg: int = 2,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Build a 2-modality BNet, forward each head, plot module param counts."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    BNet_config = {
        "n_bands": 6,
        "SAMP_RATE": 250,
        "n_chs": [n_chs_eeg, n_chs_meg],
        "n_classes": [n_classes_eeg, n_classes_meg],
        "n_fc1": 1024,
        "d_ratio1": 0.5,
        "n_fc2": 256,
        "d_ratio2": 0.5,
    }
    MNet_config = {
        "classes": list(range(max(n_classes_eeg, n_classes_meg))),
        "n_chs": 32,
        "n_fc1": 1024,
        "d_ratio1": 0.5,
        "n_fc2": 256,
        "d_ratio2": 0.5,
    }
    model = scitex_nn.BNet_v1(BNet_config, MNet_config)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(f"BNet_v1 — {n_total:,} parameters total")

    x_eeg = torch.randn(batch, n_chs_eeg, seq_len)
    x_meg = torch.randn(batch, n_chs_meg, seq_len)

    y_eeg = model(x_eeg, i_head=0)
    y_meg = model(x_meg, i_head=1)
    assert y_eeg.shape == (batch, n_classes_eeg)
    assert y_meg.shape == (batch, n_classes_meg)
    logger.info(f"head 0 (EEG) → {tuple(y_eeg.shape)}")
    logger.info(f"head 1 (MEG) → {tuple(y_meg.shape)}")

    loss = nn.functional.cross_entropy(
        y_eeg, torch.zeros(batch, dtype=torch.long)
    ) + nn.functional.cross_entropy(y_meg, torch.zeros(batch, dtype=torch.long))
    loss.backward()
    logger.info(f"loss = {loss.item():.4f} (gradients flow through both heads)")

    submodules = {
        "BHead[0] (EEG)": model.heads[0],
        "BHead[1] (MEG)": model.heads[1],
        "MNet (shared)": model.MNet,
        "fc[0] (EEG)": model.fcs[0],
        "fc[1] (MEG)": model.fcs[1],
    }
    counts = {
        name: sum(p.numel() for p in mod.parameters())
        for name, mod in submodules.items()
    }
    for name, c in counts.items():
        logger.info(f"  {name:>20s}  {c:>12,d}  ({c / n_total:.1%})")

    fig, ax = plt.subplots(axes_width_mm=160, axes_height_mm=70)
    names = list(counts.keys())
    values = list(counts.values())
    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xscale("log")
    ax.set_xyt(
        "parameter count (log)",
        "submodule",
        f"BNet_v1 parameter distribution — total {n_total:,}",
    )

    stx.io.save(fig, OUT / "bnet.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

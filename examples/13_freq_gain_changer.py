#!/usr/bin/env python3
"""FreqGainChanger — random per-band gain in the frequency domain.

``scitex_nn.FreqGainChanger`` splits the input into ``n_bands`` non-
overlapping frequency bands (via ``julius.bands.split_bands``),
applies a softmax-normalised random gain per band each forward pass,
and recombines. Use it as an SSL-style frequency-domain augmentation
analogous to ``ChannelGainChanger`` in the channel domain.

This example runs the augmentation on a flat-spectrum white-noise
input several times and overlays the per-band PSDs. The variance
across runs in each band shows the random gain spread; the overall
shape stays comparable thanks to the softmax normalisation.

Run:
    python 13_freq_gain_changer.py
    python 13_freq_gain_changer.py --n-bands 6 --n-runs 12
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    fs: int = 256,
    seq_len: int = 1024,
    n_bands: int = 4,
    n_runs: int = 8,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Apply FreqGainChanger to white noise; overlay PSDs across runs."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    fg = scitex_nn.FreqGainChanger(n_bands=n_bands, samp_rate=fs).train()
    psd_layer = scitex_nn.PSD(sample_rate=fs)

    x = torch.randn(1, 1, seq_len)
    psd_orig, freqs = psd_layer(x.squeeze())
    freqs = freqs.numpy()

    augmented = []
    for _ in range(n_runs):
        y = fg(x.clone())
        psd_y, _ = psd_layer(y.squeeze())
        augmented.append(psd_y.numpy())
    aug = np.stack(augmented)
    logger.info(f"per-run PSD std (avg): {aug.std(axis=0).mean():.3e}")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=42,
    )
    axes[0].plot(freqs, psd_orig.numpy(), color="black", lw=1.0)
    axes[0].set_yscale("log")
    axes[0].set_xyt(None, "power", f"Original white noise — flat PSD ({n_bands} bands)")

    for i, mag in enumerate(aug):
        axes[1].plot(freqs, mag, lw=0.6, alpha=0.7, label=f"run {i}" if i < 3 else None)
    axes[1].plot(freqs, aug.mean(axis=0), color="black", lw=1.4, label="mean over runs")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].set_xyt(
        "frequency [Hz]",
        "power",
        f"After FreqGainChanger ({n_runs} runs) — softmax-normalized random per-band gain",
    )

    stx.io.save(fig, OUT / "freq_gain_changer.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

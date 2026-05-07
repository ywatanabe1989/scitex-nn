#!/usr/bin/env python3
"""DifferentiableBandPassFilter — bands as learnable hyperparameters.

``DifferentiableBandPassFilter`` wraps the static ``BandPassFilter``
so the centre frequencies (``pha_mids``, ``amp_mids``) carry gradients.
Forward looks identical; backward updates the band locations.

The figure plots the impulse response of every filter (one row per
band) on a magnitude-vs-frequency panel so the band tiling is visible
at a glance.

Run:
    python 06b_differentiable_bandpass.py
    python 06b_differentiable_bandpass.py --pha-n-bands 6 --amp-n-bands 8
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    fs: int = 500,
    seq_len: int = 1024,
    pha_low: float = 2.0,
    pha_high: float = 20.0,
    pha_n_bands: int = 6,
    amp_low: float = 30.0,
    amp_high: float = 120.0,
    amp_n_bands: int = 8,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Forward + backward through DifferentiableBandPassFilter."""
    OUT = Path(CONFIG.SDIR_RUN)

    filt = scitex_nn.DifferentiableBandPassFilter(
        sig_len=seq_len,
        fs=fs,
        pha_low_hz=pha_low,
        pha_high_hz=pha_high,
        pha_n_bands=pha_n_bands,
        amp_low_hz=amp_low,
        amp_high_hz=amp_high,
        amp_n_bands=amp_n_bands,
    )
    n_total = pha_n_bands + amp_n_bands

    x = torch.randn(1, 1, seq_len, requires_grad=True)
    y = filt(x)
    logger.info(f"output shape: {tuple(y.shape)}")
    assert y.shape == (1, 1, n_total, seq_len), y.shape

    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "no gradient flowed"
    logger.info(f"loss = {loss.item():.4f}; grad norm = {x.grad.norm().item():.4f}")

    impulse = torch.zeros(1, 1, seq_len)
    impulse[0, 0, seq_len // 2] = 1.0
    with torch.no_grad():
        ir = filt(impulse).squeeze().numpy()
    freqs = np.fft.rfftfreq(seq_len, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(ir, axis=-1))
    mag /= mag.max(axis=-1, keepdims=True) + 1e-12

    pha_mids = filt.pha_mids.detach().numpy()
    amp_mids = filt.amp_mids.detach().numpy()
    centres = np.concatenate([pha_mids, amp_mids])

    fig, ax = plt.subplots(axes_width_mm=160, axes_height_mm=80)
    for k, c in enumerate(centres):
        kind = "pha" if k < pha_n_bands else "amp"
        label = f"{kind} @ {c:.1f} Hz" if k % 4 == 0 else None
        ax.plot(freqs, mag[k], lw=0.8, label=label)
    ax.set_xlim(0, min(fs / 2, amp_high * 1.2))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.set_xyt(
        "frequency [Hz]",
        "magnitude (normalized)",
        f"DifferentiableBandPassFilter — {pha_n_bands} pha + {amp_n_bands} amp bands",
    )

    stx.io.save(fig, OUT / "differentiable_bandpass.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

#!/usr/bin/env python3
"""ModulationIndex — Tort 2010 KL-divergence MI on a single coupling.

``scitex_nn.ModulationIndex`` computes the modulation index defined by
Tort et al. (2010): the KL-divergence between the observed amplitude
distribution over phase bins and a uniform reference, normalised by
``log(n_bins)``. MI is in ``[0, 1]``.

The example builds a textbook PAC signal — a 6 Hz theta wave whose
phase modulates the amplitude of a 60 Hz gamma carrier — and contrasts
it with a 6 Hz theta unrelated to a separate 60 Hz gamma. The MI is
high (≈0.4) for the coupled pair and near zero for the uncoupled pair.

Run:
    python 09_modulation_index.py
    python 09_modulation_index.py --duration 8 --n-bins 24
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch

import scitex_nn


def _bandpass(x, fs, lo, hi, order=4):
    """SciPy bandpass for the demo signal — clearer than the nn filter here."""
    from scipy.signal import butter, filtfilt

    b, a = butter(order, [lo, hi], btype="band", fs=fs)
    return filtfilt(b, a, x).copy()


@stx.session
def main(
    duration: float = 6.0,
    fs: int = 500,
    pha_freq: float = 6.0,
    amp_freq: float = 60.0,
    coupling_strength: float = 0.8,
    n_bins: int = 18,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Compute MI on a coupled vs uncoupled (theta, gamma) pair."""
    OUT = Path(CONFIG.SDIR_RUN)
    rng = np.random.default_rng(seed)

    seq_len = int(duration * fs)
    t = np.arange(seq_len) / fs
    theta = np.sin(2 * np.pi * pha_freq * t)

    envelope_coupled = 1 + coupling_strength * (theta + 1) / 2
    gamma_coupled = envelope_coupled * np.sin(2 * np.pi * amp_freq * t)
    sig_coupled = theta + gamma_coupled + 0.05 * rng.standard_normal(seq_len)

    sig_uncoupled = (
        theta + np.sin(2 * np.pi * amp_freq * t) + 0.05 * rng.standard_normal(seq_len)
    )

    def _mi(sig):
        pha_band = _bandpass(sig, fs, pha_freq - 1.5, pha_freq + 1.5)
        amp_band = _bandpass(sig, fs, amp_freq - 10, amp_freq + 10)
        pha = torch.from_numpy(np.angle(_analytic(pha_band))).float()
        amp = torch.from_numpy(np.abs(_analytic(amp_band))).float()
        pha = pha.view(1, 1, 1, 1, seq_len)
        amp = amp.view(1, 1, 1, 1, seq_len)
        mi_layer = scitex_nn.ModulationIndex(n_bins=n_bins)
        return float(mi_layer(pha, amp).item())

    def _analytic(x):
        from scipy.signal import hilbert as scipy_hilbert

        return scipy_hilbert(x)

    mi_c = _mi(sig_coupled)
    mi_u = _mi(sig_uncoupled)
    logger.info(f"MI coupled   = {mi_c:.4f}")
    logger.info(f"MI uncoupled = {mi_u:.4f}")
    assert mi_c > mi_u, f"coupled MI ({mi_c}) should exceed uncoupled MI ({mi_u})"

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        sharey=True,
        axes_width_mm=160,
        axes_height_mm=42,
    )
    axes[0].plot(t, sig_coupled, color="black", lw=0.5)
    axes[0].plot(t, theta * 1.5, color="grey", ls=":", label=f"theta {pha_freq:.0f} Hz")
    axes[0].set_xlim(0, min(2.0, duration))
    axes[0].legend(loc="upper right")
    axes[0].set_xyt(
        None,
        "amplitude",
        f"Coupled (theta phase modulates gamma amplitude) — MI = {mi_c:.3f}",
    )
    axes[1].plot(t, sig_uncoupled, color="black", lw=0.5)
    axes[1].plot(t, theta * 1.5, color="grey", ls=":", label=f"theta {pha_freq:.0f} Hz")
    axes[1].legend(loc="upper right")
    axes[1].set_xyt(
        "time [s]",
        "amplitude",
        f"Uncoupled (independent gamma) — MI = {mi_u:.3f}",
    )

    stx.io.save(fig, OUT / "modulation_index.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

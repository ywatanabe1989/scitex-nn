#!/usr/bin/env python3
"""PAC — phase-amplitude coupling comodulogram.

``scitex_nn.PAC`` chains the package's bandpass bank, ``Hilbert``, and
``ModulationIndex`` into a single ``nn.Module``. Input ``(B, C, T)``
gives an MI heat-map indexed by phase-frequency × amplitude-frequency
— the canonical comodulogram in the Tort 2010 sense.

This example feeds a synthetic theta-gamma coupled signal (theta
phase modulates 60 Hz gamma amplitude) and shows the comodulogram has
a clear hot-spot at ``(pha≈6 Hz, amp≈60 Hz)``. With ``pha_n_bands``
and ``amp_n_bands`` modest, the whole pipeline runs on CPU in seconds.

Run:
    python 10_pac.py
    python 10_pac.py --duration 4 --pha-n-bands 12 --amp-n-bands 16
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    duration: float = 4.0,
    fs: int = 250,
    pha_freq: float = 6.0,
    amp_freq: float = 60.0,
    coupling_strength: float = 0.8,
    pha_start: float = 2.0,
    pha_end: float = 20.0,
    pha_n_bands: int = 12,
    amp_start: float = 30.0,
    amp_end: float = 100.0,
    amp_n_bands: int = 16,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Comodulogram of synthetic theta-gamma PAC signal."""
    OUT = Path(CONFIG.SDIR_RUN)
    rng = np.random.default_rng(seed)

    seq_len = int(duration * fs)
    t = np.arange(seq_len) / fs
    theta = np.sin(2 * np.pi * pha_freq * t)
    envelope = 1 + coupling_strength * (theta + 1) / 2
    gamma = envelope * np.sin(2 * np.pi * amp_freq * t)
    sig = (theta + gamma + 0.05 * rng.standard_normal(seq_len)).astype(np.float32)

    x = torch.from_numpy(sig).view(1, 1, -1)
    pac = scitex_nn.PAC(
        seq_len=seq_len,
        fs=fs,
        pha_start_hz=pha_start,
        pha_end_hz=pha_end,
        pha_n_bands=pha_n_bands,
        amp_start_hz=amp_start,
        amp_end_hz=amp_end,
        amp_n_bands=amp_n_bands,
        fp16=False,
    )
    como = pac(x).squeeze().detach().float().numpy()
    pha_mids = pac.PHA_MIDS_HZ.numpy()
    amp_mids = pac.AMP_MIDS_HZ.numpy()
    logger.info(f"comodulogram shape: {como.shape}; max MI = {como.max():.4f}")

    pha_peak = pha_mids[int(np.argmax(como.max(axis=1)))]
    amp_peak = amp_mids[int(np.argmax(como.max(axis=0)))]
    logger.info(f"peak at phase ≈ {pha_peak:.2f} Hz, amp ≈ {amp_peak:.2f} Hz")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=False,
        sharey=False,
        axes_width_mm=70,
        axes_height_mm=70,
    )
    show_t = min(2.0, duration)
    n_show = int(show_t * fs)
    axes[0].plot(t[:n_show], sig[:n_show], color="black", lw=0.5)
    axes[0].plot(t[:n_show], theta[:n_show] * 1.5, color="grey", ls=":", label="theta")
    axes[0].set_xyt(
        "time [s]",
        "amplitude",
        f"Synthetic PAC ({pha_freq:.0f}-{amp_freq:.0f} Hz coupling)",
    )
    axes[0].legend(loc="upper right")

    im = axes[1].imshow(
        como,
        aspect="auto",
        origin="lower",
        extent=(amp_mids[0], amp_mids[-1], pha_mids[0], pha_mids[-1]),
        cmap="viridis",
    )
    axes[1].set_xyt(
        "amplitude freq [Hz]",
        "phase freq [Hz]",
        f"Comodulogram — peak at ({pha_peak:.1f}, {amp_peak:.1f}) Hz",
    )

    stx.io.save(fig, OUT / "pac.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

#!/usr/bin/env python3
"""Spectrogram — STFT magnitude on a chirp.

``scitex_nn.Spectrogram`` is a per-channel STFT wrapper. It returns
``(spectrograms, freqs, times)`` where ``spectrograms`` is shaped
``(B, C, F, T)``. The window defaults to Hann; ``hop_length`` defaults
to ``n_fft // 4``.

This example sweeps a 5 → 60 Hz linear chirp and shows the
time-frequency distribution. The bright diagonal traces the
instantaneous frequency. We also plot the FFT-magnitude side-by-side
to make explicit that the spectrogram resolves time, the PSD does not.

Run:
    python 07_spectrogram.py
    python 07_spectrogram.py --duration 4 --fs 500 --n-fft 256
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch
from scipy.signal import chirp

import scitex_nn


@stx.session
def main(
    duration: float = 2.0,
    fs: int = 500,
    f0: float = 5.0,
    f1: float = 60.0,
    n_fft: int = 256,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """STFT of a 5→60 Hz chirp; show the time-frequency ridge."""
    OUT = Path(CONFIG.SDIR_RUN)

    seq_len = int(fs * duration)
    t = np.arange(seq_len) / fs
    sig_np = chirp(t, f0=f0, t1=t[-1], f1=f1).astype(np.float32)
    x = torch.from_numpy(sig_np).view(1, 1, -1)

    sg = scitex_nn.Spectrogram(sampling_rate=fs, n_fft=n_fft)
    spec, freqs, times = sg(x)
    spec_db = 20 * torch.log10(spec.squeeze() + 1e-10)
    logger.info(f"spectrogram shape: {tuple(spec.shape)}")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=False,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=50,
    )
    axes[0].plot(t, sig_np, color="black", lw=0.5)
    axes[0].set_xlim(0, duration)
    axes[0].set_xyt(
        "time [s]",
        "amplitude",
        f"Linear chirp {f0:.0f}→{f1:.0f} Hz, fs={fs}",
    )

    axes[1].imshow(
        spec_db.numpy(),
        aspect="auto",
        origin="lower",
        extent=(0, float(times[-1]), 0, float(freqs[-1])),
        cmap="viridis",
    )
    axes[1].plot(t, np.linspace(f0, f1, seq_len), color="white", ls=":", lw=0.8)
    axes[1].set_xyt(
        "time [s]",
        "frequency [Hz]",
        f"STFT magnitude (dB) — n_fft={n_fft}, hop={n_fft // 4}",
    )

    stx.io.save(fig, OUT / "spectrogram.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

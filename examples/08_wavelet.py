#!/usr/bin/env python3
"""Wavelet — Morlet continuous wavelet transform.

``scitex_nn.Wavelet`` builds a bank of complex Morlet kernels at
log- or linear-spaced centre frequencies (default: linear, one per Hz
up to Nyquist) and convolves the input with each kernel. The output
gives ``(phase, amplitude, freqs)`` per (batch, channel, frequency,
time) sample.

Compared to ``Spectrogram`` (fixed window length per FFT) the Morlet
transform adapts the window length to the carrier frequency — better
time resolution at high frequency, better frequency resolution at low.
This example reuses the same chirp from ``07_spectrogram`` to make the
trade-off legible.

Run:
    python 08_wavelet.py
    python 08_wavelet.py --duration 3 --fs 500
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
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Morlet CWT of a chirp; show amplitude scalogram."""
    OUT = Path(CONFIG.SDIR_RUN)

    seq_len = int(fs * duration)
    t = np.arange(seq_len) / fs
    sig = chirp(t, f0=f0, t1=t[-1], f1=f1).astype(np.float32)
    x = torch.from_numpy(sig).view(1, 1, -1)

    wav = scitex_nn.Wavelet(samp_rate=fs, freq_scale="linear", out_scale="linear")
    pha, amp, freqs = wav(x)
    amp = amp.squeeze().numpy()
    freqs = freqs.squeeze().numpy()
    logger.info(
        f"scalogram shape: {amp.shape}; freqs span {freqs.min():.1f}-{freqs.max():.1f} Hz"
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=False,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=50,
    )
    axes[0].plot(t, sig, color="black", lw=0.5)
    axes[0].set_xlim(0, duration)
    axes[0].set_xyt("time [s]", "amplitude", f"Chirp {f0:.0f}→{f1:.0f} Hz")

    axes[1].imshow(
        amp,
        aspect="auto",
        origin="lower",
        extent=(0.0, float(t[-1]), float(freqs[0]), float(freqs[-1])),
        cmap="viridis",
    )
    axes[1].plot(t, np.linspace(f0, f1, seq_len), color="white", ls=":", lw=0.8)
    axes[1].set_xyt(
        "time [s]",
        "frequency [Hz]",
        "Morlet CWT amplitude — adaptive time-frequency resolution",
    )

    stx.io.save(fig, OUT / "wavelet.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

#!/usr/bin/env python3
"""Filter bank — Low/High/Band/BandStop frequency responses.

Each filter class in ``scitex_nn`` is a thin ``nn.Module`` wrapper
around an FIR kernel (registered as a buffer; non-trainable). To see
what a filter does, push an impulse through it and FFT the output —
the magnitude is the frequency response.

This example runs an impulse through:
- ``LowPassFilter(cutoffs_hz=20)``
- ``HighPassFilter(cutoffs_hz=20)``
- ``BandPassFilter(bands=[[8, 30]])``
- ``BandStopFilter(bands=[[8, 30]])``

then plots the four responses on a single ``magnitude vs Hz`` panel.
The crossings at the design cutoffs make the band edges read directly.

Run:
    python 05_filter_bank.py
    python 05_filter_bank.py --fs 500 --seq-len 1024 --cutoff 20 --band 8 30
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
    cutoff: float = 20.0,
    band_low: float = 8.0,
    band_high: float = 30.0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Push an impulse through each filter; plot magnitude vs frequency."""
    OUT = Path(CONFIG.SDIR_RUN)

    impulse = torch.zeros(1, 1, seq_len)
    impulse[0, 0, seq_len // 2] = 1.0

    bands = torch.tensor([[band_low, band_high]], dtype=torch.float32)
    filters = {
        f"LowPass(<{cutoff:.0f} Hz)": scitex_nn.LowPassFilter(cutoff, fs, seq_len),
        f"HighPass(>{cutoff:.0f} Hz)": scitex_nn.HighPassFilter(cutoff, fs, seq_len),
        f"BandPass({band_low:.0f}-{band_high:.0f} Hz)": scitex_nn.BandPassFilter(
            bands, fs, seq_len
        ),
        f"BandStop({band_low:.0f}-{band_high:.0f} Hz)": scitex_nn.BandStopFilter(
            bands, fs, seq_len
        ),
    }

    freqs = np.fft.rfftfreq(seq_len, d=1.0 / fs)
    responses = {}
    for name, filt in filters.items():
        y = filt(impulse).detach().numpy().reshape(-1)
        # Peel any trailing length mismatch from the per-filter output.
        y = y[:seq_len]
        mag = np.abs(np.fft.rfft(y))
        mag = mag / mag.max()
        responses[name] = mag
        passband_max_hz = freqs[np.argmax(mag)]
        logger.info(f"{name:>26s}  peak at {passband_max_hz:6.2f} Hz")

    fig, ax = plt.subplots(axes_width_mm=160, axes_height_mm=70)
    for name, mag in responses.items():
        ax.plot(freqs, mag, lw=1.2, label=name)
    ax.axvline(cutoff, color="grey", ls=":", lw=0.6)
    ax.axvline(band_low, color="grey", ls=":", lw=0.6)
    ax.axvline(band_high, color="grey", ls=":", lw=0.6)
    ax.set_xlim(0, min(fs / 2, 80))
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.set_xyt(
        "frequency [Hz]",
        "magnitude (normalized)",
        f"scitex_nn filter bank — fs={fs} Hz, seq_len={seq_len}",
    )

    stx.io.save(fig, OUT / "filter_bank.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

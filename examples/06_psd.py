#!/usr/bin/env python3
"""PSD — power spectral density via FFT.

``scitex_nn.PSD`` is a one-line wrapper around ``torch.fft.rfft``: it
returns ``(power, freqs)`` for a real-valued signal. The kernel is
batched and runs on GPU; gradients flow through.

This example contrasts the PSD of a 10 Hz sine, a 10+25 Hz two-tone,
and pink (1/f) noise. The reference is ``scipy.signal.welch`` with
``nperseg = seq_len`` and ``noverlap = 0`` (one Hann window — the
simplest periodogram). The two curves overlap to noise-floor.

Run:
    python 06_psd.py
    python 06_psd.py --fs 1000 --seq-len 4096
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch
from scipy.signal import welch

import scitex_nn


def _pink(n, rng):
    """Approximate 1/f noise: white noise filtered by 1/√f in FFT domain."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size)
    f[0] = 1
    spec /= np.sqrt(f)
    return np.fft.irfft(spec, n=n)


@stx.session
def main(
    fs: int = 500,
    seq_len: int = 2048,
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """PSD of three signals; compare scitex_nn.PSD vs scipy.signal.welch."""
    OUT = Path(CONFIG.SDIR_RUN)
    rng = np.random.default_rng(seed)

    t = np.arange(seq_len) / fs
    signals = {
        "10 Hz sine": np.sin(2 * np.pi * 10 * t),
        "10+25 Hz two-tone": (
            np.sin(2 * np.pi * 10 * t) + 0.6 * np.sin(2 * np.pi * 25 * t)
        ),
        "pink noise (1/f)": _pink(seq_len, rng),
    }

    psd_layer = scitex_nn.PSD(sample_rate=fs)

    fig, axes = plt.subplots(
        nrows=len(signals),
        ncols=1,
        sharex=True,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=42,
    )
    for ax, (name, sig) in zip(axes, signals.items()):
        x = torch.from_numpy(sig).float()
        psd_stx, freqs_stx = psd_layer(x)
        f_ref, p_ref = welch(sig, fs=fs, nperseg=seq_len, noverlap=0)
        ax.plot(freqs_stx.numpy(), psd_stx.numpy(), lw=1.4, label="scitex_nn.PSD")
        ax.plot(f_ref, p_ref, "--", lw=1.0, label="scipy welch")
        ax.set_xlim(0, 60)
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.set_xyt(None, "power", name)
        peak = freqs_stx.numpy()[np.argmax(psd_stx.numpy())]
        logger.info(f"{name:>22s}  peak at {peak:6.2f} Hz")
    axes[-1].set_xyt("frequency [Hz]", "power", list(signals.keys())[-1])

    stx.io.save(fig, OUT / "psd.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

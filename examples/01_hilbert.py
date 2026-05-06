#!/usr/bin/env python3
"""Hilbert transform — analytic signal envelope vs scipy reference.

Demonstrates that ``scitex_nn.Hilbert`` produces the same envelope as
``scipy.signal.hilbert`` to float-precision (max|diff| ≈ 1e-7) on a
clean 10 Hz sine, after the steepness-50 sigmoid was replaced with
the canonical hard-step analytic-signal mask.

Run:
    python 01_hilbert.py
    python 01_hilbert.py --freq 40 --fs 1024 --t-sec 2
    python 01_hilbert.py --help
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch
from scipy.signal import hilbert as scipy_hilbert

import scitex_nn


@stx.session
def main(
    freq: float = 10.0,
    fs: int = 512,
    t_sec: float = 1.0,
    CONFIG=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Compare scitex_nn.Hilbert envelope to scipy on a pure sine."""
    OUT = Path(CONFIG.SDIR_RUN)
    n = int(fs * t_sec)

    t = np.linspace(0.0, t_sec, n, endpoint=False)
    sig_np = np.sin(2 * np.pi * freq * t).astype(np.float32)

    H = scitex_nn.Hilbert(seq_len=n, dim=-1)
    out = H(torch.from_numpy(sig_np))
    pha_stx = out[..., 0].numpy()
    amp_stx = out[..., 1].numpy()

    z_scipy = scipy_hilbert(sig_np)
    amp_scipy = np.abs(z_scipy)
    pha_scipy = np.angle(z_scipy)

    diff = float(np.abs(amp_stx - amp_scipy).max())
    flatness_stx = float(np.std(amp_stx) / np.mean(amp_stx))
    flatness_scipy = float(np.std(amp_scipy) / np.mean(amp_scipy))
    logger.info(f"max|amp_scitex_nn - amp_scipy| = {diff:.2e}")
    logger.info(f"envelope flatness scitex_nn = {flatness_stx:.2e}")
    logger.info(f"envelope flatness scipy     = {flatness_scipy:.2e}")
    assert diff < 1e-5, "Hilbert envelope drifted from scipy reference"

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10, 8))
    axes[0].plot(t, sig_np, color="black")
    axes[0].set_xyt(None, "signal", f"{freq} Hz sine, fs={fs}")

    axes[1].plot(t, amp_scipy, label="scipy.signal.hilbert", lw=2)
    axes[1].plot(t, amp_stx, "--", label="scitex_nn.Hilbert", lw=2)
    axes[1].axhline(1.0, color="grey", ls=":", label="expected (=1.0)")
    axes[1].set_ylim(0.0, 1.5)
    axes[1].legend(loc="lower right")
    axes[1].set_xyt(None, "envelope", "Amplitude envelope (both lines flat at 1.0)")

    axes[2].plot(t, pha_scipy, label="scipy phase", lw=1)
    axes[2].plot(t, pha_stx, "--", label="scitex_nn phase", lw=1)
    axes[2].set_xyt("time [s]", "phase [rad]", "Phase (overlapping)")
    axes[2].legend(loc="lower right")

    stx.io.save(fig, OUT / "hilbert_vs_scipy.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

#!/usr/bin/env python3
"""Hilbert transform — analytic signal envelope vs scipy reference.

Reproduces the canonical example from the scipy docs:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html

A 20 → 100 Hz linear chirp ridden by ``1 + 0.5·sin(2π·3·t)``, hilbert-
transformed both by ``scitex_nn.Hilbert`` and ``scipy.signal.hilbert``.
The two envelopes overlap to float-32 noise (max|diff| ≈ 1e-7) after
the day-1 sigmoid soft-step was replaced with the canonical hard-step
analytic-signal mask.

Run:
    python 01_hilbert.py
    python 01_hilbert.py --duration 2 --fs 800 --f0 30 --f1 200
    python 01_hilbert.py --help
"""

from pathlib import Path

import numpy as np
import scitex as stx
import torch
from scipy.signal import chirp
from scipy.signal import hilbert as scipy_hilbert

import scitex_nn


@stx.session
def main(
    duration: float = 1.0,
    fs: int = 400,
    f0: float = 20.0,
    f1: float = 100.0,
    am_freq: float = 3.0,
    am_depth: float = 0.5,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Compare ``scitex_nn.Hilbert`` to ``scipy.signal.hilbert`` on the
    canonical scipy-doc chirp + amplitude-modulation example."""
    OUT = Path(CONFIG.SDIR_RUN)
    n = int(fs * duration)
    t = np.arange(n) / fs

    # Build the scipy-doc demo signal verbatim:
    #   signal = chirp(t, f0, t[-1], f1)
    #   signal *= (1.0 + am_depth * sin(2π·am_freq·t))
    base = chirp(t, f0, t[-1], f1)
    am = 1.0 + am_depth * np.sin(2 * np.pi * am_freq * t)
    sig_np = (base * am).astype(np.float32)
    expected_envelope = am.astype(np.float32)

    H = scitex_nn.Hilbert(seq_len=n, dim=-1)
    out = H(torch.from_numpy(sig_np))
    pha_stx = out[..., 0].numpy()
    amp_stx = out[..., 1].numpy()

    z_scipy = scipy_hilbert(sig_np)
    amp_scipy = np.abs(z_scipy)
    pha_scipy = np.angle(z_scipy)

    diff = float(np.abs(amp_stx - amp_scipy).max())
    rel_diff = diff / float(np.mean(amp_scipy))
    logger.info(f"max|amp_scitex_nn - amp_scipy| = {diff:.2e}")
    logger.info(f"relative max diff             = {rel_diff:.2e}")
    assert rel_diff < 1e-5, "Hilbert envelope drifted from scipy reference"

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        sharey=False,
        axes_width_mm=160,
        axes_height_mm=40,
    )
    # Align signal- and envelope-panel y-axes so the envelope is read off
    # the same scale as the raw signal.
    sig_ymax = float(max(np.abs(sig_np).max(), expected_envelope.max())) * 1.1
    axes[0].plot(t, sig_np, color="black", label="signal")
    axes[0].plot(
        t,
        expected_envelope,
        color="grey",
        ls=":",
        label=f"input envelope: 1 + {am_depth}·sin(2π·{am_freq:.0f}·t)",
    )
    axes[0].set_ylim(-sig_ymax, sig_ymax)
    axes[0].legend(loc="upper right")
    axes[0].set_xyt(
        None,
        "signal",
        f"Chirp {f0:.0f}→{f1:.0f} Hz over {duration}s, fs={fs} (scipy-docs example)",
    )

    axes[1].plot(t, amp_scipy, label="scipy.signal.hilbert", lw=2)
    axes[1].plot(t, amp_stx, "--", label="scitex_nn.Hilbert", lw=2)
    axes[1].plot(t, expected_envelope, color="grey", ls=":", label="input envelope")
    axes[1].set_ylim(-sig_ymax, sig_ymax)
    axes[1].legend(loc="lower right")
    axes[1].set_xyt(
        None,
        "envelope",
        f"Recovered envelope (max|diff vs scipy| = {diff:.2e})",
    )

    axes[2].plot(t, pha_scipy, label="scipy phase", lw=1)
    axes[2].plot(t, pha_stx, "--", label="scitex_nn phase", lw=1)
    axes[2].set_xyt("time [s]", "phase [rad]", "Phase (overlapping)")
    axes[2].legend(loc="lower right")

    stx.io.save(fig, OUT / "hilbert_vs_scipy.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

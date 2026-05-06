#!/usr/bin/env python3
"""GaussianFilter — differentiable temporal smoothing.

``GaussianFilter`` convolves the last axis with a normalized Gaussian
kernel. The kernel size is ``6·sigma`` (±3 std). It is registered as a
buffer (no learnable params) and runs end-to-end on GPU.

This example smooths a noisy sine wave at three sigmas and overlays
them on the clean reference. Larger ``sigma`` ⇒ stronger smoothing ⇒
more attenuation of high-frequency noise but also of the signal edge.

Run:
    python 04_gaussian_filter.py
    python 04_gaussian_filter.py --sigmas 4 12 30
"""

from pathlib import Path

import scitex as stx
import torch

import scitex_nn


@stx.session
def main(
    seq_len: int = 512,
    fs: int = 256,
    f0: float = 4.0,
    noise_std: float = 0.5,
    sigmas: tuple = (4, 12, 30),
    seed: int = 0,
    CONFIG=stx.session.INJECTED,
    COLORS=stx.session.INJECTED,
    rngg=stx.session.INJECTED,
    plt=stx.session.INJECTED,
    logger=stx.session.INJECTED,
):
    """Smooth a noisy sine at multiple sigmas; overlay against the clean signal."""
    OUT = Path(CONFIG.SDIR_RUN)
    torch.manual_seed(seed)

    t = torch.arange(seq_len) / fs
    clean = torch.sin(2 * torch.pi * f0 * t)
    noisy = clean + noise_std * torch.randn_like(clean)
    x = noisy.view(1, 1, -1)

    smoothed = {}
    for sigma in sigmas:
        filt = scitex_nn.GaussianFilter(sigma=int(sigma))
        y = filt(x).squeeze().detach().numpy()
        smoothed[int(sigma)] = y
        rmse = float(((y - clean.numpy()) ** 2).mean() ** 0.5)
        logger.info(f"sigma={sigma:>3d}  RMSE vs clean = {rmse:.3f}")

    fig, axes = plt.subplots(
        nrows=len(sigmas) + 1,
        ncols=1,
        sharex=True,
        sharey=True,
        axes_width_mm=160,
        axes_height_mm=28,
    )
    axes[0].plot(t.numpy(), noisy.numpy(), color="black", lw=0.5, label="noisy input")
    axes[0].plot(
        t.numpy(), clean.numpy(), color="grey", ls=":", label="clean reference"
    )
    axes[0].legend(loc="upper right")
    axes[0].set_xyt(
        None, "amplitude", f"Input: {f0:.0f} Hz sine + noise(σ={noise_std})"
    )

    for ax, sigma in zip(axes[1:], sigmas):
        ax.plot(t.numpy(), smoothed[int(sigma)], lw=1.2, label=f"σ={sigma}")
        ax.plot(t.numpy(), clean.numpy(), color="grey", ls=":", label="clean")
        ax.legend(loc="upper right")
        ax.set_xyt(None, "amplitude", f"GaussianFilter(sigma={sigma})")
    axes[-1].set_xyt("time [s]", "amplitude", f"GaussianFilter(sigma={sigmas[-1]})")

    stx.io.save(fig, OUT / "gaussian_filter.png")
    return 0


if __name__ == "__main__":
    main()

# EOF

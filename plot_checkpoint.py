"""
plot_checkpoint.py — render an FWI checkpoint .npz from
`Results_strain_fwi/<backend>/stageN/epoch_NNNN.npz`.

3-panel layout (top to bottom):
    Vs (current iterate)
    Vs_true   (reference)
    Vs - Vs_true  (error)

Usage:
    python plot_checkpoint.py
    python plot_checkpoint.py --npz <path> --out <png>
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


CMAP_VEL = "jet"
CMAP_DIFF = "seismic"
DIFF_GAIN = 0.4
DPI = 140
FIGSIZE = (14, 9)
DX = DZ = 20.0


def render(npz_path, out_path, true_vs_path):
    d = np.load(npz_path)
    Vs = d["Vs"]
    it = int(d["iter"])
    loss = float(d["loss"])
    Vs_true = np.load(true_vs_path)

    nz, nx = Vs.shape
    extent = [0, nx * DX / 1000, nz * DZ / 1000, 0]

    v_lo = float(min(Vs.min(), Vs_true.min()))
    v_hi = float(max(Vs.max(), Vs_true.max()))

    diff = Vs - Vs_true
    dmax = float(np.quantile(np.abs(diff), 0.99) * DIFF_GAIN)

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, constrained_layout=True)

    im0 = axes[0].imshow(Vs, cmap=CMAP_VEL, vmin=v_lo, vmax=v_hi,
                         extent=extent, aspect="auto")
    axes[0].set_title(f"Vs at iter {it}   loss = {loss:.4e}")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Vs_true, cmap=CMAP_VEL, vmin=v_lo, vmax=v_hi,
                         extent=extent, aspect="auto")
    axes[1].set_title("Vs_true")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap=CMAP_DIFF, vmin=-dmax, vmax=dmax,
                         extent=extent, aspect="auto")
    axes[2].set_title("Vs − Vs_true")
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz",
                   default="Results_strain_fwi/adjoint/stage1/epoch_0010.npz")
    p.add_argument("--true",
                   default="marmousi_models/vs_true.npy")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or args.npz.replace(".npz", ".png")
    render(args.npz, out, args.true)


if __name__ == "__main__":
    main()

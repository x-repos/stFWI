"""
plot_fwi_results.py — compare AD vs adjoint inverted Vs.

Loads:
    Results_strain_fwi/ad/final.npz
    Results_strain_fwi/adjoint/final.npz

Renders a 3×2 figure:
    row 1: Vs_true                       | Vs_init
    row 2: Vs_inv (AD)                   | Vs_inv (adjoint)
    row 3: Vs_inv (AD) − Vs_true         | Vs_inv (adjoint) − Vs_true

Same colour limits within each row so the panels are directly comparable.
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ── Knobs ────────────────────────────────────────────────────────────────
CMAP_VEL = "jet"
CMAP_DIFF = "seismic"
DIFF_GAIN = 0.4         # smaller = more saturated diff panels (q=0.99 baseline)
OUT_PNG  = "Results_strain_fwi/fwi_compare.png"
DPI      = 140
FIGSIZE  = (15, 10)


def load(path):
    d = np.load(path)
    return dict(
        Vs_inv=d["Vs_inv"],
        Vs_init=d["Vs_init"],
        Vs_true=d["Vs_true"],
        dx=float(d["dx"]),
        dz=float(d["dz"]),
        src_x=np.atleast_1d(d["src_x"]).astype(float),
        src_z=np.atleast_1d(d["src_z"]).astype(float),
        rec_x=d["rec_x"],
        rec_z=d["rec_z"],
        backend=str(d["backend"]),
    )


def render(ad_npz, adj_npz, out_path):
    A = load(ad_npz)
    B = load(adj_npz)
    assert A["Vs_true"].shape == B["Vs_true"].shape

    nz, nx = A["Vs_true"].shape
    dx, dz = A["dx"], A["dz"]
    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    # Common Vs colour range (so all velocity panels share the bar)
    v_lo = float(min(A["Vs_true"].min(), A["Vs_init"].min(),
                     A["Vs_inv"].min(),  B["Vs_inv"].min()))
    v_hi = float(max(A["Vs_true"].max(), A["Vs_init"].max(),
                     A["Vs_inv"].max(),  B["Vs_inv"].max()))

    # Diff colour range
    diff_AD  = A["Vs_inv"] - A["Vs_true"]
    diff_adj = B["Vs_inv"] - B["Vs_true"]
    dmax = max(np.quantile(np.abs(diff_AD),  0.99),
               np.quantile(np.abs(diff_adj), 0.99)) * DIFF_GAIN

    fig, axes = plt.subplots(3, 2, figsize=FIGSIZE, constrained_layout=True)

    # Row 1: true | init
    im00 = axes[0, 0].imshow(A["Vs_true"], cmap=CMAP_VEL, vmin=v_lo, vmax=v_hi,
                              extent=extent, aspect="auto")
    axes[0, 0].set_title("Vs_true")
    fig.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(A["Vs_init"], cmap=CMAP_VEL, vmin=v_lo, vmax=v_hi,
                              extent=extent, aspect="auto")
    axes[0, 1].set_title("Vs_init")
    fig.colorbar(im01, ax=axes[0, 1])

    # Row 2: inverted (AD) | inverted (adjoint)
    im10 = axes[1, 0].imshow(A["Vs_inv"], cmap=CMAP_VEL, vmin=v_lo, vmax=v_hi,
                              extent=extent, aspect="auto")
    axes[1, 0].set_title(f"Vs_inv  (backend = '{A['backend']}')")
    fig.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(B["Vs_inv"], cmap=CMAP_VEL, vmin=v_lo, vmax=v_hi,
                              extent=extent, aspect="auto")
    axes[1, 1].set_title(f"Vs_inv  (backend = '{B['backend']}')")
    fig.colorbar(im11, ax=axes[1, 1])

    # Row 3: diff (inv − true)
    im20 = axes[2, 0].imshow(diff_AD, cmap=CMAP_DIFF, vmin=-dmax, vmax=dmax,
                              extent=extent, aspect="auto")
    axes[2, 0].set_title("Vs_inv − Vs_true   (AD)")
    fig.colorbar(im20, ax=axes[2, 0])

    im21 = axes[2, 1].imshow(diff_adj, cmap=CMAP_DIFF, vmin=-dmax, vmax=dmax,
                              extent=extent, aspect="auto")
    axes[2, 1].set_title("Vs_inv − Vs_true   (adjoint)")
    fig.colorbar(im21, ax=axes[2, 1])

    # Markers on every panel
    for ax in axes.flat:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.scatter(A["src_x"] * dx / 1000, A["src_z"] * dz / 1000,
                   marker="*", s=80, c="yellow", edgecolors="black", zorder=5)
        ax.scatter(np.asarray(A["rec_x"]) * dx / 1000,
                   np.asarray(A["rec_z"]) * dz / 1000,
                   marker="v", s=8, c="lime", edgecolors="black",
                   linewidths=0.4, zorder=5)

    fig.suptitle("FWI Vs inversion: AD vs adjoint backend", fontsize=12)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ad",  default="Results_strain_fwi/ad/final.npz")
    p.add_argument("--adj", default="Results_strain_fwi/adjoint/final.npz")
    p.add_argument("--out", default=OUT_PNG)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    render(args.ad, args.adj, args.out)


if __name__ == "__main__":
    main()

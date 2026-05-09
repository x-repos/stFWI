"""
plot_first_gradient.py — render the first-FWI gradients (AD + adjoint)
produced by compute_first_gradient.py.

4 stacked panels:
    1) ∂L/∂Vs  (AD,        with stage-1 preconditioning applied)
    2) ∂L/∂Vs  (adjoint+imaging, same preconditioning)
    3) Illumination H_inv used in steps 1 & 2
    4) Vs_init − Vs_true  (target update direction reference)
"""
from __future__ import annotations

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


# ── Plotting knobs ───────────────────────────────────────────────────────
GAIN_GRAD  = 0.05    # both gradient panels share this gain
GAIN_MODEL = 0.15
Q_GRAD  = 0.99
Q_MODEL = 0.995
CMAP_DIVERGING = "seismic"
CMAP_ILLUM     = "jet"

OUT_PNG = "first_gradient.png"
DPI     = 140
FIGSIZE = (12, 14)

SRC_KW = dict(marker='*', s=80, c='yellow', edgecolors='black', zorder=5)
REC_KW = dict(marker='v', s=8,  c='lime',   edgecolors='black', linewidths=0.4, zorder=5)


def sym_clip(arr, q=0.99, gain=1.0):
    return float(np.quantile(np.abs(arr), q) * gain)


def render(npz_path, out_path):
    d = np.load(npz_path, allow_pickle=True)

    def _arr(key):
        return np.asarray(d[key])

    def _scalar(key, cast=float):
        x = np.asarray(d[key])
        return cast(x.item() if x.shape == () else x.ravel()[0])

    g_AD     = _arr("g_Vs_AD")
    g_HC     = _arr("g_Vs_HC") if "g_Vs_HC" in d.files else None
    # Prefer eprecond3 (used by adjoint backend if present); otherwise
    # fall back to H_inv (used by AD).
    if "illumination_eprecond3" in d.files and g_HC is not None:
        illum = _arr("illumination_eprecond3")
    elif "illumination" in d.files:
        illum = _arr("illumination")
    else:
        illum = None
    Vs_init  = _arr("Vs_init")
    Vs_true  = _arr("Vs_true")
    src_x    = np.atleast_1d(_arr("src_x")).astype(float).ravel()
    src_z    = np.atleast_1d(_arr("src_z")).astype(float).ravel()
    rec_x    = _arr("rec_x")
    rec_z    = _arr("rec_z")
    dx       = _scalar("dx")
    dz       = _scalar("dz")
    nz       = _scalar("nz", int)
    nx       = _scalar("nx", int)
    n_shots  = _scalar("n_shots", int)
    f_c      = _scalar("f_c")
    cutoff   = _scalar("cutoff")
    loss_AD  = _scalar("loss_AD")
    loss_HC  = _scalar("loss_HC") if "loss_HC" in d.files else None

    extent = [0, nx * dx / 1000, nz * dz / 1000, 0]

    has_hc = g_HC is not None
    n_panels = 4 if has_hc else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=FIGSIZE, constrained_layout=True)

    # Shared colour limit so AD and adjoint are directly comparable
    if has_hc:
        vmax_grad = max(sym_clip(g_AD, Q_GRAD, GAIN_GRAD),
                        sym_clip(g_HC, Q_GRAD, GAIN_GRAD))
    else:
        vmax_grad = sym_clip(g_AD, Q_GRAD, GAIN_GRAD)

    # Panel 1: AD gradient
    im0 = axes[0].imshow(g_AD, cmap=CMAP_DIVERGING, vmin=-vmax_grad, vmax=vmax_grad,
                         extent=extent, aspect="auto")
    axes[0].set_title(f"∂L/∂Vs   AD   "
                      f"(n_shots={n_shots}, fc={f_c:g} Hz, cutoff={cutoff:g} Hz, "
                      f"loss={loss_AD:.4e})")
    fig.colorbar(im0, ax=axes[0])
    next_ax = 1

    # Panel 2: adjoint+imaging gradient
    if has_hc:
        im1 = axes[next_ax].imshow(g_HC, cmap=CMAP_DIVERGING,
                                   vmin=-vmax_grad, vmax=vmax_grad,
                                   extent=extent, aspect="auto")
        axes[next_ax].set_title(f"∂L/∂Vs   adjoint + imaging   "
                                f"(loss={loss_HC:.4e})")
        fig.colorbar(im1, ax=axes[next_ax])
        next_ax += 1

    # Panel: illumination (skip if not available)
    if illum is not None and illum.ndim >= 2:
        if illum.shape != (nz, nx):
            p_off = (illum.shape[1] - nx) // 2
            illum_phys = illum[:nz, p_off:p_off + nx]
        else:
            illum_phys = illum
        im2 = axes[next_ax].imshow(illum_phys, cmap=CMAP_ILLUM, vmin=0, vmax=1,
                                   extent=extent, aspect="auto")
        title = "Illumination  H_inv  (post-norm to [0,1])"
        if "illumination_eprecond3" in d.files and g_HC is not None:
            title = ("Illumination  eprecond3 = (1/We) normalised   "
                     "[Plessix-Mulder, used by adjoint]")
        axes[next_ax].set_title(title)
        fig.colorbar(im2, ax=axes[next_ax])
        next_ax += 1

    # Panel: model-error reference
    dvs = Vs_init - Vs_true
    vmax = sym_clip(dvs, q=Q_MODEL, gain=GAIN_MODEL)
    im3 = axes[next_ax].imshow(dvs, cmap=CMAP_DIVERGING, vmin=-vmax, vmax=vmax,
                               extent=extent, aspect="auto")
    axes[next_ax].set_title("Vs_init − Vs_true   (target update direction)")
    fig.colorbar(im3, ax=axes[next_ax])

    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("z [km]")
        ax.scatter(src_x * dx / 1000, src_z * dz / 1000, **SRC_KW)
        ax.scatter(np.asarray(rec_x) * dx / 1000,
                   np.asarray(rec_z) * dz / 1000, **REC_KW)

    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="Results_first_gradient/first_gradient.npz")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.npz) or ".", OUT_PNG)
    render(args.npz, out)


if __name__ == "__main__":
    main()

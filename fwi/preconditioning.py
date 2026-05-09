"""
fwi/preconditioning.py — gradient preconditioners for FWI.

Ported from xFWI's `xfwi/preconditioning.py` and adapted to the
Strain_FWI-main array convention (z-major, shape (nz, nx)) and free-surface
flag.

Contents
--------
- `eprecond1(Ws, Wr, eps=5e-7)` — Shin et al. (2001):
      We = sqrt(Ws · Wr) + eps·max(...)
  Requires both forward (Ws) and adjoint (Wr) wavefield energies.

- `eprecond3(Ws, rec_x, rec_z, dx, dz, free_surface, pad, eps=5e-7)` —
  Plessix & Mulder (2004) Hessian approximation:
      We = sqrt(Ws) · [ arcsinh((line_max-c)/y) - arcsinh((line_min-c)/y) ]
  Forward energy only. Auto-detects horizontal vs vertical receiver line.

- `taper_grad(shape, dz, gradt1, gradt2, exp_taper=2.0, gradb1=None, gradb2=None)`
  DENISE-style depth-dependent gradient taper (zeroes water column,
  optional bottom roll-off).

These are *post-AD* preconditioners — apply them to the gradient AFTER
`jax.value_and_grad`, BEFORE handing to the optimizer. Pure NumPy keeps
them out of the JIT trace; the work is one matmul-equivalent per iter,
negligible vs forward modelling.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Wavefield-energy preconditioners
# ─────────────────────────────────────────────────────────────────────────────

def eprecond1(Ws: np.ndarray, Wr: np.ndarray, eps: float = 5e-7) -> np.ndarray:
    """
    Shin et al. (2001) energy preconditioner.

        We = sqrt(Ws · Wr) + eps · max(sqrt(Ws · Wr))

    Apply via `grad /= We` (or multiply by 1/We).

    Parameters
    ----------
    Ws, Wr : (nz, nx) arrays
        Forward and adjoint wavefield energies (sum of squared field over
        all timesteps for whichever component is recording — typically
        sum of vx² + vz² + ε·² for the chosen component).
    eps : float
        Regularization. Default 5e-7 matches xFWI / DENISE.

    Returns
    -------
    We : (nz, nx) array, same dtype as Ws.
    """
    Ws = np.asarray(Ws)
    Wr = np.asarray(Wr)
    We = np.sqrt(Ws * Wr)
    We = We + eps * float(np.max(We))
    return We


def eprecond3(Ws: np.ndarray,
              rec_x: np.ndarray, rec_z: np.ndarray,
              dx: float, dz: float,
              free_surface: bool, pad: int,
              eps: float = 5e-7) -> np.ndarray:
    """
    Plessix & Mulder (2004) Hessian-approximation preconditioner.

    Uses ONLY the forward energy `Ws` plus an analytical 1/r integral
    over the receiver line — no adjoint energy needed (memory-efficient).

        We = sqrt(Ws) · [ arcsinh((line_max - c)/y) - arcsinh((line_min - c)/y) ]

    where `c` is the coordinate along the receiver line and `y` the
    perpendicular distance from each grid point to that line (clamped to
    one grid spacing to avoid division by zero).

    Auto-detects whether the receiver line is horizontal (varying x,
    constant z) or vertical (varying z, constant x).

    Parameters
    ----------
    Ws : (nz, nx)
        Forward wavefield energy on the FULL padded grid (including PML
        and any free-surface offset). Caller is responsible for matching
        Ws's shape to the conventions below.
    rec_x, rec_z : (nrec,) int
        Physical-domain (un-padded) receiver indices.
    dx, dz : float
        Grid spacing.
    free_surface : bool
        If True, the top of the padded grid IS the surface (no top-PML).
        Affects the z-coordinate origin in the padded grid.
    pad : int
        PML padding cells.
    eps : float
        Regularization. Default 5e-7.

    Returns
    -------
    We : (nz, nx) on the same padded grid as Ws.
    """
    Ws = np.asarray(Ws)
    nz_t, nx_t = Ws.shape
    z_start = 0 if free_surface else pad

    # Physical-coordinate grid for the full padded array (origin = 0,0 at
    # the (z_start, pad) cell, mirroring forward.py's interior placement).
    coord_z = (np.arange(nz_t) - z_start) * dz   # m
    coord_x = (np.arange(nx_t) - pad)     * dx   # m
    cz, cx = np.meshgrid(coord_z, coord_x, indexing="ij")  # (nz, nx)

    # Detect line orientation by which receiver coordinate varies more.
    if np.std(rec_x) >= np.std(rec_z):
        # Horizontal receiver line (typical surface acquisition):
        #   line varies in x, constant z.
        line_min = float(np.min(rec_x) * dx)
        line_max = float(np.max(rec_x) * dx)
        perp_pos = float(rec_z[0] * dz)
        c_perp, c_line, sp = cz, cx, dz
    else:
        # Vertical receiver line:
        line_min = float(np.min(rec_z) * dz)
        line_max = float(np.max(rec_z) * dz)
        perp_pos = float(rec_x[0] * dx)
        c_perp, c_line, sp = cx, cz, dx

    # Perpendicular distance, clamped to one grid spacing to avoid div/0.
    y = np.maximum(np.abs(c_perp - perp_pos), sp)

    We = np.sqrt(Ws) * (np.arcsinh((line_max - c_line) / y)
                        - np.arcsinh((line_min - c_line) / y))
    We = np.abs(We)
    We = We + eps * float(np.max(We))
    return We


# ─────────────────────────────────────────────────────────────────────────────
# DENISE-style depth-dependent gradient taper
# ─────────────────────────────────────────────────────────────────────────────

def taper_grad(shape: tuple,
               dz: float,
               gradt1: int, gradt2: int,
               exp_taper: float = 2.0,
               gradb1: int | None = None,
               gradb2: int | None = None) -> np.ndarray:
    """
    Depth-dependent gradient taper (DENISE convention).

        depth ∈ [0, gradt1)        : taper = 0  (water column / above target)
        depth ∈ [gradt1, gradt2]   : Gaussian ramp from 0 toward 1
        depth >  gradt2            : taper ∝ depth^exp_taper (depth boost)
        optional bottom roll:
        depth ∈ [gradb1, gradb2]   : Gaussian ramp DOWN to 0
        depth >  gradb2            : taper = 0

    Note: shape convention is (nz, nx) here — opposite to xFWI which uses
    (nx, nz). The taper is broadcast across nx.

    Returns
    -------
    taper : (nz, nx)
    """
    nz, nx = shape
    t = np.ones(nz)
    for j in range(nz):
        if j < gradt1:
            t[j] = 0.0
        elif j <= gradt2:
            t[j] = np.exp(-float(gradt2 - j) ** 2)
        else:
            t[j] = (j * dz) ** exp_taper

    if gradb1 is not None and gradb2 is not None:
        denom = max(float(gradb2 - gradb1), 1.0)
        for j in range(nz):
            if j > gradb2:
                t[j] = 0.0
            elif j >= gradb1:
                t[j] *= np.exp(-float(j - gradb1) ** 2 / denom ** 2 * 4.0)

    return np.broadcast_to(t[:, None], (nz, nx)).copy()

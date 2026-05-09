"""
fwi/projections.py — feasibility projections for FWI inversion variables.

Two physical constraints commonly enforced inside a bound-constrained
optimiser (or before/after each iteration):

  K>0  (bulk modulus positive)
       ⇔   λ + (2/3)μ > 0
       ⇔   Vp/Vs ≥ 2/√3 ≈ 1.1547   (continuum, no margin)
       ⇔   Vp/Vs ≥ (1+K_margin)·2/√3   (with a small safety margin r)

  CFL  (numerical stability of the explicit FD scheme)
       Vp_max ·  Δt  ≤  C · Δx
       ⇒   Vp ≤ vp_cap  pointwise, where  vp_cap = C·Δx/Δt.

Helpers below take a model and return a **projected** copy that
satisfies the constraints. They are deliberately simple (no
optimisation), cheap, and pure-NumPy — call them between iterations.

Convention: arrays use `(nz, nx)` layout, matching `forward.py`.

Ported from xfwi/optim.py's `_project_to_feasible*` helpers, restricted
to **velocity-space (Vs, Vp, ρ)** which is the parameterisation used by
Strain_FWI-main's forward kernel and inversion.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Stability / physical constants
# ─────────────────────────────────────────────────────────────────────────────

R_CONTINUUM = 2.0 / np.sqrt(3.0)        # ≈ 1.1547  (vp/vs at K = 0)


def k_safe_ratio(K_margin: float = 0.04) -> float:
    """
    Vp/Vs lower bound used in the K>0 projection.

        r = (1 + K_margin) · 2/√3

    K_margin = 0   → on the physical cliff (r ≈ 1.155)
    K_margin = 0.04 → ≈ 1.20  (4% safety, xFWI default)
    """
    return (1.0 + float(K_margin)) * R_CONTINUUM


def cfl_vp_cap(dx: float, dz: float, dt: float, cfl_coeff: float) -> float:
    """
    Maximum allowed Vp from CFL:

        vp_cap = cfl_coeff · min(Δx, Δz) / Δt

    `cfl_coeff` is the scheme-specific stability constant (the same `S`
    used in `Main_ex_Vs.py` to derive `dt`). For an order-N staggered FD
    in 2D it's the coefficient that makes the von-Neumann condition tight.
    """
    return float(cfl_coeff) * float(min(dx, dz)) / float(dt)


# ─────────────────────────────────────────────────────────────────────────────
# Velocity-space projection
# ─────────────────────────────────────────────────────────────────────────────

def project_velocity(Vs: np.ndarray,
                     Vp: np.ndarray,
                     rho: np.ndarray,
                     K_margin: float = 0.04,
                     vp_cap: float | None = None,
                     vs_floor: float = 1e-3,
                     rho_floor: float = 1e-3):
    """
    Project (Vs, Vp, ρ) to satisfy:
       1.  Vs ≤ Vp / r       (K-margin)        — clip Vs from above.
       2.  Vp ≤ vp_cap       (CFL ceiling)     — clip Vp from above.
       3.  Vs, ρ ≥ floor                        — avoid degenerate values.
       Note: when (2) tightens Vp, (1) is re-applied so Vs respects the
       new Vp.

    Argument order is (Vs, Vp, ρ) to match the rest of Strain_FWI-main
    (e.g. `forward_jax(Vs, Vp, rho, ...)`).

    Parameters
    ----------
    Vs, Vp, rho : (nz, nx) arrays
    K_margin : float
        Safety margin above K=0 cliff (xFWI default 0.04).
    vp_cap : float, optional
        CFL cap from `cfl_vp_cap(...)`. If None, no Vp clipping.
    vs_floor, rho_floor : float
        Minimum allowed values.

    Returns
    -------
    (Vs_p, Vp_p, rho_p) : projected copies (np.ndarray).
    """
    vp_p = np.asarray(Vp, dtype=np.float64).copy()
    vs_p = np.asarray(Vs, dtype=np.float64).copy()
    rho_p = np.asarray(rho, dtype=np.float64).copy()

    r = k_safe_ratio(K_margin)

    # CFL on Vp first — that determines the actual ceiling Vs has to live under
    if vp_cap is not None:
        np.minimum(vp_p, float(vp_cap), out=vp_p)

    # K-margin: vs ≤ vp / r
    vs_cap = vp_p / r
    np.minimum(vs_p, vs_cap, out=vs_p)

    # Floors
    np.maximum(vs_p, float(vs_floor), out=vs_p)
    np.maximum(rho_p, float(rho_floor), out=rho_p)

    return vs_p, vp_p, rho_p

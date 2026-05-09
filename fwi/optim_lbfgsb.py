"""
fwi/optim_lbfgsb.py — multistage SciPy L-BFGS-B FWI loop.

Wraps `scipy.optimize.minimize(method='L-BFGS-B')` and dispatches the
gradient computation through `fwi.compute_gradient(backend=...)`. Adds
the production infrastructure layered around the bare gradient:

  • Multistage frequency bands  (per-stage Ricker fc, Butterworth cutoff,
    cosine end-taper, illumination ε, smoothing σ, source taper).
  • Per-block parameter scaling   `x_scaled = x / max(|x|)`   so all
    parameter blocks share a similar magnitude in L-BFGS-B's view.
  • Velocity-space feasibility projection  (K-margin + optional CFL cap)
    applied at every gradient evaluation BEFORE the forward.
  • Per-cell bounds in scaled space.
  • Plateau detection per stage   (last-N-evals vs prev-N-evals < tol).
  • Loss history + per-stage parameter checkpoints.

Returns the final  (Vs, Vp, ρ)  and a list of stage history dicts.

The function is parameterisation-agnostic on the gradient side — the
forward kernel takes velocities, AD/adjoint produce velocity gradients,
no Lamé/impedance conversion is involved.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.signal import butter

from .forward import ricker_jax, build_forward_fn
from .gradient import compute_gradient
from .projections import project_velocity, cfl_vp_cap
from .illumination import compute_illumination_for_stage
from .preconditioning import eprecond3


_KEY_TO_IDX = {"Vs": 0, "Vp": 1, "rho": 2}


def _compute_eprecond3_preconditioner(
    *, nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
    rec_x, rec_z, fd_order, component,
    Vs, Vp, rho, src_x, src_z, src_wavelet,
    free_surface, eps,
):
    """
    Build a [0, 1] gradient multiplier from the Plessix-Mulder (2004)
    eprecond3 formula, on the **physical** grid:

        Ws(x)  = Σ_shots Σ_t [u_s(x, t)]²              (forward energy)
        We(x)  = sqrt(Ws) · Δarcsinh-of-receiver-line   (eprecond3)
        prec(x) = (1 / We) / max(1 / We)                normalised to [0, 1]

    The returned `prec` is meant to be **multiplied** into the gradient
    inside `_postprocess`, the same way `H_inv` is — so it slots into
    `compute_gradient(..., illumination=prec, ...)` with no other code
    changes downstream.
    """
    comp_to_var = {0: "vx_full", 1: "vz_full", 2: "ex_full", 3: "ez_full"}
    run_shot = build_forward_fn(
        nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
        rec_x, rec_z, fd_order=fd_order, free_surface=free_surface,
        return_vars=[comp_to_var[component]],
    )

    if free_surface:
        nz_t = nz_dom + pad
    else:
        nz_t = nz_dom + 2 * pad
    nx_t = nx_dom + 2 * pad

    Ws = np.zeros((nz_t, nx_t))
    src_x_arr = jnp.asarray(src_x, dtype=jnp.int32)
    src_z_arr = jnp.asarray(src_z, dtype=jnp.int32)
    for i in range(len(src_x)):
        (field,) = run_shot(
            jnp.asarray(Vs), jnp.asarray(Vp), jnp.asarray(rho),
            src_x_arr[i], src_z_arr[i], src_wavelet,
        )
        Ws += np.asarray(jnp.sum(field ** 2, axis=0))
        if (i + 1) % 10 == 0 or i == len(src_x) - 1:
            print(f"    eprecond3 Ws: shot {i+1}/{len(src_x)}")

    We_full = eprecond3(
        Ws, rec_x=rec_x, rec_z=rec_z, dx=dx, dz=dz,
        free_surface=free_surface, pad=pad, eps=eps,
    )

    z_start = 0 if free_surface else pad
    We_phys = We_full[z_start:z_start + nz_dom, pad:pad + nx_dom]

    prec = 1.0 / We_phys
    prec = prec / np.max(prec)            # normalise to [0, 1]
    return prec


def _block_scales(params):
    return [max(float(np.abs(p).max()), 1e-12) for p in params]


def _active_keys(invert):
    if invert == "vs":
        return ["Vs"]
    if invert == "vs_vp":
        return ["Vs", "Vp"]
    if invert == "all":
        return ["Vs", "Vp", "rho"]
    raise ValueError(f"invert={invert!r}; use 'vs', 'vs_vp', or 'all'")


class _Plateau:
    """Stop a stage early when  mean(last_N losses) > (1 − tol)·mean(prev_N).
    Disabled when window is None or 0."""

    def __init__(self, window: Optional[int], rel_tol: float):
        self.window = (window or 0)
        self.rel_tol = float(rel_tol)
        self.history: list[float] = []

    def push(self, loss: float) -> bool:
        self.history.append(float(loss))
        if self.window <= 0:
            return False
        if len(self.history) < 2 * self.window:
            return False
        last = np.mean(self.history[-self.window:])
        prev = np.mean(self.history[-2 * self.window:-self.window])
        return last >= prev * (1.0 - self.rel_tol)


def run_multiscale_fwi(
    *,
    Vs_init: np.ndarray,
    Vp_init: np.ndarray,
    rho_init: np.ndarray,
    src_x: np.ndarray, src_z: np.ndarray,
    rec_x: np.ndarray, rec_z: np.ndarray,
    nx_dom: int, nz_dom: int,
    dx: float, dz: float, dt: float, nt: int, pad: int,
    observed: np.ndarray,                          # (n_shots, nt, nrec)
    stages: Sequence[dict],
    component: int = 2,
    backend: str = "ad",
    fd_order: int = 8,
    free_surface: bool = True,
    src_type: str = "force",
    block_size: int = 200,
    auto_scale: Optional[float] = None,
    K_margin: float = 0.04,
    cfl_coeff: Optional[float] = None,
    vs_bounds: Optional[tuple] = None,
    vp_bounds: Optional[tuple] = None,
    rho_bounds: Optional[tuple] = None,
    plateau_window: Optional[int] = None,
    plateau_rel_tol: float = 1e-3,
    save_dir: Optional[str] = None,
    save_every: Optional[int] = None,
    verbose: bool = True,
):
    """
    Multistage L-BFGS-B FWI loop. See module docstring.

    Each stage dict supports:
        f_c                 : float    Ricker dominant frequency [Hz]   (required)
        cutoff              : float    Butterworth lowpass cutoff [Hz]   (None disables)
        n_iters             : int      max L-BFGS-B iterations            (default 50)
        invert              : str      'vs' | 'vs_vp' | 'all'            (default 'vs')
        n_taper_pct         : float    fraction of nt for cosine taper    (default 0.0)
        illumination_eps    : float    if set, eprecond illumination ON   (default None)
        grad_smooth_sigma   : float    Gaussian smooth on raw gradient    (default None)
        source_taper        : (nz, nx) (default None)
    """
    if not stages:
        raise ValueError("`stages` must be a non-empty list")

    if auto_scale is None:
        auto_scale = float(1.0 / (np.mean(observed ** 2) + 1e-45))
    if verbose:
        print(f"auto_scale = {auto_scale:.4e}")

    # CFL cap (if cfl_coeff given) tightened by user vp upper-bound.
    vp_cap = None
    if cfl_coeff is not None:
        vp_cap = cfl_vp_cap(dx=dx, dz=dz, dt=dt, cfl_coeff=cfl_coeff)
    if vp_bounds is not None and vp_bounds[1] is not None:
        vp_cap = vp_bounds[1] if vp_cap is None else min(vp_cap, vp_bounds[1])
    if verbose and vp_cap is not None:
        print(f"vp_cap (CFL/user) = {vp_cap:.3f}")

    # Current model
    params = (
        np.asarray(Vs_init,  dtype=np.float64).copy(),
        np.asarray(Vp_init,  dtype=np.float64).copy(),
        np.asarray(rho_init, dtype=np.float64).copy(),
    )

    history = []

    for istage, stage in enumerate(stages):
        f_c               = float(stage["f_c"])
        cutoff            = stage.get("cutoff", None)
        n_iters           = int(stage.get("n_iters", 50))
        invert            = stage.get("invert", "vs")
        n_taper_pct       = float(stage.get("n_taper_pct", 0.0))
        illumination_eps  = stage.get("illumination_eps", None)
        grad_smooth_sigma = stage.get("grad_smooth_sigma", None)
        source_taper      = stage.get("source_taper", None)

        if verbose:
            print()
            print("=" * 70)
            print(f"Stage {istage+1}/{len(stages)}  f_c={f_c} Hz  "
                  f"cutoff={cutoff}  invert={invert}  n_iters={n_iters}  "
                  f"backend={backend}")
            print("=" * 70)

        # Per-stage source wavelet
        src_wavelet = ricker_jax(jnp.arange(nt) * dt, f_c, 1.5 / f_c)

        # Optional Butterworth lowpass
        sos = None
        if cutoff is not None:
            sos = butter(6, cutoff, fs=1.0 / dt, output="sos")

        n_taper = int(n_taper_pct * nt)

        # Optional preconditioner — computed once at the START of the stage
        # at the current model. Two flavours:
        #   precon='illum'      (default)  →  H_inv = 1/(Ws + ε·max Ws), normalised.
        #   precon='eprecond3'             →  Plessix-Mulder Hessian approximation.
        # Either way we build a [0, 1] multiplier and pass it through
        # compute_gradient(..., illumination=prec) — which multiplies the
        # gradient by it inside `_postprocess`.
        precon = stage.get("precon", "illum")
        illumination = None
        if illumination_eps is not None and precon != "none":
            if precon == "illum":
                if verbose:
                    print("  Computing illumination (1/(Ws + ε·max))...")
                il = compute_illumination_for_stage(
                    nz_dom, nx_dom, dx, dz, dt, nt, f_c, pad, block_size,
                    rec_x, rec_z, fd_order, component,
                    jnp.asarray(params[0]), jnp.asarray(params[1]),
                    jnp.asarray(params[2]),
                    src_x, src_z, src_wavelet, eps=illumination_eps,
                )
                illumination = np.asarray(il)
            elif precon == "eprecond3":
                if verbose:
                    print("  Computing eprecond3 (Plessix-Mulder)...")
                illumination = _compute_eprecond3_preconditioner(
                    nz_dom=nz_dom, nx_dom=nx_dom,
                    dx=dx, dz=dz, dt=dt, nt=nt, f_c=f_c, pad=pad,
                    block_size=block_size,
                    rec_x=rec_x, rec_z=rec_z, fd_order=fd_order,
                    component=component,
                    Vs=params[0], Vp=params[1], rho=params[2],
                    src_x=src_x, src_z=src_z, src_wavelet=src_wavelet,
                    free_surface=free_surface, eps=illumination_eps,
                )
            else:
                raise ValueError(
                    f"unknown precon={precon!r}; use 'illum', 'eprecond3', or 'none'")

        active = _active_keys(invert)
        scales = _block_scales(params)
        if verbose:
            print(f"  block scales (Vs, Vp, ρ) = "
                  f"({scales[0]:.3f}, {scales[1]:.3f}, {scales[2]:.3f})")
            print(f"  active blocks = {active}")

        n_cells = nz_dom * nx_dom

        # Pack / unpack helpers
        def pack(p_tuple):
            chunks = []
            for key in active:
                idx = _KEY_TO_IDX[key]
                chunks.append((p_tuple[idx] / scales[idx]).ravel())
            return np.concatenate(chunks)

        def unpack(x):
            full = list(params)        # start from frozen current model
            offset = 0
            for key in active:
                idx = _KEY_TO_IDX[key]
                full[idx] = (x[offset:offset + n_cells] * scales[idx]
                             ).reshape(nz_dom, nx_dom)
                offset += n_cells
            return tuple(full)

        # Per-cell bounds in scaled space
        bounds = []
        for key in active:
            lohi = {"Vs": vs_bounds, "Vp": vp_bounds, "rho": rho_bounds}[key]
            sc = scales[_KEY_TO_IDX[key]]
            if lohi is None:
                lo, hi = None, None
            else:
                lo, hi = lohi
            lo_s = None if lo is None else float(lo) / sc
            hi_s = None if hi is None else float(hi) / sc
            bounds.extend([(lo_s, hi_s)] * n_cells)

        x0 = pack(params)
        plateau = _Plateau(plateau_window, plateau_rel_tol)

        # Optional per-iteration checkpointing (fires from scipy callback,
        # which runs once per L-BFGS iteration, NOT per function eval).
        stage_save_dir = (os.path.join(save_dir, f"stage{istage+1}")
                          if save_dir is not None else None)
        if stage_save_dir is not None and save_every is not None:
            os.makedirs(stage_save_dir, exist_ok=True)
            # Save the initial model (epoch 0) for reference.
            np.savez_compressed(
                os.path.join(stage_save_dir, "epoch_0000.npz"),
                Vs=params[0], Vp=params[1], rho=params[2],
                iter=0, loss=float("nan"),
            )
            if verbose:
                print(f"    -> Saved epoch 0  (initial model)")

        # State the optimiser will modify
        eval_count = [0]
        iter_count = [0]
        loss_history: list[float] = []
        last_x = [x0.copy()]

        def f_and_g(x):
            eval_count[0] += 1
            last_x[0] = x.copy()
            params_curr = unpack(x)
            # Feasibility projection on velocity (K-margin + CFL cap)
            params_proj = project_velocity(
                *params_curr, K_margin=K_margin, vp_cap=vp_cap,
            )
            (g_Vs, g_Vp, g_rho), loss = compute_gradient(
                backend=backend,
                Vs=params_proj[0], Vp=params_proj[1], rho=params_proj[2],
                src_x=src_x, src_z=src_z, src_wavelet=src_wavelet,
                observed=observed, rec_x=rec_x, rec_z=rec_z,
                nx_dom=nx_dom, nz_dom=nz_dom,
                dx=dx, dz=dz, dt=dt, nt=nt, fc=f_c, pad=pad,
                component=component, invert=invert, scale=auto_scale,
                sos=sos, n_taper=n_taper,
                fd_order=fd_order, free_surface=free_surface, src_type=src_type,
                block_size=block_size,
                grad_smooth_sigma=grad_smooth_sigma,
                source_taper=source_taper,
                illumination=illumination,
            )
            # Pack the (scaled) gradient for active blocks only.
            full_grads = (g_Vs, g_Vp, g_rho)
            chunks = []
            for key in active:
                idx = _KEY_TO_IDX[key]
                chunks.append((full_grads[idx] * scales[idx]).ravel())
            g_scaled = np.concatenate(chunks).astype(np.float64)

            loss_history.append(float(loss))
            if verbose:
                print(f"  iter {eval_count[0]:4d}  loss={float(loss):.4e}  "
                      f"|g|_∞={np.max(np.abs(g_scaled)):.3e}")

            # Save by FUNCTION-EVALUATION count (matches the iter N printed
            # in the log). One L-BFGS iteration may include several evals
            # during the Wolfe line search; saving here checkpoints every
            # `save_every` evals.
            if (stage_save_dir is not None and save_every is not None
                    and eval_count[0] % save_every == 0):
                p_curr = unpack(np.asarray(x))
                p_proj = project_velocity(*p_curr,
                                          K_margin=K_margin, vp_cap=vp_cap)
                np.savez_compressed(
                    os.path.join(stage_save_dir,
                                 f"eval_{eval_count[0]:04d}.npz"),
                    Vs=p_proj[0], Vp=p_proj[1], rho=p_proj[2],
                    eval=eval_count[0], loss=float(loss),
                )
                if verbose:
                    print(f"    -> Saved eval {eval_count[0]}")

            if plateau.push(float(loss)):
                if verbose:
                    print("  ↳ plateau detected — aborting stage")
                # Raising aborts scipy.minimize; we catch externally.
                raise _PlateauStop()

            return float(loss), g_scaled

        def iter_callback(xk):
            """Fires once per L-BFGS iteration (not per function eval)."""
            iter_count[0] += 1
            if (stage_save_dir is None or save_every is None
                    or iter_count[0] % save_every != 0):
                return
            p_curr = unpack(np.asarray(xk))
            # Apply the same projection we apply during gradient eval, so the
            # checkpointed model is the actual feasible model used.
            p_proj = project_velocity(*p_curr, K_margin=K_margin, vp_cap=vp_cap)
            np.savez_compressed(
                os.path.join(stage_save_dir, f"epoch_{iter_count[0]:04d}.npz"),
                Vs=p_proj[0], Vp=p_proj[1], rho=p_proj[2],
                iter=iter_count[0],
                loss=float(loss_history[-1]) if loss_history else float("nan"),
            )
            if verbose:
                print(f"    -> Saved epoch {iter_count[0]}")

        t_start = time.time()
        try:
            res = minimize(
                f_and_g, x0,
                method="L-BFGS-B", jac=True, bounds=bounds,
                callback=iter_callback,
                options=dict(
                    maxiter=n_iters,
                    # Disable the gradient-norm stopping criterion: with
                    # block scaling + filtered residuals, |g|_∞ can sit
                    # well below SciPy's default `gtol=1e-5` even when
                    # the model is far from the optimum. Let the plateau
                    # detector (or maxiter) terminate the stage instead.
                    gtol=0.0,
                    # Keep ftol off the default-tight setting; rely on
                    # plateau / maxiter for stopping.
                    ftol=1e-30,
                    disp=False,
                ),
            )
            res_x = res.x
            res_fun = float(res.fun)
            res_success = bool(res.success)
        except _PlateauStop:
            # Roll back to the last x scipy passed us.
            res_x = last_x[0]
            res_fun = loss_history[-1] if loss_history else float("nan")
            res_success = False
        t_stage = time.time() - t_start

        # Update model from optimiser result + final projection
        params_new = unpack(res_x)
        params = project_velocity(*params_new, K_margin=K_margin, vp_cap=vp_cap)

        history.append(dict(
            stage=istage, f_c=f_c, cutoff=cutoff, invert=invert,
            n_iters_max=n_iters, n_evals=eval_count[0],
            loss_history=np.array(loss_history),
            final_loss=res_fun, success=res_success,
            time_s=t_stage,
            Vs=params[0].copy(), Vp=params[1].copy(), rho=params[2].copy(),
        ))

        if verbose:
            print(f"  Stage {istage+1} done in {t_stage:.1f}s  "
                  f"final_loss={res_fun:.4e}  success={res_success}")

        if save_dir is not None:
            stage_dir = os.path.join(save_dir, f"stage{istage+1}")
            os.makedirs(stage_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(stage_dir, "params.npz"),
                Vs=params[0], Vp=params[1], rho=params[2],
                loss_history=np.array(loss_history),
                f_c=f_c, cutoff=cutoff, invert=invert, backend=backend,
            )

    return params, history


class _PlateauStop(Exception):
    """Internal: aborts scipy.minimize when the plateau detector fires."""
    pass

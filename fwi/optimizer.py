"""
fwi/optimizer.py — Inversion loops for FWI (L-BFGS).

Runs iterative inversion using the loss function
(which internally computes gradients via JAX autodiff) at each epoch.

Author: Minh Nhat Tran
Date: 2026
"""
import os
import time

import jax
import jax.numpy as jnp
import jaxopt

from .plots import save_velocity_plot, save_gradient_plot, save_wiggle_comparison


def run_lbfgs(loss_fn, init_params, n_epochs,
              save_every=20, save_dir='results',
              true_models=None, history_size=100,
              dx=1.0, dz=1.0,
              run_shot=None, observed=None, src_x_list=None, src_z=None,
              src_wavelet=None, dt=None, component=1, invert='all',
              early_stop_ratio=None, step_pct=0.05):
    """
    Run L-BFGS inversion.

    Parameters
    ----------
    loss_fn : callable
        value_and_grad function from build_loss_fn.
    init_params : tuple
        (Vs, Vp, rho) initial models.
    n_epochs : int
        Number of L-BFGS iterations.
    save_every : int
        Save plots every N epochs.
    save_dir : str
        Output directory.
    true_models : tuple, optional
        (Vp_true, Vs_true, rho_true) for comparison plots.
    history_size : int
        L-BFGS history size.
    dx, dz : float
        Grid spacing for plot labels.
    run_shot : callable, optional
        Forward function for wiggle comparison. If provided with observed,
        saves wiggle plots every save_every epochs.
    observed : array, optional
        Observed data (n_shots, nt, n_rec).
    src_x_list, src_z : array, optional
        Source positions.
    src_wavelet : array, optional
        Source wavelet.
    dt : float, optional
        Time step (for wiggle plot axis).
    component : int
        Component index for wiggle comparison (default 1=vz).

    Returns
    -------
    params : tuple
        Inverted (Vs, Vp, rho).
    loss_history : list
        Loss value at each epoch.
    """
    do_wiggle = (run_shot is not None and observed is not None
                 and src_x_list is not None and src_wavelet is not None and dt is not None)
    os.makedirs(save_dir, exist_ok=True)

    params    = init_params
    loss_history = []

    # Save initial model comparison before inversion starts
    if true_models is not None:
        save_velocity_plot(0, params, true_models, save_dir, dx=dx, dz=dz,
                           invert=invert, init_params=init_params)
        print("  -> Saved initial model")

    t_start = time.time()
    print(f"\nStarting L-BFGS  ({n_epochs} epochs)  →  {save_dir}/")

    # =================== MAIN INVERSION LOOP ====================
    # Initial gradient to set first max_stepsize
    # Use the most restrictive (smallest) stepsize across all inverted params
    _, init_grads = loss_fn(params)

    param_names = ['Vs', 'Vp', 'Rho']
    active = [0]  # Vs always active
    if invert in ('vs_vp', 'all'):
        active.append(1)
    if invert == 'all':
        active.append(2)

    stepsizes = []
    for i in active:
        g_max = jnp.max(jnp.abs(init_grads[i]))
        p_max = jnp.max(jnp.abs(params[i]))
        ss = step_pct * p_max / (g_max + 1e-30)
        stepsizes.append(float(ss))
        print(f"  max({param_names[i]})={p_max:.1f}  max(|grad_{param_names[i]}|)={g_max:.4e}  → stepsize={ss:.4e}")

    max_stepsize = min(stepsizes)
    print(f"  → max_stepsize={max_stepsize:.4e} (most restrictive)")

    # ============ Set up L-BFGS optimizer with initial max_stepsize ============

    optimizer = jaxopt.LBFGS(fun=loss_fn, value_and_grad=True, maxiter=1, history_size=history_size,
                              max_stepsize=float(max_stepsize), maxls=40)
    state = optimizer.init_state(params)

    loss_first = None
    
    for epoch in range(1, n_epochs + 1):
        params, state = optimizer.update(params, state)
        loss_history.append(float(state.value))

        if loss_first is None:
            loss_first = loss_history[0]

        # Recompute max_stepsize from current gradient (already in state, no extra cost)
        if state.grad is not None:
            stepsizes = []
            for i in active:
                g_max = jnp.max(jnp.abs(state.grad[i]))
                p_max = jnp.max(jnp.abs(params[i]))
                stepsizes.append(float(step_pct * p_max / (g_max + 1e-30)))
            max_stepsize = min(stepsizes)
            optimizer = jaxopt.LBFGS(fun=loss_fn, value_and_grad=True, maxiter=1, history_size=history_size,
                                      max_stepsize=float(max_stepsize), maxls=40)

        ratio = loss_history[-1] / (loss_first + 1e-30)
        print(f"  Epoch {epoch:04d}/{n_epochs}  Loss: {loss_history[-1]:.4e}  ratio: {ratio:.6f}  step={max_stepsize:.4e}")

        # Early stopping: if loss/loss_init < threshold, skip to next stage
        if early_stop_ratio is not None and ratio < early_stop_ratio:
            print(f"  ** Early stop: loss/loss_init = {ratio:.6f} < {early_stop_ratio} → moving to next stage")
            break

        if epoch % save_every == 0:
            if true_models is not None:
                save_velocity_plot(epoch, params, true_models, save_dir, dx=dx, dz=dz,
                                   invert=invert, init_params=init_params)
            if state.grad is not None:
                save_gradient_plot(epoch, state.grad, save_dir, dx=dx, dz=dz, invert=invert)
            if do_wiggle:
                Vs, Vp, rho = params
                mid = len(src_x_list) // 2
                pred = run_shot(Vs, Vp, rho, jnp.int32(src_x_list[mid]), jnp.int32(src_z[mid]), src_wavelet)
                save_wiggle_comparison(epoch, observed[mid], pred[component], dt, save_dir, shot_idx=mid)
            print(f"    -> Saved figures for epoch {epoch}")

    elapsed = time.time() - t_start
    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal inversion time: {int(mins)}m {secs:.1f}s")

    # Save inverted model parameters
    import numpy as np
    Vs_inv, Vp_inv, rho_inv = params
    np.save(os.path.join(save_dir, 'Vs_inverted.npy'), np.array(Vs_inv))
    np.save(os.path.join(save_dir, 'Vp_inverted.npy'), np.array(Vp_inv))
    np.save(os.path.join(save_dir, 'rho_inverted.npy'), np.array(rho_inv))
    np.save(os.path.join(save_dir, 'loss_history.npy'), np.array(loss_history))
    print(f"  Inverted params saved to {save_dir}/")

    return params, loss_history


def run_lbfgs_alternating(loss_fn, init_params, n_epochs,
                          save_every=20, save_dir='results',
                          true_models=None, history_size=100,
                          dx=1.0, dz=1.0,
                          run_shot=None, observed=None, src_x_list=None, src_z=None,
                          src_wavelet=None, dt=None, component=1, invert='vs_vp',
                          early_stop_ratio=None, step_pct=0.05):
    """
    Alternating L-BFGS: separate optimizer + stepsize for each parameter.

    Each epoch:
      1) Update Vs only (zero Vp/rho gradient, Vs stepsize)
      2) Update Vp only (zero Vs/rho gradient, Vp stepsize)

    Same signature as run_lbfgs so it can be swapped in Main scripts.
    """
    do_wiggle = (run_shot is not None and observed is not None
                 and src_x_list is not None and src_wavelet is not None and dt is not None)
    os.makedirs(save_dir, exist_ok=True)

    params = init_params
    loss_history = []

    if true_models is not None:
        save_velocity_plot(0, params, true_models, save_dir, dx=dx, dz=dz,
                           invert=invert, init_params=init_params)
        print("  -> Saved initial model")

    # Wrappers: zero out gradients for the parameter NOT being updated
    def loss_fn_vs(params):
        loss, grads = loss_fn(params)
        return loss, (grads[0], jnp.zeros_like(grads[1]), jnp.zeros_like(grads[2]))

    def loss_fn_vp(params):
        loss, grads = loss_fn(params)
        return loss, (jnp.zeros_like(grads[0]), grads[1], jnp.zeros_like(grads[2]))

    # Initial stepsizes (separate for each param)
    _, init_grads = loss_fn(params)

    ss_vs = float(step_pct * jnp.max(jnp.abs(params[0])) / (jnp.max(jnp.abs(init_grads[0])) + 1e-30))
    ss_vp = float(step_pct * jnp.max(jnp.abs(params[1])) / (jnp.max(jnp.abs(init_grads[1])) + 1e-30))
    print(f"  Vs: max={jnp.max(jnp.abs(params[0])):.1f}  |grad|={jnp.max(jnp.abs(init_grads[0])):.4e}  → ss={ss_vs:.4e}")
    print(f"  Vp: max={jnp.max(jnp.abs(params[1])):.1f}  |grad|={jnp.max(jnp.abs(init_grads[1])):.4e}  → ss={ss_vp:.4e}")

    # Two separate L-BFGS optimizers with their own history and stepsize
    opt_vs = jaxopt.LBFGS(fun=loss_fn_vs, value_and_grad=True, maxiter=1,
                           history_size=history_size, max_stepsize=ss_vs, maxls=40)
    opt_vp = jaxopt.LBFGS(fun=loss_fn_vp, value_and_grad=True, maxiter=1,
                           history_size=history_size, max_stepsize=ss_vp, maxls=40)

    state_vs = opt_vs.init_state(params)
    state_vp = opt_vp.init_state(params)

    t_start = time.time()
    print(f"\nStarting Alternating L-BFGS  ({n_epochs} epochs)  →  {save_dir}/")

    loss_first = None

    for epoch in range(1, n_epochs + 1):
        # Step 1: Update Vs (Vp fixed)
        params, state_vs = opt_vs.update(params, state_vs)
        # Step 2: Update Vp (Vs fixed)
        params, state_vp = opt_vp.update(params, state_vp)

        loss_val = float(state_vp.value)
        loss_history.append(loss_val)

        if loss_first is None:
            loss_first = loss_history[0]

        # Recompute stepsizes from current gradients
        if state_vs.grad is not None and state_vp.grad is not None:
            ss_vs = float(step_pct * jnp.max(jnp.abs(params[0])) / (jnp.max(jnp.abs(state_vs.grad[0])) + 1e-30))
            ss_vp = float(step_pct * jnp.max(jnp.abs(params[1])) / (jnp.max(jnp.abs(state_vp.grad[1])) + 1e-30))
            opt_vs = jaxopt.LBFGS(fun=loss_fn_vs, value_and_grad=True, maxiter=1,
                                   history_size=history_size, max_stepsize=ss_vs, maxls=40)
            opt_vp = jaxopt.LBFGS(fun=loss_fn_vp, value_and_grad=True, maxiter=1,
                                   history_size=history_size, max_stepsize=ss_vp, maxls=40)

        ratio = loss_val / (loss_first + 1e-30)
        print(f"  Epoch {epoch:04d}/{n_epochs}  Loss: {loss_val:.4e}  ratio: {ratio:.6f}  ss_vs={ss_vs:.4e}  ss_vp={ss_vp:.4e}")

        if early_stop_ratio is not None and ratio < early_stop_ratio:
            print(f"  ** Early stop: ratio={ratio:.6f} < {early_stop_ratio}")
            break

        if epoch % save_every == 0:
            if true_models is not None:
                save_velocity_plot(epoch, params, true_models, save_dir, dx=dx, dz=dz,
                                   invert=invert, init_params=init_params)
            if state_vs.grad is not None and state_vp.grad is not None:
                combined_grad = (state_vs.grad[0], state_vp.grad[1], state_vs.grad[2])
                save_gradient_plot(epoch, combined_grad, save_dir, dx=dx, dz=dz, invert=invert)
            if do_wiggle:
                Vs, Vp, rho = params
                mid = len(src_x_list) // 2
                pred = run_shot(Vs, Vp, rho, jnp.int32(src_x_list[mid]), jnp.int32(src_z[mid]), src_wavelet)
                save_wiggle_comparison(epoch, observed[mid], pred[component], dt, save_dir, shot_idx=mid)
            print(f"    -> Saved figures for epoch {epoch}")

    elapsed = time.time() - t_start
    mins, secs = divmod(elapsed, 60)
    print(f"\nTotal inversion time: {int(mins)}m {secs:.1f}s")

    import numpy as np
    Vs_inv, Vp_inv, rho_inv = params
    np.save(os.path.join(save_dir, 'Vs_inverted.npy'), np.array(Vs_inv))
    np.save(os.path.join(save_dir, 'Vp_inverted.npy'), np.array(Vp_inv))
    np.save(os.path.join(save_dir, 'rho_inverted.npy'), np.array(rho_inv))
    np.save(os.path.join(save_dir, 'loss_history.npy'), np.array(loss_history))
    print(f"  Inverted params saved to {save_dir}/")

    return params, loss_history



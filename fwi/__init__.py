from .adjoint_jax_kernel import adjoint_jax, pml_apply_T
from .adjoint_solver import forward_simple, adjoint_simple, inner_product
from .forward import forward_jax, build_forward_fn, ricker_jax
from .imaging import imaging_condition
from .gradient import compute_gradient
from .optim_lbfgsb import run_multiscale_fwi
from .preconditioning import eprecond1, eprecond3, taper_grad
from .projections import (
    R_CONTINUUM, k_safe_ratio, cfl_vp_cap, project_velocity,
)
from .illumination import compute_illumination_for_stage, save_illumination_plot
from .loss import build_loss_fn
from .optimizer import run_lbfgs, run_lbfgs_alternating
from .plots import save_acquisition_plot, save_wavelet_plot, save_multistage_loss_curve, save_shot_qc
from .taper import build_source_taper, save_source_taper_plot

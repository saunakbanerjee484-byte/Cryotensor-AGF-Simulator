# pyright: reportArgumentType=false, reportCallIssue=false, reportReturnType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""
core_physics/thaw_consolidation/solver.py
=============================================
Module 5's main pipeline: sparse tridiagonal assembly, the implicit
Backward-Euler time-march with a moving (thaw-front) boundary, and the
vectorized settlement integral.

    d(u)/dt = d/dz[ cv(z,t) * d(u)/dz ]                (excess pore pressure)
    (I - dt * L) u^(n+1) = u^n + b^(n+1)                (Backward Euler)

The ONLY Python-level loop permitted anywhere in Module 5 is the outer
transient time-march below (`for step in range(1, n_steps+1)`) -- it is
required for the moving thaw boundary and the void-ratio-dependent (hence
step-dependent) diffusion operator; every operation INSIDE each iteration is
a fully vectorized NumPy/`scipy.sparse` expression over the whole depth grid.
"""

from __future__ import annotations

import html as _html

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .param import ConsolidationParams, DynamicConsolidationParams
from .boundaries import (
    extract_thaw_front_indices, thaw_front_depth_m, resample_front_to_solver_grid,
    active_thawed_mask, newly_thawed_mask,
)
from .physics import (
    overburden_effective_stress_profile, dynamic_cv, staggered_midpoint_average,
    instantaneous_thaw_void_ratio, void_ratio_from_effective_stress,
    settlement_from_void_ratio_history,
)

THAW_SIGMA_REF_FRACTION = 0.02   # [-] post-thaw "near-zero" effective stress, as a fraction of
                                  # the full overburden sigma'_0 -- the compression-line anchor
                                  # point right after instantaneous thaw strain (avoids log(0))
RESIDUAL_VOID_RATIO_FLOOR = 0.30   # [-] a soil cannot physically densify below this in this model
THAW_SIGMA_REF_FLOOR_PA = 50.0   # [Pa] absolute floor (sigma'_0 -> 0 at the ground surface itself)


def build_sparse_diffusion_operator(cv_mid: np.ndarray, dz: float, n: int) -> sp.csr_matrix:
    """
    Assemble the tridiagonal diffusion operator L (L @ u ~= d/dz[cv du/dz]) from
    the staggered-grid midpoint cv values via `scipy.sparse.diags`. Pure
    vectorized array construction -- no loop over depth nodes.
    """
    a = cv_mid / dz ** 2                    # length n-1
    main = np.zeros(n)
    main[:-1] -= a
    main[1:] -= a
    return sp.diags([a, main, a], offsets=[-1, 0, 1], shape=(n, n), format="csr")


def resolve_surcharge_at_step(dyn: DynamicConsolidationParams, step: int, n_steps: int) -> float:
    """Feature 3: look up q(t) for this solver step; zero if no history was supplied."""
    if dyn.surcharge_history_pa is None:
        return 0.0
    arr = np.asarray(dyn.surcharge_history_pa)
    idx = min(step, len(arr) - 1)
    return float(arr[idx])


from typing import Any

def run_thaw_consolidation_simulation(T_history_column: np.ndarray, t_module1: np.ndarray,
                                       y_module1: np.ndarray, p: ConsolidationParams,
                                       dyn: DynamicConsolidationParams) -> dict[str, Any]:
    """
    Full Module 5 solve, driven by a 1D vertical temperature-history column
    handed off from Module 1 (see modules/module5.py for how that column is
    extracted from Module 1's 2D spatial `T_history`).

    Pipeline: Step 1 (vectorized thaw-front extraction) -> Step 2/3 (sparse
    Backward-Euler time march with dynamic cv and a moving boundary) -> Step 4
    (vectorized settlement integral).
    """
    n = int(dyn.n_nodes)
    L = dyn.domain_depth_m
    z = np.linspace(0.0, L, n)
    dz = L / (n - 1)

    dt = dyn.dt_hours * 3600.0
    n_steps = max(1, int(round(dyn.total_days * 86400.0 / dt)))
    record_every = max(1, int(dyn.record_every))

    # ---- Step 1: vectorized thaw-front extraction (Module 1's column -> physical depth) ----
    front_idx_m1 = extract_thaw_front_indices(T_history_column)             # (n_time_m1,)
    front_depth_m1 = thaw_front_depth_m(front_idx_m1, y_module1)             # (n_time_m1,) [m]

    t_solver = np.arange(0, n_steps + 1) * dt
    front_idx_solver = resample_front_to_solver_grid(front_depth_m1, t_module1, t_solver, z)

    # ---- Initial state: everything frozen, at the pre-thaw void ratio ----
    e0_field = np.broadcast_to(np.asarray(p.initial_void_ratio_e0, dtype=float), (n,)).copy()
    e_field = e0_field.copy()
    u_field = np.zeros(n)
    sigma_geostatic = overburden_effective_stress_profile(z, p.effective_unit_weight_n_per_m3)
    # sigma_total_field is the CURRENT full-drainage target effective stress at each node --
    # starts at the geostatic overburden and is PERMANENTLY raised by any surcharge applied
    # to that node while it is active (Feature 3). This is what makes preloading actually
    # increase long-run settlement rather than just transiently delaying it: the ultimate
    # (fully-drained) effective stress genuinely goes up, not just the transient pore pressure.
    sigma_total_field = sigma_geostatic.copy()
    e_thaw_field = instantaneous_thaw_void_ratio(e0_field, p.thaw_strain_alpha)
    sigma_ref_field = np.maximum(THAW_SIGMA_REF_FRACTION * sigma_geostatic, THAW_SIGMA_REF_FLOOR_PA)

    Cc = p.compression_index_Cc
    Ck = p.permeability_index_Ck
    cv0 = p.coeff_consolidation_cv0

    t_list = [0.0]
    e_history = [e_field.copy()]
    u_history = [u_field.copy()]
    front_depth_history = [z[front_idx_solver[0]]]

    prev_front_idx = int(front_idx_solver[0])
    for step in range(1, n_steps + 1):
        curr_front_idx = int(front_idx_solver[step])
        active_mask = active_thawed_mask(curr_front_idx, n)
        newly_mask = newly_thawed_mask(prev_front_idx, curr_front_idx, n)

        # Feature 3: an incremental surcharge step PERMANENTLY raises the target effective
        # stress for already-active nodes (real preloading increases ultimate settlement),
        # and instantaneously (undrained) raises their current excess pore pressure by the
        # same increment -- classical superposition of a new loading stage.
        q_prev = resolve_surcharge_at_step(dyn, step - 1, n_steps)
        q_curr = resolve_surcharge_at_step(dyn, step, n_steps)
        dq = max(q_curr - q_prev, 0.0)
        if dq > 0.0:
            sigma_total_field = np.where(active_mask, sigma_total_field + dq, sigma_total_field)
            u_field = np.where(active_mask, u_field + dq, u_field)

        # Morgenstern-Nixon instantaneous excess-pore-pressure generation on newly-thawed nodes
        # (picking up whatever sigma_total_field -- geostatic + any already-applied surcharge --
        # applies at the moment they thaw), paired with the instantaneous thaw-strain void-
        # ratio drop (e0 -> e_thaw).
        u_field = np.where(newly_mask, sigma_total_field, u_field)
        e_field = np.where(newly_mask, e_thaw_field, e_field)

        # Feature 2: dynamic cv from the CURRENT void ratio field, staggered-grid averaged
        cv_array = dynamic_cv(e_field, e0_field, cv0, Ck)
        cv_mid = staggered_midpoint_average(cv_array)
        Lop = build_sparse_diffusion_operator(cv_mid, dz, n)

        # Backward Euler: (I/dt - L) u_new = u_old/dt
        A = (sp.diags(np.full(n, 1.0 / dt), format="lil") - Lop).tolil()
        b_vec = u_field / dt

        # Free-draining ground surface: Dirichlet u=0 at z=0
        A[0, :] = 0.0
        A[0, 0] = 1.0
        b_vec[0] = 0.0

        # Frozen nodes (not yet thawed): pinned at u=0 -- no pore pressure exists there yet
        frozen_idx = np.where(~active_mask)[0]
        if frozen_idx.size > 0:
            A[frozen_idx, :] = 0.0
            A[frozen_idx, frozen_idx] = 1.0
            b_vec[frozen_idx] = 0.0

        u_field = spla.spsolve(A.tocsr(), b_vec)
        u_field = np.where(active_mask, np.clip(u_field, 0.0, None), 0.0)

        # Effective stress increases as u dissipates -> void ratio collapses further along
        # the compression line anchored at (e_thaw, sigma_ref) for active/thawed nodes.
        sigma_eff_curr = sigma_total_field - u_field
        e_collapsed = void_ratio_from_effective_stress(sigma_eff_curr, sigma_ref_field, e_thaw_field, Cc)
        e_collapsed = np.maximum(e_collapsed, RESIDUAL_VOID_RATIO_FLOOR)
        e_field = np.where(active_mask, np.minimum(e_field, e_collapsed), e0_field)

        prev_front_idx = curr_front_idx

        if step % record_every == 0 or step == n_steps:
            t_list.append(step * dt)
            e_history.append(e_field.copy())
            u_history.append(u_field.copy())
            front_depth_history.append(z[curr_front_idx])

    # ---- Step 4: vectorized settlement integral (one np.trapz call, no loop over frames) ----
    e_history_2d = np.array(e_history)          # (n_frames, n)
    u_history_2d = np.array(u_history)          # (n_frames, n)
    settlement_m = settlement_from_void_ratio_history(e_history_2d, e0_field, z)

    settlement_rate_mm_day = np.zeros_like(settlement_m)
    if len(settlement_m) > 1:
        t_arr = np.array(t_list)
        dt_days = np.diff(t_arr) / 86400.0
        dS_mm = np.diff(settlement_m) * 1000.0
        settlement_rate_mm_day[1:] = dS_mm / np.clip(dt_days, 1.0e-9, None)

    return {
        "z": z, "t": np.array(t_list),
        "e_history": e_history_2d, "u_history": u_history_2d,
        "front_depth_history": np.array(front_depth_history),
        "settlement_m": settlement_m,
        "settlement_rate_mm_day": settlement_rate_mm_day,
        "max_settlement_rate_mm_day": float(np.max(np.abs(settlement_rate_mm_day))),
        "final_settlement_mm": float(settlement_m[-1] * 1000.0),
        "e0_field": e0_field,
        "params": p, "dyn_params": dyn,
    }


# =====================================================================================
# Deterministic engineering assessment (100% classical-physics thresholds, no ML)
# =====================================================================================
CRITICAL_SETTLEMENT_RATE_MM_DAY = 20.0
WARNING_SETTLEMENT_RATE_MM_DAY = 8.0


def evaluate_module5_status(max_settlement_rate_mm_day: float) -> tuple[str, str]:
    """
    RED   : sudden, rapid settlement -- void-ratio collapse outpacing safe limits.
    YELLOW: measurable ongoing consolidation settlement -- monitor.
    GREEN : settlement has stabilized (thaw consolidation essentially complete).
    """
    if max_settlement_rate_mm_day >= CRITICAL_SETTLEMENT_RATE_MM_DAY:
        return (
            "red",
            f"CRITICAL: Peak settlement rate ({max_settlement_rate_mm_day:.2f} mm/day) indicates "
            "rapid void-ratio collapse. Sudden differential settlement risk -- halt loading/traffic "
            "over the thawed zone.",
        )
    if max_settlement_rate_mm_day >= WARNING_SETTLEMENT_RATE_MM_DAY:
        return (
            "yellow",
            f"MODERATE: Active thaw consolidation in progress ({max_settlement_rate_mm_day:.2f} "
            "mm/day). Monitor settlement instrumentation closely.",
        )
    return (
        "green",
        f"STABLE: Settlement rate ({max_settlement_rate_mm_day:.2f} mm/day) is low -- thaw "
        "consolidation is approaching equilibrium.",
    )


# =====================================================================================
# "No Black Box" marquee content (pure string generation -- no Streamlit dependency)
# =====================================================================================
def render_thaw_consolidation_marquee() -> str:
    text = (
        "🌊 THAW CONSOLIDATION ACTIVE | ∂u/∂t = ∂/∂z[cv(z,t)·∂u/∂z] | "
        "cv(e) = cv0·10^[(e-e0)/Ck] | MORGENSTERN-NIXON VOID-RATIO COLLAPSE..."
    )
    safe_text = _html.escape(text)
    return f"""
<style>
.thaw-marquee-wrap {{
    width: 100%;
    background: #170a06;
    border: 1px solid #8c4a1f;
    border-radius: 6px;
    padding: 8px 0;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: inset 0 0 12px rgba(255, 140, 0, 0.15);
}}
.thaw-marquee-wrap marquee {{
    color: #ff9d3a;
    font-family: "Courier New", monospace;
    font-size: 15px;
    letter-spacing: 0.5px;
    text-shadow: 0 0 6px rgba(255, 157, 58, 0.65);
}}
</style>
<div class="thaw-marquee-wrap">
<marquee behavior="scroll" direction="left" scrollamount="6">{safe_text}</marquee>
</div>
"""
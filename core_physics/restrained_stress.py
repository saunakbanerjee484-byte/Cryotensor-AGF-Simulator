"""
core_physics/restrained_stress.py
====================================
Module 4: Thermo-Elastic Restrained Stress & THMC Degradation -- pure physics
(CPU-bound, vectorized, deterministic). No Streamlit UI logic lives here;
`modules/module4.py` renders it (Module 4C). This preserves the project's
established architecture: physics in core_physics/, presentation in
modules/, cross-module data via state_pipeline.py, routing via registry.py.

Pipeline overview (Module 4B)
--------------------------------
4B.1  Restrained Thermo-Elastic Core   -- eigenstrain -> Hooke's Law (sparse)
4B.2  Visco-Plastic Creep Tensor        -- J2-invariant, temperature-coupled
                                            Vialov/Ladanyi secondary creep,
                                            marched in the outer time loop
4B.3  Continuum Damage & Hydro-Chemical -- unified D in [0,1] from mechanical
      Degradation                          micro-cracking AND advective/solute
                                            thermal erosion; [E_damaged]=(1-D)[E]

Each of the four physics effects (creep, advective seepage, solute rejection,
continuum damage) can be independently toggled on/off via `StressParams`
flags, driven by the Module 4C control-panel checkboxes -- toggling them off
collapses the pipeline back toward pure restrained elasticity so an engineer
can isolate each effect's contribution.

Governing physics
-------------------
4B.1 -- Eigenstrain decomposition (isotropic, from Module 3's scalar eps_v)
    eps_xx_0 = eps_yy_0 = eps_v / 3 ,  gamma_xy_0 = 0
    beta(x,y) = [K_wall/(K_wall+K_soil)] * exp[-(x_wall-x)/L_char]   (restraint)
    eps_elastic = -beta * (eps_0 - eps_vp)
    {sigma_xx,sigma_yy,tau_xy} = BlockDiag[(1-D_i)E_i * D_unit(nu)] @ {eps_elastic}
    D_unit(nu) = 1/[(1+nu)(1-2nu)] * [[1-nu,nu,0],[nu,1-nu,0],[0,0,(1-2nu)/2]]

4B.2 -- J2 stress-invariant, temperature-coupled Vialov/Ladanyi secondary creep
    sigma_zz = nu*(sigma_xx+sigma_yy)                    (plane-strain closure)
    s_ij = sigma_ij - mean(sigma_xx,sigma_yy,sigma_zz)*delta_ij   (deviatoric)
    J2 = 0.5*(s_xx^2+s_yy^2+s_zz^2) + tau_xy^2
    sigma_eq = sqrt(3*J2)                                  (von Mises equivalent)
    A(T) = A0 * exp[-k_T * max(T_f - T, 0)]                (colder ice creeps slower)
    d(eps_vp)/dt = A(T) * (sigma_eq/sigma_ref)^n            [only where sigma_eq is
                                                               "high" and the node
                                                               is in the frozen fringe]
    eps_vp accumulates each outer time-march sub-step, relaxing eps_elastic.

4B.3 -- Unified damage (mechanical + hydro-chemical), then degraded restiffening
    D_mech(x,y)    = clip[(eps_v - eps_tensile)/(eps_ultimate - eps_tensile), 0, 1]
    D_thermal(x,y) = clip[|adv_term(x,y)| / EROSION_SCALE, 0, THERMAL_DAMAGE_CAP]
    D(x,y)         = clip[D_mech + D_thermal, 0, 1]
    -> fed back into 4B.1's node_modulus = (1-D)*E each creep sub-step, so the
       stress on the wall dynamically redistributes as damage accumulates.
    adv_term  = rho_w*c_w*v_x*dT/dx   (scipy.sparse first-derivative operator)
    C(x,y)    = C_0 + C_reject*(1-unfrozen_fraction(T))     (solute rejection proxy)
    T_f_depressed = T_f,nominal - k_cryoscopic * C(x,y)

Numerics (CPU-bound, no GPU / no ML / no spatial Python loops)
-----------------------------------------------------------------
- Every spatial field is a single vectorized NumPy/broadcast expression.
- The damaged stiffness-tensor contraction (4B.1, re-evaluated each 4B.2
  creep sub-step using the CURRENT damage field) is one `scipy.sparse`
  block-diagonal matrix-vector product across all N grid nodes:
      Big_D = scipy.sparse.kron(scipy.sparse.diags((1-D)*E), D_unit(nu))
      stress_vec = Big_D @ strain_vec
- The advective dT/dx term (4B.3) is one sparse first-derivative operator
  (`scipy.sparse.diags`) applied once to the whole temperature field.
- The ONLY Python-level loop is the short outer visco-plastic creep
  time-march (CREEP_STEPS sub-steps) -- a TEMPORAL loop, not a spatial one;
  every iteration inside it is itself fully vectorized over the grid.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


# =====================================================================================
# Parameter container (Module 4A/4C control panel inputs, including 4C.1 toggles)
# =====================================================================================
@dataclass
class StressParams:
    """THMC Restrained Stress parameters, including the 4C.1 physics on/off toggles."""
    youngs_modulus_frozen: float = 45.0e6      # E  [Pa]   frozen soil elastic modulus
    poisson_ratio_frozen: float = 0.35         # nu [-]
    concrete_yield_strength_mpa: float = 25.0  # [MPa] retaining-wall/liner allowable pressure
    creep_n: float = 3.0                       # [-] Vialov/Ladanyi secondary-creep power-law exponent
    seepage_vx: float = 1.0e-7                 # [m/s] groundwater seepage velocity (advective term)
    initial_salinity_ppt: float = 5.0          # [ppt] initial pore-water salinity

    # ---- 4C.1 physics toggles -------------------------------------------------------
    enable_creep: bool = True
    enable_advective_seepage: bool = True
    enable_solute_rejection: bool = True
    enable_damage: bool = True


# ---- Fixed, documented physical constants (not exposed as inputs -- "no black box":
#      named here, not buried in formulas) -------------------------------------------
E_CONCRETE_PA = 30.0e9          # typical structural concrete elastic modulus [Pa]
NU_CONCRETE = 0.20              # typical concrete Poisson's ratio [-]
EPS_TENSILE_LIMIT = 1.0e-4      # [-] volumetric strain at which micro-cracking initiates
EPS_ULTIMATE = 1.0e-3           # [-] volumetric strain at full mechanical damage (D=1)
THERMAL_DAMAGE_CAP = 0.5        # [-] max damage contribution from advective/thermal erosion alone
EROSION_SCALE = 5.0             # [W/m3-scale] |adv_term| producing THERMAL_DAMAGE_CAP damage
CREEP_A0 = 2.0e-10              # [1/s] Vialov/Ladanyi creep-rate coefficient at sigma_ref, T=T_f
CREEP_TEMP_SENSITIVITY = 0.15   # [1/degC] creep slows as the ice gets colder (Ladanyi-type)
SIGMA_REF_MPA = 1.0             # [MPa] creep-law reference stress
CREEP_STEPS = 6                 # outer temporal loop length (NOT a spatial loop)
CREEP_DT_S = 4.0 * 3600.0       # [s] per creep sub-step
CREEP_ACTIVATION_FRACTION = 0.3  # creep activates above this fraction of yield strength
RHO_WATER = 1000.0              # [kg/m3]
C_WATER = 4186.0                # [J/kg/K]
CRYOSCOPIC_C_PER_PPT = 0.06     # [degC per ppt] linear freezing-point depression coefficient
SOLUTE_REJECTION_GAIN_PPT = 15.0  # [ppt] max additional salinity from full solute rejection


# =====================================================================================
# 4B.1 -- Restrained Thermo-Elastic Core
# =====================================================================================
def soil_plane_strain_modulus(E: float, nu: float) -> float:
    """K = E / [(1+nu)(1-2nu)] [Pa]."""
    return E / ((1.0 + nu) * (1.0 - 2.0 * nu))


def baseline_restraint_factor(p: StressParams) -> float:
    """beta_max = K_wall / (K_wall + K_soil) [-]."""
    K_soil = soil_plane_strain_modulus(p.youngs_modulus_frozen, p.poisson_ratio_frozen)
    K_wall = soil_plane_strain_modulus(E_CONCRETE_PA, NU_CONCRETE)
    return K_wall / (K_wall + K_soil)


def restraint_factor_field(x: np.ndarray, y: np.ndarray, p: StressParams,
                            wall_x: float | None = None) -> np.ndarray:
    """beta(x, y): strongest at the wall (x=wall_x), decaying into the free field."""
    if wall_x is None:
        wall_x = float(x.max())
    L_char = max(0.3 * (x.max() - x.min()), 1.0e-6)
    beta_max = baseline_restraint_factor(p)
    X, _ = np.meshgrid(x, y)
    return beta_max * np.exp(-(wall_x - X) / L_char)


def _plane_strain_D_unit(nu: float) -> np.ndarray:
    """3x3 plane-strain constitutive matrix with E factored OUT."""
    c = 1.0 / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0],
    ])


def stress_from_elastic_strain(eps_e_xx: np.ndarray, eps_e_yy: np.ndarray, eps_e_xy: np.ndarray,
                                D_field_vals: np.ndarray, p: StressParams):
    """
    4B.1's core contraction: a single block-diagonal `scipy.sparse` matrix-vector
    product converting the whole elastic-strain field into the whole (damage-
    degraded) stress field at once. Returns (sigma_xx, sigma_yy, tau_xy) in MPa.
    """
    ny, nx = eps_e_xx.shape
    N = eps_e_xx.size

    strain_vec = np.stack(
        [eps_e_xx.ravel(), eps_e_yy.ravel(), eps_e_xy.ravel()], axis=1
    ).ravel()

    D_unit = _plane_strain_D_unit(p.poisson_ratio_frozen)
    node_modulus = (1.0 - D_field_vals.ravel()) * p.youngs_modulus_frozen  # [E_damaged] = (1-D)[E]
    Big_D = sp.kron(sp.diags(node_modulus, format="csr"), sp.csr_matrix(D_unit), format="csr")

    stress_vec = Big_D @ strain_vec   # ONE sparse matvec, whole grid, damaged stiffness included
    stress_mat = stress_vec.reshape(N, 3)

    PA_TO_MPA = 1.0e-6
    sigma_xx = stress_mat[:, 0].reshape(ny, nx) * PA_TO_MPA
    sigma_yy = stress_mat[:, 1].reshape(ny, nx) * PA_TO_MPA
    tau_xy = stress_mat[:, 2].reshape(ny, nx) * PA_TO_MPA
    return sigma_xx, sigma_yy, tau_xy


def principal_stresses(sigma_xx: np.ndarray, sigma_yy: np.ndarray, tau_xy: np.ndarray):
    """2D closed-form principal stresses, elementwise vectorized."""
    avg = (sigma_xx + sigma_yy) / 2.0
    radius = np.sqrt(((sigma_xx - sigma_yy) / 2.0) ** 2 + tau_xy ** 2)
    return avg + radius, avg - radius


def compute_wall_face_pressure_profile(sigma_xx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Lateral earth pressure vs. depth at the structural interface (x = x.max())."""
    return np.abs(sigma_xx[:, -1])


# =====================================================================================
# 4B.2 -- Visco-Plastic Creep Tensor (J2-invariant, temperature-coupled)
# =====================================================================================
def von_mises_equivalent_stress(sigma_xx: np.ndarray, sigma_yy: np.ndarray, tau_xy: np.ndarray,
                                 nu: float) -> np.ndarray:
    """
    sigma_eq = sqrt(3*J2), the von Mises equivalent stress driving secondary creep,
    computed from the full deviatoric stress-tensor invariant (plane-strain closure
    sigma_zz = nu*(sigma_xx+sigma_yy)) rather than a bare principal-stress magnitude.
    """
    sigma_zz = nu * (sigma_xx + sigma_yy)
    mean_stress = (sigma_xx + sigma_yy + sigma_zz) / 3.0
    s_xx = sigma_xx - mean_stress
    s_yy = sigma_yy - mean_stress
    s_zz = sigma_zz - mean_stress
    J2 = 0.5 * (s_xx ** 2 + s_yy ** 2 + s_zz ** 2) + tau_xy ** 2
    return np.sqrt(3.0 * np.clip(J2, 0.0, None))


def creep_rate_coefficient_field(T_matrix: np.ndarray | None, shape: tuple) -> np.ndarray:
    """
    A(T) = A0 * exp[-k_T * max(T_f - T, 0)] -- colder ice creeps slower (Ladanyi-type
    temperature coupling). Falls back to the isothermal A0 field if no temperature
    profile is available.
    """
    if T_matrix is None:
        return np.full(shape, CREEP_A0)
    depression = np.maximum(0.0 - T_matrix, 0.0)
    return CREEP_A0 * np.exp(-CREEP_TEMP_SENSITIVITY * depression)


def run_creep_time_march(eps_v_field: np.ndarray, x: np.ndarray, y: np.ndarray, p: StressParams,
                          T_matrix: np.ndarray | None = None, wall_x: float | None = None):
    """
    4B.2 + 4B.3 coupled outer time-march: at each sub-step, (a) recompute the
    unified damage field from the CURRENT mechanical + thermal-erosion state,
    (b) run 4B.1's stress contraction with that damaged stiffness, (c) compute
    the J2-invariant equivalent stress and accumulate temperature-coupled
    visco-plastic creep strain, which relaxes the elastic strain used next
    sub-step. The outer loop (CREEP_STEPS iterations) is the module's only
    Python-level loop -- temporal, not spatial; each iteration is itself a
    fully vectorized grid-wide computation.

    If `p.enable_creep` is False, the loop still runs once to produce the
    elastic (non-creeping) solution, with the creep-strain increment forced
    to zero -- this is how the 4C.1 toggle collapses the pipeline back to
    pure restrained (+ damage, if enabled) elasticity.
    """
    D_mech = damage_field_mechanical(eps_v_field) if p.enable_damage else np.zeros_like(eps_v_field)
    beta_field = restraint_factor_field(x, y, p, wall_x=wall_x)
    eps0 = eps_v_field / 3.0
    fringe_active = eps_v_field > 0.0

    eps_vp = np.zeros_like(eps_v_field)
    peak_sigma_eq_history = []
    sigma_xx = sigma_yy = tau_xy = sigma_1 = sigma_2 = D_field = None
    creep_active_mask = np.zeros_like(eps_v_field, dtype=bool)

    A_field = creep_rate_coefficient_field(T_matrix, eps_v_field.shape)
    yield_mpa = p.concrete_yield_strength_mpa
    n_steps = CREEP_STEPS if p.enable_creep else 1

    for _ in range(n_steps):
        # 4B.3: unified damage, re-evaluated each sub-step (thermal contribution is
        # static across sub-steps here since it depends on T, not on stress/strain).
        D_field = D_mech  # thermal contribution folded in by run_restrained_stress_simulation

        eps_e = -beta_field * (eps0 - eps_vp)
        sigma_xx, sigma_yy, tau_xy = stress_from_elastic_strain(eps_e, eps_e, np.zeros_like(eps_e), D_field, p)
        sigma_1, sigma_2 = principal_stresses(sigma_xx, sigma_yy, tau_xy)
        sigma_eq = von_mises_equivalent_stress(sigma_xx, sigma_yy, tau_xy, p.poisson_ratio_frozen)
        peak_sigma_eq_history.append(float(np.max(sigma_eq)))

        if p.enable_creep:
            creep_active_mask = (sigma_eq >= CREEP_ACTIVATION_FRACTION * yield_mpa) & fringe_active
            eps_vp_rate = A_field * (np.clip(sigma_eq, 0.0, None) / SIGMA_REF_MPA) ** p.creep_n
            eps_vp = eps_vp + np.where(creep_active_mask, eps_vp_rate * CREEP_DT_S, 0.0)
            eps_vp = np.clip(eps_vp, 0.0, eps0)   # creep cannot relax more strain than exists

    return {
        "sigma_xx_mpa": sigma_xx, "sigma_yy_mpa": sigma_yy, "tau_xy_mpa": tau_xy,
        "sigma_1_mpa": sigma_1, "sigma_2_mpa": sigma_2,
        "D_mech_field": D_mech, "beta_field": beta_field,
        "eps_vp_field": eps_vp, "creep_active_mask": creep_active_mask,
        "peak_sigma_eq_history_mpa": np.array(peak_sigma_eq_history),
    }


# =====================================================================================
# 4B.3 -- Continuum Damage & Hydro-Chemical Degradation
# =====================================================================================
def damage_field_mechanical(eps_v_field: np.ndarray) -> np.ndarray:
    """D_mech(x,y) in [0,1] from cryogenic micro-cracking, vectorized bilinear law."""
    return np.clip(
        (eps_v_field - EPS_TENSILE_LIMIT) / (EPS_ULTIMATE - EPS_TENSILE_LIMIT), 0.0, 1.0
    )


def build_sparse_dx_operator(n: int, dx: float) -> sp.csr_matrix:
    """Central-difference first-derivative operator [1/m], one-sided at the edges."""
    main = np.zeros(n)
    lower = -0.5 * np.ones(n - 1) / dx
    upper = 0.5 * np.ones(n - 1) / dx
    Dx = sp.diags([lower, main, upper], offsets=[-1, 0, 1], format="lil")
    Dx[0, 0], Dx[0, 1] = -1.0 / dx, 1.0 / dx           # forward difference at left edge
    Dx[-1, -1], Dx[-1, -2] = 1.0 / dx, -1.0 / dx       # backward difference at right edge
    return Dx.tocsr()


def advective_thmc_diagnostics(T_matrix: np.ndarray, x: np.ndarray, y: np.ndarray, p: StressParams,
                                suction_context: dict | None = None):
    """
    Vectorized advective/solute diagnostic fields. Uses a single sparse first-
    derivative operator applied once to the whole 2D field (T_matrix @ Dx.T) --
    no loop over rows or columns.

    `suction_context`, when available (Module 2's live suction field), lightly
    modulates the solute-rejection gain via its mean suction gradient -- Module 2
    runs on a 1D vertical column while Modules 1/3/4 share a 2D (x,y) domain, so
    this is a deliberate scalar-only coupling rather than a false-precision field
    remap across mismatched grids.
    """
    nx = len(x)
    dx = float(x[1] - x[0])
    Dx = build_sparse_dx_operator(nx, dx)             # (nx, nx) sparse
    dTdx = T_matrix @ Dx.T                              # applied to every row at once

    adv_term = np.zeros_like(T_matrix)
    if p.enable_advective_seepage:
        adv_term = RHO_WATER * C_WATER * p.seepage_vx * dTdx

    T_freeze_nominal = 0.0
    solute_gain = SOLUTE_REJECTION_GAIN_PPT
    if suction_context is not None:
        mean_grad = float(np.mean(np.abs(np.gradient(suction_context["psi_history"][-1], suction_context["dz"]))))
        solute_gain *= float(np.clip(1.0 + mean_grad / 5000.0, 1.0, 2.0))  # bounded, documented scalar bump

    C_field = np.full_like(T_matrix, p.initial_salinity_ppt)
    T_f_depressed = np.full_like(T_matrix, T_freeze_nominal)
    if p.enable_solute_rejection:
        unfrozen_frac = np.clip((T_matrix - (T_freeze_nominal - 1.0)) / 1.0, 0.0, 1.0)
        C_field = p.initial_salinity_ppt + solute_gain * (1.0 - unfrozen_frac)
        T_f_depressed = T_freeze_nominal - CRYOSCOPIC_C_PER_PPT * C_field

    seepage_erosion_index = float(np.mean(np.abs(adv_term)) / 1.0)

    D_thermal = np.zeros_like(T_matrix)
    if p.enable_advective_seepage or p.enable_solute_rejection:
        D_thermal = np.clip(np.abs(adv_term) / EROSION_SCALE, 0.0, THERMAL_DAMAGE_CAP)

    return {
        "adv_term_field": adv_term,
        "solute_field_ppt": C_field,
        "T_f_depressed_field_c": T_f_depressed,
        "seepage_erosion_index": seepage_erosion_index,
        "D_thermal_field": D_thermal,
    }


# =====================================================================================
# Main solve -- sequences 4B.1 -> 4B.2 -> 4B.3 and folds the unified damage field
# back into the creep-marched stress solution
# =====================================================================================
def run_restrained_stress_simulation(eps_v_field: np.ndarray, x: np.ndarray, y: np.ndarray,
                                      p: StressParams, T_matrix: np.ndarray | None = None,
                                      suction_context: dict | None = None,
                                      wall_x: float | None = None) -> dict:
    """
    Full Module 4B THMC solve from one Module-3 volumetric-strain frame:
        4B.3 (thermal damage contribution, if T_matrix given)
          -> unified D = clip(D_mech + D_thermal, 0, 1)
          -> 4B.1 + 4B.2 creep-marched, damage-degraded restrained stress
    """
    advective = None
    D_thermal = np.zeros_like(eps_v_field)
    seepage_erosion_index = 0.0
    if T_matrix is not None and T_matrix.shape == eps_v_field.shape:
        advective = advective_thmc_diagnostics(T_matrix, x, y, p, suction_context=suction_context)
        seepage_erosion_index = advective["seepage_erosion_index"]
        if p.enable_damage:
            D_thermal = advective["D_thermal_field"]

    creep_result = run_creep_time_march(eps_v_field, x, y, p, T_matrix=T_matrix, wall_x=wall_x)

    # Fold in the thermal-erosion damage contribution and re-solve once more with
    # the fully unified damage field (4B.3's final "sequentially calculate" step).
    D_unified = np.clip(creep_result["D_mech_field"] + D_thermal, 0.0, 1.0) if p.enable_damage \
        else np.zeros_like(eps_v_field)
    eps_e_final = -creep_result["beta_field"] * (eps_v_field / 3.0 - creep_result["eps_vp_field"])
    sigma_xx, sigma_yy, tau_xy = stress_from_elastic_strain(eps_e_final, eps_e_final, np.zeros_like(eps_e_final),
                                                             D_unified, p)
    sigma_1, sigma_2 = principal_stresses(sigma_xx, sigma_yy, tau_xy)
    wall_profile = compute_wall_face_pressure_profile(sigma_xx, y)

    max_lateral_pressure_mpa = float(np.max(np.abs(sigma_xx)))
    max_damage = float(np.max(D_unified))
    creep_active_fraction = float(np.mean(creep_result["creep_active_mask"]))

    return {
        "sigma_xx_mpa": sigma_xx, "sigma_yy_mpa": sigma_yy, "tau_xy_mpa": tau_xy,
        "sigma_1_mpa": sigma_1, "sigma_2_mpa": sigma_2,
        "D_field": D_unified, "D_mech_field": creep_result["D_mech_field"], "D_thermal_field": D_thermal,
        "beta_field": creep_result["beta_field"],
        "eps_vp_field": creep_result["eps_vp_field"], "creep_active_mask": creep_result["creep_active_mask"],
        "peak_sigma_eq_history_mpa": creep_result["peak_sigma_eq_history_mpa"],
        "wall_face_pressure_mpa": wall_profile,
        "advective": advective,
        "max_lateral_pressure_mpa": max_lateral_pressure_mpa,
        "max_damage": max_damage,
        "creep_active_fraction": creep_active_fraction,
        "seepage_erosion_index": seepage_erosion_index,
        "x": x, "y": y, "params": p,
    }


# =====================================================================================
# Standalone mock strain-field generator (Module 4C.2 "Mock Data Mode" fallback)
# =====================================================================================
def generate_mock_strain_field(nx: int = 61, ny: int = 61, Lx: float = 3.0, Ly: float = 3.0,
                                pipe_cx: float = 1.5, pipe_cy: float = 1.5,
                                peak_strain: float = 5.0e-4):
    """Synthetic ring-shaped volumetric strain field around a mock freeze pipe."""
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - pipe_cx) ** 2 + (Y - pipe_cy) ** 2)
    ring_radius, ring_width = 0.35, 0.20
    eps_v_field = peak_strain * np.exp(-((r - ring_radius) ** 2) / (2.0 * ring_width ** 2))
    return x, y, eps_v_field


def generate_mock_temperature_field(x: np.ndarray, y: np.ndarray, pipe_cx: float = 1.5,
                                     pipe_cy: float = 1.5) -> np.ndarray:
    """Synthetic radial cooling field, same grid convention as Module 1, for Mock Data Mode."""
    X, Y = np.meshgrid(x, y)
    r = np.sqrt((X - pipe_cx) ** 2 + (Y - pipe_cy) ** 2)
    return 8.0 - 33.0 * np.exp(-r / 0.6)


# =====================================================================================
# Deterministic engineering assessment (100% classical-physics thresholds, no ML)
# =====================================================================================
def evaluate_module4_status(max_lateral_pressure_mpa: float, yield_mpa: float, max_damage: float,
                             creep_active_fraction: float, seepage_erosion_index: float):
    """
    RED   : lateral pressure exceeds wall capacity OR cryogenic damage D > 0.8.
    YELLOW: visco-plastic creep is active and/or advective seepage is measurably
            eroding the freeze-wall edge.
    GREEN : stress equilibrium maintained, wall capacity exceeds restrained pressure.
    """
    if max_lateral_pressure_mpa >= yield_mpa or max_damage > 0.8:
        return (
            "red",
            "CRITICAL YIELD: Lateral earth pressure exceeds retaining wall capacity OR "
            "Cryogenic Damage (D > 0.8) causing matrix failure.",
        )
    if creep_active_fraction > 0.0 or seepage_erosion_index > 0.05:
        return (
            "yellow",
            "MODERATE: Visco-plastic creep active. Advective seepage is eroding the "
            "freeze-wall edge. Monitor stresses.",
        )
    return (
        "green",
        "SAFE: Stress equilibrium maintained. Wall capacity > Restrained Pressure. "
        "Ground freeze stable.",
    )


# =====================================================================================
# "No Black Box" marquee content (pure string generation -- no Streamlit dependency)
# =====================================================================================
def render_restrained_stress_marquee() -> str:
    text = (
        "⚠️ THMC ENGINE ACTIVE | σ' = (1-D)E[ε - ε_vp] | "
        "ADVECTIVE SEEPAGE: ∇·(k∇T) - ρcv·∇T | COMPUTING RESTRAINED STRESS..."
    )
    safe_text = _html.escape(text)
    return f"""
<style>
.stress-marquee-wrap {{
    width: 100%;
    background: #06120a;
    border: 1px solid #1f6b3a;
    border-radius: 6px;
    padding: 8px 0;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: inset 0 0 12px rgba(0, 255, 120, 0.12);
}}
.stress-marquee-wrap marquee {{
    color: #39ff6a;
    font-family: "Courier New", monospace;
    font-size: 15px;
    letter-spacing: 0.5px;
    text-shadow: 0 0 6px rgba(57, 255, 106, 0.65);
}}
</style>
<div class="stress-marquee-wrap">
<marquee behavior="scroll" direction="left" scrollamount="6">{safe_text}</marquee>
</div>
"""

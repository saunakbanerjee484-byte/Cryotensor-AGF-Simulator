# pyright: reportArgumentType=false, reportCallIssue=false, reportReturnType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
"""
core_physics/cryosuction.py
=============================
Module 2: SFCC Cryogenic Suction Solver -- pure physics (CPU-bound, vectorized,
deterministic). No Streamlit UI logic lives here; `modules/module2.py` renders it.

Couples the Soil Freezing Characteristic Curve (SFCC, van Genuchten form) to a
1D vertical Richards' equation (moisture-content / diffusive form) through the
Generalized Clausius-Clapeyron relation, to track unfrozen-water migration and
cryogenic ice-lens feeding from a groundwater source toward an advancing
freezing front.

Governing physics
------------------
Generalized Clausius-Clapeyron cryogenic suction (frozen soil):
    psi_cryo(T) = [rho_w * L_f / (T_f,K * 1000)] * max(T_f - T, 0)      [kPa]

SFCC (van Genuchten, with an Air-Entry-Value plateau):
    theta_u(psi) = theta_r + (theta_s - theta_r) * [1 + (alpha * psi_eff)^n]^(-m)
    m = 1 - 1/n ,  psi_eff = max(psi_total - AEV, 0)

Mualem-van Genuchten relative hydraulic conductivity:
    K(theta) = k_sat * Se^0.5 * [1 - (1 - Se^(1/m))^m]^2 ,  Se = (theta-theta_r)/(theta_s-theta_r)

Richards' equation, diffusive (theta-based) form, z positive downward:
    d(theta)/dt = d/dz[ D(theta) * d(theta)/dz ] - dK(theta)/dz
    D(theta) = K(theta) / C(theta) ,  C(theta) = -d(theta)/d(psi)   (specific moisture capacity)

Numerics (CPU-bound, no GPU / no ML / no spatial Python loops)
-----------------------------------------------------------------
- theta is the primary unknown. K(theta), C(theta), D(theta) are all closed-form,
  fully vectorized NumPy expressions (theta -> psi inversion of van Genuchten is
  analytic, so no root-finding loop is needed either).
- The spatial diffusion operator is assembled ONCE per Picard sub-iteration as a
  tridiagonal `scipy.sparse.diags` matrix, built from harmonic-mean nodal
  diffusivities via pure array slicing (D[:-1], D[1:]) -- no loop over depth nodes.
- Backward-Euler (fully implicit) time integration for stability near saturation.
- Nonlinearity (D, K, C all depend on theta) is resolved with a Picard fixed-point
  iteration -- a handful of `scipy.sparse.linalg.spsolve` calls per step.
- The freezing front is represented by a MOCK temperature array (a placeholder for
  the real T(z,t) field that will eventually be handed off from Module 1's Stefan
  solver) advancing per a Stefan-type sqrt(time) law -- itself built with vectorized
  NumPy ops over the depth array.
- The ONLY genuine Python-level loop in this module is the outer transient time
  march, which is unavoidable for any implicit transient PDE solve.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =====================================================================================
# Physical constants (fixed, not user-editable -- fundamental properties of water/ice)
# =====================================================================================
RHO_WATER = 1000.0            # [kg/m3]
LATENT_HEAT_FUSION = 3.34e5   # [J/kg]  latent heat of fusion of water
T_FREEZE_C = 0.0              # [degC]  nominal freezing point at 1 atm
KELVIN_OFFSET = 273.15


# =====================================================================================
# Parameter containers (Sub-Modules 2A / 2B / 2C)
# =====================================================================================
@dataclass
class SFCCParams:
    """Sub-Module 2A -- Soil Freezing Characteristic Curve (van Genuchten)."""
    air_entry_kpa: float = 2.0        # [kPa]  Air Entry Value
    vg_n: float = 1.80                # [-]    van Genuchten n
    vg_alpha: float = 0.05            # [1/kPa] van Genuchten alpha
    theta_r: float = 0.05             # [-]    residual unfrozen water content
    theta_s: float = 0.42             # [-]    saturated water content (porosity)
    k_sat: float = 5.0e-8             # [m/s]  saturated hydraulic conductivity

    @property
    def m(self) -> float:
        return 1.0 - 1.0 / self.vg_n


@dataclass
class BoundaryParams:
    """Sub-Module 2B -- Hydrogeological Boundary Setup."""
    distance_to_water_table_m: float = 3.0    # [m]   domain depth, z=0 (front) to z=L (water table)
    initial_matric_suction_kpa: float = 5.0   # [kPa] background (pre-freeze) suction, uniform IC
    surface_temp_c: float = -12.0             # [degC] freeze-pipe / frozen-surface temperature (mock)
    ground_temp_c: float = 8.0                # [degC] undisturbed ground temperature ahead of the front
    temp_gradient_c_per_m: float = 4.0        # [degC/m] mock temperature gradient within the frozen zone
    front_advance_rate: float = 0.35          # [m / sqrt(day)] MOCK Stefan-type front advance coefficient
                                               # (placeholder for real T(z,t) handed off by Module 1)


@dataclass
class SolverParams:
    """Sub-Module 2C -- discretization / time-integration controls."""
    n_nodes: int = 41
    dt_hours: float = 2.0
    total_days: float = 10.0
    picard_iters: int = 5
    record_every: int = 6


# =====================================================================================
# Vectorized field physics -- NO spatial Python loops anywhere below
# =====================================================================================
def cryogenic_suction_kpa(T_celsius: np.ndarray, T_freeze_c: float = T_FREEZE_C) -> np.ndarray:
    """Generalized Clausius-Clapeyron cryogenic suction [kPa], vectorized over T."""
    Tf_K = T_freeze_c + KELVIN_OFFSET
    depression = np.maximum(T_freeze_c - T_celsius, 0.0)
    coeff = RHO_WATER * LATENT_HEAT_FUSION / (1000.0 * Tf_K)   # kPa per degC of depression
    return coeff * depression


def van_genuchten_theta(psi_total_kpa: np.ndarray, p: SFCCParams) -> np.ndarray:
    """Unfrozen/unsaturated water content theta(psi), vectorized. psi in kPa (total suction)."""
    psi_eff = np.maximum(psi_total_kpa - p.air_entry_kpa, 0.0)
    theta = p.theta_r + (p.theta_s - p.theta_r) * (1.0 + (p.vg_alpha * psi_eff) ** p.vg_n) ** (-p.m)
    return np.clip(theta, p.theta_r, p.theta_s)


def psi_from_theta(theta: np.ndarray, p: SFCCParams) -> np.ndarray:
    """Analytic inverse of the van Genuchten SFCC: psi(theta) [kPa], vectorized (no root-finding)."""
    Se = np.clip((theta - p.theta_r) / (p.theta_s - p.theta_r), 1e-6, 1.0 - 1e-9)
    psi_eff = ((Se ** (-1.0 / p.m) - 1.0) ** (1.0 / p.vg_n)) / p.vg_alpha
    return psi_eff + p.air_entry_kpa


def specific_moisture_capacity(psi_total_kpa: np.ndarray, p: SFCCParams) -> np.ndarray:
    """C(psi) = -d(theta)/d(psi) [1/kPa], analytic derivative of the van Genuchten SFCC, vectorized."""
    psi_eff = np.maximum(psi_total_kpa - p.air_entry_kpa, 1e-9)
    active = psi_total_kpa > p.air_entry_kpa
    base = 1.0 + (p.vg_alpha * psi_eff) ** p.vg_n
    dtheta_dpsi = (
        -(p.theta_s - p.theta_r) * p.m * p.vg_n * p.vg_alpha
        * (p.vg_alpha * psi_eff) ** (p.vg_n - 1.0)
        * base ** (-p.m - 1.0)
    )
    C = np.where(active, -dtheta_dpsi, 1.0e-4)
    # Floor at 1e-4 [1/kPa]: a standard "modified Picard" regularization (Celia et al. 1990)
    # that prevents the theta-based Richards equation from degenerating as C -> 0 at full
    # saturation, without materially altering behavior in the active unsaturated range.
    return np.clip(C, 1.0e-4, None)


def hydraulic_conductivity(theta: np.ndarray, p: SFCCParams) -> np.ndarray:
    """Mualem-van Genuchten relative K -> absolute K(theta) [m/s], vectorized."""
    Se = np.clip((theta - p.theta_r) / (p.theta_s - p.theta_r), 1e-6, 1.0)
    Kr = np.sqrt(Se) * (1.0 - (1.0 - Se ** (1.0 / p.m)) ** p.m) ** 2
    return p.k_sat * np.clip(Kr, 1e-12, 1.0)


def diffusivity(theta: np.ndarray, p: SFCCParams) -> np.ndarray:
    """
    D(theta) = K(theta) / C(theta) [m2/s], vectorized.

    The ceiling is scaled to k_sat (D_max = k_sat / C_floor) rather than a fixed constant --
    a diffusivity cap that is independent of the soil's own conductivity would let a
    near-saturation numerical artifact (C -> C_floor) produce transport rates unbounded by
    the soil's actual ability to conduct water, defeating the k_sat-driven "capillary
    barrier" physics the alert system in this module depends on.
    """
    psi = psi_from_theta(theta, p)
    K = hydraulic_conductivity(theta, p)
    C = specific_moisture_capacity(psi, p)
    D = K / np.clip(C, 1.0e-4, None)
    D_max = p.k_sat / 1.0e-4
    return np.clip(D, 1e-16, D_max)


def mock_temperature_front(z: np.ndarray, t_seconds: float, b: BoundaryParams):
    """
    MOCK freezing-front temperature array T(z) at time t (placeholder for the real
    Module-1 Stefan-solver hand-off). Front advances per a Stefan-type sqrt(time)
    law; fully vectorized over the depth array z.

    Returns (T_celsius[z], front_depth_m).
    """
    t_days = max(t_seconds / 86400.0, 0.0)
    front_depth = float(np.clip(b.front_advance_rate * np.sqrt(t_days), 0.0, z.max()))
    T_gradient_profile = b.surface_temp_c + b.temp_gradient_c_per_m * z
    T = np.where(z <= front_depth, T_gradient_profile, b.ground_temp_c)
    return T, front_depth


def build_richards_operator(theta_guess: np.ndarray, dz: float, N: int, p: SFCCParams):
    """
    Assemble the 1D diffusive-flux tridiagonal operator L (such that L @ theta ~=
    d/dz[D d(theta)/dz]) via harmonic-mean nodal diffusivities, plus the gravity
    source term -dK/dz. Pure vectorized array slicing -- no loop over depth nodes.
    """
    D = diffusivity(theta_guess, p)
    D_half = 2.0 * D[:-1] * D[1:] / (D[:-1] + D[1:] + 1e-14)   # harmonic mean, length N-1
    a = D_half / dz ** 2

    main = np.zeros(N)
    main[:-1] -= a
    main[1:] -= a
    L = sp.diags([a, main, a], offsets=[-1, 0, 1], shape=(N, N))

    K = hydraulic_conductivity(theta_guess, p)
    dKdz = np.gradient(K, dz)
    gravity_source = -dKdz
    return L, gravity_source


# =====================================================================================
# Main transient solve
# =====================================================================================
def run_cryosuction_simulation(sfcc: SFCCParams, bnd: BoundaryParams, solver: SolverParams):
    """
    Run the fully-implicit, Picard-linearized transient Richards' equation solve,
    coupled to the mock freezing-front temperature array through the Clausius-
    Clapeyron cryogenic suction relation.

    Returns a dict with:
        z, t                         : depth [m] and recorded time [s] arrays
        theta_history, psi_history   : list of 1D arrays per recorded frame
        front_depth_history          : mock freeze-front depth per recorded frame [m]
        influx_velocity              : |Darcy flux| at the near-surface node per frame [m/s]
        suction_gradient             : mean |d(psi)/dz| per frame [kPa/m]
        sfcc_curve                   : (T_range_c, theta_u) for the 2A SFCC display curve
        params                       : dict of the three parameter dataclasses used
    """
    N = int(solver.n_nodes)
    L_domain = bnd.distance_to_water_table_m
    z = np.linspace(0.0, L_domain, N)
    dz = L_domain / (N - 1)

    dt = solver.dt_hours * 3600.0
    n_steps = max(1, int(round(solver.total_days * 86400.0 / dt)))
    record_every = max(1, int(solver.record_every))

    # Initial condition: uniform background matric suction (pre-freeze), no front yet.
    theta = van_genuchten_theta(np.full(N, bnd.initial_matric_suction_kpa), sfcc)

    t_list = [0.0]
    theta_history = [theta.copy()]
    psi_history = [psi_from_theta(theta, sfcc)]
    front_depth_history = [0.0]
    influx_list = [0.0]
    gradient_list = [0.0]

    for step in range(1, n_steps + 1):
        t = step * dt
        theta_old = theta.copy()

        T_z, front_depth = mock_temperature_front(z, t, bnd)
        psi_cryo_top = cryogenic_suction_kpa(np.array([T_z[0]]))[0]
        theta_top_bc = float(van_genuchten_theta(
            np.array([psi_cryo_top + bnd.initial_matric_suction_kpa]), sfcc
        )[0])
        theta_bottom_bc = sfcc.theta_s   # free water supply at the groundwater table

        theta_guess = theta_old.copy()
        for _ in range(solver.picard_iters):
            L_op, gravity_source = build_richards_operator(theta_guess, dz, N, sfcc)
            A = (sp.diags(np.full(N, 1.0 / dt)) - L_op).tolil()
            b_vec = theta_old / dt + gravity_source

            A[0, :] = 0.0
            A[0, 0] = 1.0
            b_vec[0] = theta_top_bc

            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            b_vec[-1] = theta_bottom_bc

            theta_new = spla.spsolve(A.tocsr(), b_vec)
            theta_guess = np.clip(theta_new, sfcc.theta_r, sfcc.theta_s)

        theta = theta_guess

        if step % record_every == 0 or step == n_steps:
            psi_now = psi_from_theta(theta, sfcc)
            D_now = diffusivity(theta, sfcc)
            K_now = hydraulic_conductivity(theta, sfcc)
            flux = -D_now * np.gradient(theta, dz) + K_now       # Darcy flux, vectorized
            influx = float(np.abs(flux[1]))                       # near-surface (toward-front) flux
            grad = float(np.mean(np.abs(np.gradient(psi_now, dz))))

            t_list.append(t)
            theta_history.append(theta.copy())
            psi_history.append(psi_now.copy())
            front_depth_history.append(front_depth)
            influx_list.append(influx)
            gradient_list.append(grad)

    # SFCC display curve (2A): theta_u vs. sub-zero temperature, independent of the transient run
    T_range_c = np.linspace(-10.0, 0.0, 200)
    psi_curve = cryogenic_suction_kpa(T_range_c) + bnd.initial_matric_suction_kpa
    theta_curve = van_genuchten_theta(psi_curve, sfcc)

    return {
        "z": z,
        "t": np.array(t_list),
        "theta_history": theta_history,
        "psi_history": psi_history,
        "front_depth_history": np.array(front_depth_history),
        "influx_velocity": np.array(influx_list),
        "suction_gradient": np.array(gradient_list),
        "sfcc_curve": (T_range_c, theta_curve),
        "params": {"sfcc": sfcc, "bnd": bnd, "solver": solver},
    }


# =====================================================================================
# Deterministic engineering assessment (100% classical-physics thresholds, no ML)
# =====================================================================================
# Named, documented, auditable constants -- "no black box". Calibrated against the
# realistic output range of this solver (frost-susceptible silt vs. clay-barrier vs.
# isothermal/no-front regimes): a continuous Darcy flux on the order of 1e-6-1e-5 m/s
# is the classic "highly frost-susceptible, unrestricted water supply" signature in
# frozen-soil literature, whereas fluxes below ~1e-7 m/s reflect a conductivity- or
# gradient-limited (capillary-barrier) regime.
CRITICAL_INFLUX_VELOCITY_MPS = 5.0e-6   # sustained Darcy flux above this -> continuous water-table feeding
WARNING_INFLUX_VELOCITY_MPS = 5.0e-8    # moderate, non-negligible moisture migration
CAPILLARY_BARRIER_K_SAT_MPS = 2.0e-9    # k_sat below this is conductivity-limiting (capillary barrier)
ACTIVE_GRADIENT_THRESHOLD_KPA_M = 20.0  # suction gradient considered "actively driving" flow


def evaluate_module2_status(mean_influx_velocity: float, k_sat: float, mean_suction_gradient: float):
    """
    RED   : sustained high moisture flux -> continuous feeding from the water table.
    YELLOW: an active suction gradient exists but growth is throttled by low k_sat
            (capillary barrier) or by a moderate (sub-critical) flux.
    GREEN : flux is minimal -> effectively a closed system.
    """
    if mean_influx_velocity >= CRITICAL_INFLUX_VELOCITY_MPS:
        return (
            "red",
            "CRITICAL: Continuous moisture feeding from water table detected. "
            "Massive ice lens formation in progress. Catastrophic frost heave imminent.",
        )

    active_gradient = mean_suction_gradient >= ACTIVE_GRADIENT_THRESHOLD_KPA_M
    throttled = k_sat < CAPILLARY_BARRIER_K_SAT_MPS
    moderate_flux = mean_influx_velocity >= WARNING_INFLUX_VELOCITY_MPS

    if active_gradient and (throttled or moderate_flux):
        return (
            "yellow",
            "MODERATE: Active suction gradient. Ice lens growth restricted by capillary barrier. "
            "Monitor closely.",
        )

    return (
        "green",
        "OPTIMAL: Closed-system freezing confirmed. Minimal moisture migration. "
        "Safe to proceed to Frost Heave Tensor.",
    )


# =====================================================================================
# "No Black Box" marquee content (pure string generation -- no Streamlit dependency)
# =====================================================================================
def render_cryosuction_marquee() -> str:
    text = (
        "🌊 CRYOSUCTION ACTIVE | CLAPEYRON-RICHARDS COUPLING: "
        "∂θ/∂t = ∇·[K(θ)∇(ψ + z)] | COMPUTING ICE LENS FLUX TENSOR..."
    )
    safe_text = html.escape(text)
    return f"""
<style>
.cryosuction-marquee-wrap {{
    width: 100%;
    background: #06120a;
    border: 1px solid #1f6b3a;
    border-radius: 6px;
    padding: 8px 0;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: inset 0 0 12px rgba(0, 255, 120, 0.12);
}}
.cryosuction-marquee-wrap marquee {{
    color: #39ff6a;
    font-family: "Courier New", monospace;
    font-size: 15px;
    letter-spacing: 0.5px;
    text-shadow: 0 0 6px rgba(57, 255, 106, 0.65);
}}
</style>
<div class="cryosuction-marquee-wrap">
<marquee behavior="scroll" direction="left" scrollamount="6">{safe_text}</marquee>
</div>
"""

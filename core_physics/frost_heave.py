"""
core_physics/frost_heave.py
=============================
Module 3: Volumetric Frost Heave Tensor -- pure physics (CPU-bound, vectorized,
deterministic). No Streamlit UI logic lives here; `modules/module3.py` renders it.

Converts a 2D temperature field (handed off from Module 1's Stefan solver via the
shared session-state pipeline, see `state_pipeline.py`) into a spatially resolved
volumetric heave-rate field and volumetric strain tensor field, combining two
classical frost-heave mechanisms:

Governing physics
------------------
1. In-situ expansion (primary heave): the 9% volumetric expansion of pore water
   converting to ice, scaled by porosity and the local phase-change (isotherm)
   velocity:
       h_insitu_dot = 0.09 * n * (dz/dt)

   The isotherm velocity dz/dt is recovered from the temperature field itself
   (no explicit front-tracking) via the standard identity relating an isotherm's
   normal velocity to its local temporal and spatial temperature derivatives:
       dz/dt = |dT/dt| / |grad(T)|

2. Segregation heave (secondary heave / ice lensing): the Konrad & Morgenstern
   Segregation Potential (SP) model, in which the water-intake velocity feeding
   an ice lens is proportional to the temperature gradient in the frozen fringe,
   with SP itself decaying exponentially under overburden (effective) pressure:
       v_s = SP * grad(T)
       SP  = SP0 * exp(-a * P)

Both mechanisms are restricted to the active freezing fringe -- the narrow
sub-zero band (e.g. 0 degC to -0.5 degC) where unfrozen pore water still exists
and ice segregation is physically active -- isolated via pure NumPy boolean
masking.

3. The volumetric strain tensor: the (isotropic) volumetric strain increment per
   recorded interval is the local heave rate normalized by the vertical cell
   size:
       d(eps_v) = h_dot * dt / dy

   and the ground-surface uplift profile is the depth integral of the heave-rate
   field:
       uplift_rate(x) = integral over y of h_dot(x, y) dy

Numerics (CPU-bound, no GPU / no ML / no spatial Python loops)
-----------------------------------------------------------------
- The freezing fringe is isolated with pure NumPy boolean masking:
      fringe_mask = (T <= T_upper) & (T > T_lower)
- Spatial temperature gradients use `np.gradient()` over the full 2D field.
- The isotherm/front velocity field, segregation flux field, in-situ expansion
  field, strain field, and surface-uplift profile are all single vectorized
  NumPy expressions over the whole grid -- there is NO Python `for` loop over
  grid nodes anywhere in this module (the depth-integral for the uplift profile
  is a single `np.sum(..., axis=0)`, not a loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# =====================================================================================
# Parameter container (Module 3 control panel inputs)
# =====================================================================================
@dataclass
class HeaveParams:
    """Sub-Module 3 -- Volumetric Frost Heave Tensor parameters."""
    porosity: float = 0.42                          # n [-]
    sp0_mm2_per_c_day: float = 2.0                   # SP0 [mm^2 / (degC . day)] (Konrad classification: ~1-3 = moderate)
    overburden_pressure_kpa: float = 50.0            # P [kPa]
    pressure_decay_coeff_per_kpa: float = 0.03       # a [1/kPa]
    fringe_t_upper_c: float = 0.0                    # freezing fringe upper bound [degC]
    fringe_t_lower_c: float = -0.5                   # freezing fringe lower bound [degC]
    structural_tolerance_mm_per_day: float = 2.0     # allowable heave rate [mm/day]
    water_to_ice_expansion: float = 0.09             # [-] fixed physical constant (~9%)

    @property
    def sp0_si(self) -> float:
        """SP0 converted to SI: [m^2 / (degC . s)]."""
        return self.sp0_mm2_per_c_day * 1.0e-6 / 86400.0


# =====================================================================================
# Vectorized field physics -- NO spatial Python loops anywhere below
# =====================================================================================
def identify_freezing_fringe(T_field: np.ndarray, T_upper: float = 0.0, T_lower: float = -0.5) -> np.ndarray:
    """Boolean mask isolating the active freezing fringe, vectorized."""
    return (T_field <= T_upper) & (T_field > T_lower)


def temperature_gradient_field(T_field: np.ndarray, dx: float, dy: float):
    """grad(T) magnitude and components [degC/m], vectorized via np.gradient."""
    dTdy, dTdx = np.gradient(T_field, dy, dx)
    grad_mag = np.sqrt(dTdx ** 2 + dTdy ** 2)
    return grad_mag, dTdx, dTdy


def phase_front_velocity_field(T_curr: np.ndarray, T_prev: np.ndarray, dt_s: float,
                                dx: float, dy: float, eps: float = 1.0e-6) -> np.ndarray:
    """
    Isotherm (phase-front) normal velocity field [m/s], recovered from the
    temperature field's own temporal and spatial derivatives -- no explicit
    front-tracking required:
        dz/dt = |dT/dt| / |grad(T)|
    Vectorized over the whole grid.
    """
    dTdt = (T_curr - T_prev) / dt_s
    grad_mag, _, _ = temperature_gradient_field(T_curr, dx, dy)
    return np.abs(dTdt) / np.clip(grad_mag, eps, None)


def segregation_potential(p: HeaveParams) -> float:
    """SP = SP0 * exp(-a * P), converted to SI [m^2 / (degC . s)]."""
    return p.sp0_si * np.exp(-p.pressure_decay_coeff_per_kpa * p.overburden_pressure_kpa)


def compute_heave_rate_field(T_curr: np.ndarray, T_prev: np.ndarray, dt_s: float,
                              dx: float, dy: float, p: HeaveParams):
    """
    Assemble the total volumetric heave-rate field h_dot(x, y) [m/s], combining
    in-situ (9% expansion) and segregation (Konrad-Morgenstern SP) heave, masked
    to the active freezing fringe. Fully vectorized -- no spatial loop.

    Returns (h_dot_field, fringe_mask, grad_mag_field, front_velocity_field, SP_effective).
    """
    fringe = identify_freezing_fringe(T_curr, p.fringe_t_upper_c, p.fringe_t_lower_c)
    grad_mag, _, _ = temperature_gradient_field(T_curr, dx, dy)
    v_front = phase_front_velocity_field(T_curr, T_prev, dt_s, dx, dy)
    SP_eff = segregation_potential(p)

    h_insitu = p.water_to_ice_expansion * p.porosity * v_front
    v_seg = SP_eff * grad_mag

    h_dot = np.where(fringe, h_insitu + v_seg, 0.0)
    return h_dot, fringe, grad_mag, v_front, SP_eff


def compute_volumetric_strain_field(h_dot_field: np.ndarray, dt_s: float, dy: float) -> np.ndarray:
    """Volumetric strain increment field d(eps_v) = h_dot * dt / dy [-], vectorized."""
    return h_dot_field * dt_s / dy


def compute_surface_uplift_profile(h_dot_field: np.ndarray, dy: float) -> np.ndarray:
    """
    Ground-surface uplift RATE profile [m/s] as a function of x: the depth
    integral of the heave-rate field, via a single vectorized `np.sum(axis=0)`
    (not a spatial loop).
    """
    return np.sum(h_dot_field, axis=0) * dy


# =====================================================================================
# Main solve (single-frame, driven by a T(t) pair handed off from Module 1)
# =====================================================================================
def run_frost_heave_simulation(T_curr: np.ndarray, T_prev: np.ndarray, dt_s: float,
                                x: np.ndarray, y: np.ndarray, p: HeaveParams) -> dict[str, Any]:
    """
    Compute the full Module 3 field set for one recorded interval [T_prev -> T_curr].

    Returns a dict with:
        h_dot_field_mps        : total volumetric heave rate [m/s], 2D
        h_dot_field_mm_day     : same, in [mm/day] for display
        fringe_mask            : boolean 2D mask of the active freezing fringe
        grad_mag_field         : |grad(T)| [degC/m], 2D
        front_velocity_field   : isotherm velocity [m/s], 2D
        strain_field           : volumetric strain increment [-], 2D
        uplift_rate_profile_mm_day : surface uplift rate vs. x [mm/day], 1D
        SP_effective            : pressure-decayed segregation potential [m2/(degC.s)]
        max_heave_rate_mm_day   : scalar, governs the deterministic alert
        x, y                    : grid coordinate arrays (pass-through)
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    h_dot, fringe, grad_mag, v_front, SP_eff = compute_heave_rate_field(T_curr, T_prev, dt_s, dx, dy, p)
    strain_field = compute_volumetric_strain_field(h_dot, dt_s, dy)
    uplift_rate_profile = compute_surface_uplift_profile(h_dot, dy)   # [m/s]

    MPS_TO_MM_DAY = 1000.0 * 86400.0
    h_dot_mm_day = h_dot * MPS_TO_MM_DAY
    uplift_rate_mm_day = uplift_rate_profile * MPS_TO_MM_DAY

    return {
        "h_dot_field_mps": h_dot,
        "h_dot_field_mm_day": h_dot_mm_day,
        "fringe_mask": fringe,
        "grad_mag_field": grad_mag,
        "front_velocity_field": v_front,
        "strain_field": strain_field,
        "uplift_rate_profile_mm_day": uplift_rate_mm_day,
        "SP_effective": float(SP_eff),
        "max_heave_rate_mm_day": float(np.max(h_dot_mm_day)),
        "mean_fringe_heave_rate_mm_day": float(np.mean(h_dot_mm_day[fringe])) if np.any(fringe) else 0.0,
        "x": x,
        "y": y,
        "params": p,
    }


# =====================================================================================
# Standalone mock temperature-field generator (fallback when Module 1 has not been
# run yet in this session -- illustrative radial cooling only, NOT a Stefan solution)
# =====================================================================================
def generate_mock_temperature_fields(
    nx: int = 61, ny: int = 61, Lx: float = 3.0, Ly: float = 3.0,
    pipe_cx: float = 1.5, pipe_cy: float = 1.5, pipe_radius: float = 0.08,
    pipe_temp: float = -25.0, far_field_temp: float = 12.0,
    t1_days: float = 5.0, t2_days: float = 5.25,
):
    """
    Generate two closely-spaced-in-time synthetic radial-cooling temperature
    snapshots (degC) for standalone demonstration/testing of Module 3 when no
    real Module 1 output is present in the shared state pipeline. Purely
    illustrative (an exponential radial decay, not an actual Stefan solution);
    fully vectorized via `np.meshgrid`.

    Returns (x, y, T_prev, T_curr, dt_seconds).
    """
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    r = np.clip(np.sqrt((X - pipe_cx) ** 2 + (Y - pipe_cy) ** 2), pipe_radius, None)

    def field(t_days: float) -> np.ndarray:
        decay_length = 0.15 + 0.35 * np.sqrt(max(t_days, 1e-6))
        return far_field_temp + (pipe_temp - far_field_temp) * np.exp(-(r - pipe_radius) / decay_length)

    T_prev = field(t1_days)
    T_curr = field(t2_days)
    dt_seconds = (t2_days - t1_days) * 86400.0
    return x, y, T_prev, T_curr, dt_seconds


# =====================================================================================
# Deterministic engineering assessment (100% classical-physics thresholds, no ML)
# =====================================================================================
def evaluate_module3_status(max_heave_rate_mm_day: float, tolerance_mm_day: float):
    """
    RED   : local heave rate meets or exceeds the allowable structural tolerance.
    YELLOW: within 50% of tolerance -- approaching the design limit.
    GREEN : comfortably below tolerance.
    """
    if max_heave_rate_mm_day >= tolerance_mm_day:
        return (
            "red",
            f"CRITICAL: Peak local heave rate ({max_heave_rate_mm_day:.3f} mm/day) has met or "
            f"exceeded the allowable structural tolerance ({tolerance_mm_day:.3f} mm/day). "
            "Halting operation -- risk of differential heave damage to overlying structures.",
        )
    if max_heave_rate_mm_day >= 0.5 * tolerance_mm_day:
        return (
            "yellow",
            f"MODERATE: Peak local heave rate ({max_heave_rate_mm_day:.3f} mm/day) is approaching "
            f"the allowable tolerance ({tolerance_mm_day:.3f} mm/day). Monitor closely.",
        )
    return (
        "green",
        f"OPTIMAL: Peak local heave rate ({max_heave_rate_mm_day:.3f} mm/day) is well within the "
        f"allowable structural tolerance ({tolerance_mm_day:.3f} mm/day).",
    )


# =====================================================================================
# "No Black Box" marquee content (pure string generation -- no Streamlit dependency)
# =====================================================================================
def render_frost_heave_marquee() -> str:
    import html as _html
    text = (
        "🧊 FROST HEAVE TENSOR ACTIVE | KONRAD-MORGENSTERN SEGREGATION: "
        "v_s = SP·∇T , SP = SP₀·e^(-aP) | IN-SITU: ḣ = 0.09·n·(dz/dt) | "
        "VOLUMETRIC STRAIN: Δε_v = ḣ·Δt/Δy | COMPUTING ICE LENS TENSOR FIELD..."
    )
    safe_text = _html.escape(text)
    return f"""
<style>
.heave-marquee-wrap {{
    width: 100%;
    background: #170a06;
    border: 1px solid #a8501f;
    border-radius: 6px;
    padding: 8px 0;
    margin-bottom: 14px;
    overflow: hidden;
    box-shadow: inset 0 0 12px rgba(255, 140, 0, 0.15);
}}
.heave-marquee-wrap marquee {{
    color: #ff9d3d;
    font-family: "Courier New", monospace;
    font-size: 15px;
    letter-spacing: 0.5px;
    text-shadow: 0 0 6px rgba(255, 157, 61, 0.65);
}}
</style>
<div class="heave-marquee-wrap">
<marquee behavior="scroll" direction="left" scrollamount="6">{safe_text}</marquee>
</div>
"""

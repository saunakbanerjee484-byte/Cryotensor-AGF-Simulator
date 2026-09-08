"""
core_physics/cryosuction.py
============================
Sub-Module 2 (PLANNED): SFCC Cryogenic Suction Solver.

Will couple the Soil Freezing Characteristic Curve (SFCC) to a Richards'-equation
style moisture-migration solver to track cryogenic suction and ice-lens growth:

    Richards' equation (coupled form, unsaturated/frozen soil):
        d(theta_w)/dt = div( K(theta_w, T) * grad(psi + z) ) - S_ice

    SFCC (Clapp-Hornberger / van Genuchten style, frozen form):
        theta_w(T) = theta_r + (theta_s - theta_r) * [1 + (alpha * |T - T_f|)^n]^(-m)   for T < T_f

    Clausius-Clapeyron cryogenic suction:
        psi_cryo = (L / (g * T_f)) * (T_f - T)      [m of suction head]

This module intentionally contains NO implementation yet -- it defines the
parameter contract that Module 1's output (T_history) will feed into, so
Module 1 does not need to be re-architected when Module 2 lands.

Planned public API (matching the pattern used in phase_change.py):
    class SuctionParams(dataclass): ...
    def sfcc_theta_w(T, p) -> np.ndarray            # vectorized
    def cryogenic_suction_head(T, p) -> np.ndarray  # vectorized
    def run_cryosuction_simulation(T_history, p) -> dict
"""

from dataclasses import dataclass


@dataclass
class SuctionParams:
    """Placeholder parameter contract for Module 2 (values are typical soil defaults)."""
    theta_s: float = 0.42        # saturated volumetric water content [-]
    theta_r: float = 0.05        # residual volumetric water content [-]
    van_genuchten_alpha: float = 1.5   # [1/m]
    van_genuchten_n: float = 1.8       # [-]
    hydraulic_conductivity_sat: float = 1.0e-7  # [m/s]


def run_cryosuction_simulation(*args, **kwargs):
    raise NotImplementedError(
        "Module 2 (SFCC Cryogenic Suction Solver) is scheduled for the next "
        "development phase. This stub preserves the API contract consumed by app.py."
    )

"""
core_physics/thaw_consolidation.py
=====================================
Sub-Module 5 (PLANNED): Thaw Consolidation Simulator.

Will track post-freeze melt-down: rapid void ratio collapse, excess pore
water pressure generation, and settlement, using the classical thaw
consolidation framework (Morgenstern & Nixon):

    Excess pore pressure generation at the thaw front:
        u_excess = sigma'_0 * (1 - Cc / (1 + e0) * log10(...))   [simplified void-ratio-collapse form]

    Thaw consolidation ratio:
        R = alpha * sqrt(t) / c_v      (Morgenstern-Nixon thaw coefficient)

    Settlement:
        S(t) = integral_0^Xf  (e0 - e(z)) / (1 + e0) dz

Planned public API:
    class ConsolidationParams(dataclass): ...
    def excess_pore_pressure(thaw_front_history, p) -> np.ndarray
    def settlement_time_series(thaw_front_history, p) -> np.ndarray
    def run_thaw_consolidation_simulation(T_history, p) -> dict
"""

from dataclasses import dataclass


@dataclass
class ConsolidationParams:
    compression_index_Cc: float = 0.35     # [-]
    initial_void_ratio_e0: float = 0.85    # [-]
    coeff_consolidation_cv: float = 1.2e-7  # [m2/s]
    thaw_strain_alpha: float = 0.08        # [-] Morgenstern-Nixon thaw strain coefficient


def run_thaw_consolidation_simulation(*args, **kwargs):
    raise NotImplementedError(
        "Module 5 (Thaw Consolidation Simulator) is scheduled for the next "
        "development phase. This stub preserves the API contract consumed by app.py."
    )

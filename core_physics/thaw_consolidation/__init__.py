"""
core_physics/thaw_consolidation/__init__.py
===============================================
Public API for Module 5 (Thaw Consolidation Simulator). See solver.py for
the pipeline docstring and each submodule for its own physics documentation.
"""

from .param import ConsolidationParams, DynamicConsolidationParams
from .boundaries import (
    extract_thaw_front_indices,
    thaw_front_depth_m,
    resample_front_to_solver_grid,
    active_thawed_mask,
    newly_thawed_mask,
)
from .physics import (
    overburden_effective_stress_profile,
    dynamic_cv,
    staggered_midpoint_average,
    void_ratio_from_effective_stress,
    settlement_from_void_ratio_history,
)
from .solver import (
    run_thaw_consolidation_simulation,
    build_sparse_diffusion_operator,
    evaluate_module5_status,
    render_thaw_consolidation_marquee,
    CRITICAL_SETTLEMENT_RATE_MM_DAY,
    WARNING_SETTLEMENT_RATE_MM_DAY,
)

__all__ = [
    "ConsolidationParams", "DynamicConsolidationParams",
    "extract_thaw_front_indices", "thaw_front_depth_m", "resample_front_to_solver_grid",
    "active_thawed_mask", "newly_thawed_mask",
    "overburden_effective_stress_profile", "dynamic_cv", "staggered_midpoint_average",
    "void_ratio_from_effective_stress", "settlement_from_void_ratio_history",
    "run_thaw_consolidation_simulation", "build_sparse_diffusion_operator",
    "evaluate_module5_status", "render_thaw_consolidation_marquee",
    "CRITICAL_SETTLEMENT_RATE_MM_DAY", "WARNING_SETTLEMENT_RATE_MM_DAY",
]
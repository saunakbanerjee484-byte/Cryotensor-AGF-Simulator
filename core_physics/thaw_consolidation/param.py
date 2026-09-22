"""
core_physics/thaw_consolidation/params.py
=============================================
Parameter dataclasses for Module 5 (Thaw Consolidation Simulator).

ConsolidationParams carries the soil's constitutive properties. Every field
may be a plain `float` (a homogeneous profile) OR a 1D `np.ndarray` of length
`n_nodes` (a stratified profile -- e.g. a stiffer sand crust over soft
thawing clay). This costs nothing extra: `scipy.sparse.diags` and all of
NumPy's elementwise arithmetic already broadcast a per-node array exactly as
readily as a scalar, so `solver.py` never needs to know or care which one it
received.

DynamicConsolidationParams carries the solver's discretization controls and
any time-varying external forcing (currently: a surcharge/embankment load
history q(t)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

import numpy as np

ScalarOrField = Union[float, np.ndarray]


@dataclass
class ConsolidationParams:
    """Soil constitutive properties -- scalar (homogeneous) or 1D array (stratified)."""
    compression_index_Cc: ScalarOrField = 0.35        # [-]   e-log(sigma') slope
    initial_void_ratio_e0: ScalarOrField = 0.85        # [-]   pre-thaw (frozen) void ratio
    coeff_consolidation_cv0: ScalarOrField = 1.2e-7    # [m2/s] reference cv, evaluated at e = e0
    permeability_index_Ck: ScalarOrField = 0.5         # [-]   e-log(k) slope (Ck in cv(e) relation)
    thaw_strain_alpha: float = 0.08                    # [-]   Morgenstern-Nixon thaw-strain coefficient
    effective_unit_weight_n_per_m3: float = 9000.0     # [N/m3] gamma' used for the overburden profile


@dataclass
class DynamicConsolidationParams:
    """Solver discretization controls and time-varying external forcing."""
    n_nodes: int = 41                    # depth nodes, z=0 (ground surface) .. domain_depth_m
    domain_depth_m: float = 3.0          # [m]
    dt_hours: float = 2.0                # [hours] Backward-Euler timestep
    total_days: float = 15.0             # [days] total simulated duration
    record_every: int = 4                # record a frame every N solver steps
    surcharge_history_pa: np.ndarray | None = field(default=None)
    # ^ optional q(t) [Pa], one value per SOLVER step (n_steps + 1 long); if None,
    #   treated as zero surcharge throughout (see solver.resolve_surcharge_at_step).
"""
core_physics/frost_heave.py
=============================
Sub-Module 3 (PLANNED): Volumetric Frost Heave Tensor.

Will convert the phase-change field (Module 1) and moisture migration
(Module 2) into an exact volumetric strain tensor, combining:

    1. In-situ 9% expansion of pore water converting to ice:
           eps_vol_phase = 0.09 * water_content * f_ice(T)

    2. Segregation heave from ice lens growth (secondary heave), driven by
       cryogenic suction gradient and the segregation potential (SP) model:
           dh_lens/dt = SP * grad(psi_cryo)

    Combined volumetric strain tensor (diagonal, isotropic assumption for v1):
           eps_ij = (eps_vol_phase + eps_vol_segregation) / 3 * delta_ij

Planned public API:
    class HeaveParams(dataclass): ...
    def primary_heave_strain(T_field, p) -> np.ndarray   # vectorized, 9% expansion term
    def segregation_heave_strain(suction_field, p) -> np.ndarray
    def run_frost_heave_simulation(T_history, suction_history, p) -> dict
"""

from dataclasses import dataclass


@dataclass
class HeaveParams:
    water_to_ice_expansion: float = 0.09   # [-] volumetric expansion fraction, water->ice
    segregation_potential: float = 2.0e-9  # [m2/s/K] SP model coefficient (typical clay/silt)


def run_frost_heave_simulation(*args, **kwargs):
    raise NotImplementedError(
        "Module 3 (Volumetric Frost Heave Tensor) is scheduled for the next "
        "development phase. This stub preserves the API contract consumed by app.py."
    )

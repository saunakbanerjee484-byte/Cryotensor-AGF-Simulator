"""
tests/test_mechanical.py
==========================
Contract tests for the not-yet-implemented mechanical/hydraulic sub-modules
(2: cryosuction, 3: frost_heave, 4: restrained_stress, 5: thaw_consolidation).

These modules are intentionally stubbed (see each file's module docstring)
so that Module 1 can be delivered and consumed by app.py without the later
physics being re-architected. This test file exists to:

    1. Guarantee the parameter dataclasses import cleanly and hold the
       documented default values (the "design basis" for each future model).
    2. Guarantee every stub raises NotImplementedError (not silently
       returning None / fabricated numbers) if called before its real
       physics lands -- this is a deliberate CPU-bound, deterministic
       engine with NO black-box fallbacks.
    3. Once each module is implemented, the corresponding
       `test_*_stub_raises` case should be DELETED and replaced with real
       physics assertions (energy/mass balance checks, monotonicity,
       bounds), following the pattern established in test_thermal.py.

Kept fast and dependency-light so it runs on every commit for the life of
the project.
"""

import pytest

from core_physics.cryosuction import SuctionParams, run_cryosuction_simulation
from core_physics.frost_heave import HeaveParams, run_frost_heave_simulation
from core_physics.restrained_stress import StressParams, run_restrained_stress_simulation
from core_physics.thaw_consolidation import ConsolidationParams, run_thaw_consolidation_simulation


# ----------------------------------------------------------------------------
# Module 2: SFCC Cryogenic Suction Solver
# ----------------------------------------------------------------------------
def test_suction_params_defaults_are_physically_sane():
    p = SuctionParams()
    assert 0.0 <= p.theta_r < p.theta_s <= 1.0
    assert p.van_genuchten_alpha > 0.0
    assert p.van_genuchten_n > 1.0          # required for a valid van Genuchten SFCC
    assert p.hydraulic_conductivity_sat > 0.0


def test_cryosuction_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        run_cryosuction_simulation()


# ----------------------------------------------------------------------------
# Module 3: Volumetric Frost Heave Tensor
# ----------------------------------------------------------------------------
def test_heave_params_expansion_matches_physical_constant():
    p = HeaveParams()
    # The water->ice volumetric expansion is a fixed physical constant (~9%);
    # guard against silent drift of this design assumption.
    assert p.water_to_ice_expansion == pytest.approx(0.09, abs=1e-6)
    assert p.segregation_potential > 0.0


def test_frost_heave_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        run_frost_heave_simulation()


# ----------------------------------------------------------------------------
# Module 4: Thermo-Elastic Restrained Stress
# ----------------------------------------------------------------------------
def test_stress_params_bounds():
    p = StressParams()
    assert p.youngs_modulus_frozen > 0.0
    assert 0.0 < p.poisson_ratio_frozen < 0.5   # physical bound for an isotropic elastic solid
    assert 0.0 <= p.restraint_factor_beta <= 1.0
    assert p.allowable_liner_pressure_mpa > 0.0


def test_restrained_stress_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        run_restrained_stress_simulation()


# ----------------------------------------------------------------------------
# Module 5: Thaw Consolidation Simulator
# ----------------------------------------------------------------------------
def test_consolidation_params_bounds():
    p = ConsolidationParams()
    assert p.compression_index_Cc > 0.0
    assert p.initial_void_ratio_e0 > 0.0
    assert p.coeff_consolidation_cv > 0.0
    assert p.thaw_strain_alpha > 0.0


def test_thaw_consolidation_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        run_thaw_consolidation_simulation()

"""
tests/test_mechanical.py
==========================
Contract tests for the not-yet-implemented mechanical sub-modules
(4: restrained_stress, 5: thaw_consolidation).

Modules 2 (cryosuction) and 3 (frost_heave) have graduated out of this file --
see tests/test_cryosuction.py and tests/test_frost_heave.py for their real
physics assertions, following the pattern established in test_thermal.py.

These two remaining modules are intentionally stubbed (see each file's module
docstring) so that Modules 1-3 can be delivered and consumed by app.py without
the later physics being re-architected. This test file exists to:

    1. Guarantee the parameter dataclasses import cleanly and hold the
       documented default values (the "design basis" for each future model).
    2. Guarantee every stub raises NotImplementedError (not silently
       returning None / fabricated numbers) if called before its real
       physics lands -- this is a deliberate CPU-bound, deterministic
       engine with NO black-box fallbacks.
    3. Once each module is implemented, the corresponding
       `test_*_stub_raises` case should be DELETED (and this file's own
       docstring updated) following the same graduation pattern.

Kept fast and dependency-light so it runs on every commit for the life of
the project.
"""

import pytest

from core_physics.restrained_stress import StressParams, run_restrained_stress_simulation
from core_physics.thaw_consolidation import ConsolidationParams, run_thaw_consolidation_simulation


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

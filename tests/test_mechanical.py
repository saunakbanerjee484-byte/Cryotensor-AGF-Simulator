"""
tests/test_mechanical.py
==========================
Contract test for the not-yet-implemented mechanical sub-module (5: thaw
consolidation).

Modules 2 (cryosuction), 3 (frost_heave), and 4 (restrained_stress) have all
graduated out of this file -- see tests/test_cryosuction.py,
tests/test_frost_heave.py, and tests/test_restrained_stress.py for their real
physics assertions, following the pattern established in test_thermal.py.

Module 5 is intentionally stubbed (see its module docstring) so that Modules
1-4 can be delivered and consumed by app.py without the later physics being
re-architected. This test file exists to:

    1. Guarantee the parameter dataclass imports cleanly and holds the
       documented default values (the "design basis" for the future model).
    2. Guarantee the stub raises NotImplementedError (not silently
       returning None / fabricated numbers) if called before its real
       physics lands -- this is a deliberate CPU-bound, deterministic
       engine with NO black-box fallbacks.
    3. Once implemented, `test_thaw_consolidation_stub_raises_not_implemented`
       should be DELETED (and this file's own docstring updated) following
       the same graduation pattern.

Kept fast and dependency-light so it runs on every commit for the life of
the project.
"""

import pytest

from core_physics.thaw_consolidation import ConsolidationParams, run_thaw_consolidation_simulation


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

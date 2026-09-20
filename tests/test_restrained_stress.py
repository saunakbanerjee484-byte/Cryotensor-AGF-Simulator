"""
tests/test_restrained_stress.py
==================================
Regression tests for Module 4 (Thermo-Elastic Restrained Stress & THMC
Degradation), following the pattern established in test_thermal.py and
test_frost_heave.py: small grids, fast, deterministic. Covers 4B.1
(restrained thermo-elastic core), 4B.2 (J2-invariant, temperature-coupled
visco-plastic creep), and 4B.3 (unified continuum damage + hydro-chemical
degradation), including the 4C.1 physics toggles.
"""

import numpy as np
import pytest

from core_physics.restrained_stress import (
    StressParams,
    soil_plane_strain_modulus, baseline_restraint_factor, restraint_factor_field,
    damage_field_mechanical, von_mises_equivalent_stress, creep_rate_coefficient_field,
    run_creep_time_march, build_sparse_dx_operator, advective_thmc_diagnostics,
    run_restrained_stress_simulation, generate_mock_strain_field,
    generate_mock_temperature_field, evaluate_module4_status,
)


def test_stress_params_defaults_are_physically_sane():
    p = StressParams()
    assert p.youngs_modulus_frozen > 0.0
    assert 0.0 < p.poisson_ratio_frozen < 0.5
    assert p.concrete_yield_strength_mpa > 0.0
    assert p.creep_n >= 1.0
    assert p.seepage_vx >= 0.0
    assert p.initial_salinity_ppt >= 0.0
    assert p.enable_creep and p.enable_advective_seepage
    assert p.enable_solute_rejection and p.enable_damage


def test_soil_plane_strain_modulus_positive():
    K = soil_plane_strain_modulus(45.0e6, 0.35)
    assert K > 0.0 and np.isfinite(K)


def test_baseline_restraint_factor_in_unit_interval():
    p = StressParams()
    beta_max = baseline_restraint_factor(p)
    assert 0.0 < beta_max < 1.0


def test_restraint_factor_field_decays_away_from_wall():
    p = StressParams()
    x = np.linspace(0.0, 3.0, 21)
    y = np.linspace(0.0, 3.0, 21)
    beta = restraint_factor_field(x, y, p)
    assert beta.shape == (21, 21)
    assert beta[:, -1].mean() > beta[:, 0].mean()


def test_von_mises_equivalent_stress_zero_for_zero_stress():
    z = np.zeros((5, 5))
    sigma_eq = von_mises_equivalent_stress(z, z, z, 0.3)
    assert np.allclose(sigma_eq, 0.0)


def test_von_mises_equivalent_stress_nonnegative_and_finite():
    rng = np.random.default_rng(0)
    sxx, syy, txy = rng.uniform(-5, 5, (3, 6, 6))
    sigma_eq = von_mises_equivalent_stress(sxx, syy, txy, 0.3)
    assert np.all(sigma_eq >= 0.0)
    assert np.isfinite(sigma_eq).all()


def test_creep_rate_coefficient_field_decreases_with_colder_temperature():
    shape = (4, 4)
    A_isothermal = creep_rate_coefficient_field(None, shape)
    T_cold = np.full(shape, -20.0)
    T_warm = np.full(shape, -1.0)
    A_cold = creep_rate_coefficient_field(T_cold, shape)
    A_warm = creep_rate_coefficient_field(T_warm, shape)
    assert np.all(A_isothermal > 0.0)
    assert np.all(A_cold < A_warm)


def test_creep_time_march_finite_and_bounded():
    p = StressParams(concrete_yield_strength_mpa=0.5)
    x, y, eps_v = generate_mock_strain_field(nx=15, ny=15, peak_strain=5.0e-4)
    result = run_creep_time_march(eps_v, x, y, p)
    for key in ["sigma_xx_mpa", "sigma_1_mpa", "eps_vp_field"]:
        assert np.isfinite(result[key]).all()
    assert result["eps_vp_field"].min() >= 0.0
    assert np.all(result["eps_vp_field"] <= (eps_v / 3.0) + 1e-12)
    assert len(result["peak_sigma_eq_history_mpa"]) == 6


def test_creep_disabled_collapses_to_single_elastic_step():
    p = StressParams(enable_creep=False)
    x, y, eps_v = generate_mock_strain_field(nx=13, ny=13, peak_strain=5.0e-4)
    result = run_creep_time_march(eps_v, x, y, p)
    assert len(result["peak_sigma_eq_history_mpa"]) == 1
    assert np.all(result["eps_vp_field"] == 0.0)


def test_damage_field_mechanical_zero_below_limit_and_saturates_at_one():
    eps_v = np.array([0.0, 5.0e-5, 1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3])
    D = damage_field_mechanical(eps_v)
    assert D[0] == 0.0 and D[1] == 0.0
    assert D[-1] == 1.0 and D[-2] == 1.0
    assert np.all((D >= 0.0) & (D <= 1.0))
    assert np.all(np.diff(D) >= -1e-12)


def test_sparse_dx_operator_recovers_linear_gradient():
    n, dx = 25, 0.1
    Dx = build_sparse_dx_operator(n, dx)
    x = np.arange(n) * dx
    f = 3.0 * x + 7.0
    df = Dx @ f
    assert np.allclose(df, 3.0, atol=1e-8)


def test_advective_diagnostics_finite_and_shapes_match():
    p = StressParams()
    x = np.linspace(0.0, 3.0, 17)
    y = np.linspace(0.0, 3.0, 17)
    T_matrix = generate_mock_temperature_field(x, y)
    diag = advective_thmc_diagnostics(T_matrix, x, y, p)
    assert diag["adv_term_field"].shape == T_matrix.shape
    assert diag["D_thermal_field"].shape == T_matrix.shape
    assert np.isfinite(diag["adv_term_field"]).all()
    assert np.all((diag["D_thermal_field"] >= 0.0) & (diag["D_thermal_field"] <= 0.5))
    assert diag["solute_field_ppt"].min() >= p.initial_salinity_ppt - 1e-9


def test_advective_seepage_toggle_zeroes_adv_term():
    p = StressParams(enable_advective_seepage=False, enable_solute_rejection=False)
    x = np.linspace(0.0, 3.0, 11)
    y = np.linspace(0.0, 3.0, 11)
    T_matrix = generate_mock_temperature_field(x, y)
    diag = advective_thmc_diagnostics(T_matrix, x, y, p)
    assert np.all(diag["adv_term_field"] == 0.0)
    assert np.all(diag["D_thermal_field"] == 0.0)
    assert np.all(diag["solute_field_ppt"] == p.initial_salinity_ppt)


def test_run_restrained_stress_simulation_end_to_end_no_thermal_field():
    p = StressParams()
    x, y, eps_v = generate_mock_strain_field(nx=21, ny=21)
    res = run_restrained_stress_simulation(eps_v, x, y, p)
    assert res["advective"] is None
    assert np.isfinite(res["sigma_1_mpa"]).all()
    assert res["max_lateral_pressure_mpa"] >= 0.0
    assert 0.0 <= res["max_damage"] <= 1.0
    assert res["wall_face_pressure_mpa"].shape == (21,)


def test_run_restrained_stress_simulation_end_to_end_with_thermal_field():
    p = StressParams()
    x, y, eps_v = generate_mock_strain_field(nx=17, ny=17)
    T_matrix = generate_mock_temperature_field(x, y)
    res = run_restrained_stress_simulation(eps_v, x, y, p, T_matrix=T_matrix)
    assert res["advective"] is not None
    assert res["seepage_erosion_index"] >= 0.0
    assert np.isfinite(res["D_field"]).all()


def test_all_toggles_off_collapses_to_pure_restrained_elasticity():
    p = StressParams(enable_creep=False, enable_advective_seepage=False,
                      enable_solute_rejection=False, enable_damage=False)
    x, y, eps_v = generate_mock_strain_field(nx=15, ny=15, peak_strain=8.0e-4)
    T_matrix = generate_mock_temperature_field(x, y)
    res = run_restrained_stress_simulation(eps_v, x, y, p, T_matrix=T_matrix)
    assert res["max_damage"] == 0.0
    assert res["creep_active_fraction"] == 0.0
    assert res["seepage_erosion_index"] == 0.0
    assert np.isfinite(res["sigma_1_mpa"]).all()


def test_damage_reduces_peak_pressure_relative_to_pure_elasticity():
    x, y, eps_v = generate_mock_strain_field(nx=19, ny=19, peak_strain=8.0e-4)
    p_pure = StressParams(enable_creep=False, enable_advective_seepage=False,
                           enable_solute_rejection=False, enable_damage=False)
    p_damaged = StressParams(enable_creep=False, enable_advective_seepage=False,
                              enable_solute_rejection=False, enable_damage=True)
    res_pure = run_restrained_stress_simulation(eps_v, x, y, p_pure)
    res_damaged = run_restrained_stress_simulation(eps_v, x, y, p_damaged)
    assert res_damaged["max_lateral_pressure_mpa"] <= res_pure["max_lateral_pressure_mpa"]


def test_evaluate_module4_status_all_three_states_reachable():
    red_status, red_msg = evaluate_module4_status(
        max_lateral_pressure_mpa=10.0, yield_mpa=1.0, max_damage=0.1,
        creep_active_fraction=0.0, seepage_erosion_index=0.0,
    )
    assert red_status == "red" and "CRITICAL YIELD" in red_msg

    red_via_damage_status, _ = evaluate_module4_status(
        max_lateral_pressure_mpa=0.01, yield_mpa=100.0, max_damage=0.9,
        creep_active_fraction=0.0, seepage_erosion_index=0.0,
    )
    assert red_via_damage_status == "red"

    yellow_status, yellow_msg = evaluate_module4_status(
        max_lateral_pressure_mpa=0.01, yield_mpa=100.0, max_damage=0.1,
        creep_active_fraction=0.05, seepage_erosion_index=0.0,
    )
    assert yellow_status == "yellow" and "MODERATE" in yellow_msg

    green_status, green_msg = evaluate_module4_status(
        max_lateral_pressure_mpa=0.01, yield_mpa=100.0, max_damage=0.0,
        creep_active_fraction=0.0, seepage_erosion_index=0.0,
    )
    assert green_status == "green" and "SAFE" in green_msg

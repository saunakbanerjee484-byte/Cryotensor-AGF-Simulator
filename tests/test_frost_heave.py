"""
tests/test_frost_heave.py
============================
Regression tests for Module 3 (Volumetric Frost Heave Tensor), following the
pattern established in test_thermal.py: small grids, fast, deterministic.
"""

import numpy as np
import pytest

from core_physics.frost_heave import (
    HeaveParams,
    identify_freezing_fringe, temperature_gradient_field, phase_front_velocity_field,
    segregation_potential, compute_heave_rate_field, compute_volumetric_strain_field,
    compute_surface_uplift_profile, run_frost_heave_simulation,
    generate_mock_temperature_fields, evaluate_module3_status,
)


def test_identify_freezing_fringe_masks_correct_band():
    T = np.array([[1.0, -0.2], [-0.4, -1.0]])
    mask = identify_freezing_fringe(T, T_upper=0.0, T_lower=-0.5)
    expected = np.array([[False, True], [True, False]])
    assert np.array_equal(mask, expected)


def test_temperature_gradient_field_zero_for_uniform_field():
    T = np.full((10, 10), -2.0)
    grad_mag, dTdx, dTdy = temperature_gradient_field(T, dx=0.1, dy=0.1)
    assert np.allclose(grad_mag, 0.0)


def test_phase_front_velocity_zero_when_temperature_unchanged():
    T = np.linspace(-5, 5, 100).reshape(10, 10)
    v = phase_front_velocity_field(T, T, dt_s=3600.0, dx=0.1, dy=0.1)
    assert np.allclose(v, 0.0)


def test_segregation_potential_decays_with_overburden_pressure():
    p_low = HeaveParams(overburden_pressure_kpa=0.0)
    p_high = HeaveParams(overburden_pressure_kpa=200.0)
    SP_low = segregation_potential(p_low)
    SP_high = segregation_potential(p_high)
    assert SP_low > SP_high > 0.0
    # exp(-a*0) = 1 -> SP at zero overburden should equal SP0 (in SI units)
    assert SP_low == pytest.approx(p_low.sp0_si, rel=1e-9)


def test_heave_rate_field_is_zero_outside_fringe():
    x, y, T_prev, T_curr, dt_s = generate_mock_temperature_fields(nx=21, ny=21)
    p = HeaveParams()
    h_dot, fringe, *_ = compute_heave_rate_field(T_curr, T_prev, dt_s, x[1] - x[0], y[1] - y[0], p)
    assert np.all(h_dot[~fringe] == 0.0)


def test_heave_rate_field_finite_and_nonnegative():
    x, y, T_prev, T_curr, dt_s = generate_mock_temperature_fields(nx=21, ny=21)
    p = HeaveParams()
    h_dot, *_ = compute_heave_rate_field(T_curr, T_prev, dt_s, x[1] - x[0], y[1] - y[0], p)
    assert np.all(np.isfinite(h_dot))
    assert np.all(h_dot >= 0.0)   # heave is a one-directional (expansive) process


def test_volumetric_strain_field_scales_linearly_with_dt():
    h_dot = np.full((5, 5), 1.0e-8)
    strain_1 = compute_volumetric_strain_field(h_dot, dt_s=3600.0, dy=0.1)
    strain_2 = compute_volumetric_strain_field(h_dot, dt_s=7200.0, dy=0.1)
    assert np.allclose(strain_2, 2.0 * strain_1)


def test_surface_uplift_profile_shape_matches_x_axis():
    x, y, T_prev, T_curr, dt_s = generate_mock_temperature_fields(nx=21, ny=25)
    p = HeaveParams()
    h_dot, *_ = compute_heave_rate_field(T_curr, T_prev, dt_s, x[1] - x[0], y[1] - y[0], p)
    profile = compute_surface_uplift_profile(h_dot, y[1] - y[0])
    assert profile.shape == (21,)   # one value per x-column
    assert np.all(profile >= 0.0)


def test_run_frost_heave_simulation_end_to_end_small_grid():
    x, y, T_prev, T_curr, dt_s = generate_mock_temperature_fields(nx=21, ny=21)
    p = HeaveParams()
    result = run_frost_heave_simulation(T_curr, T_prev, dt_s, x, y, p)

    assert np.all(np.isfinite(result["h_dot_field_mps"]))
    assert np.all(np.isfinite(result["strain_field"]))
    assert result["max_heave_rate_mm_day"] >= 0.0
    assert result["h_dot_field_mm_day"].shape == T_curr.shape


def test_evaluate_module3_status_all_three_states_reachable():
    status_g, _ = evaluate_module3_status(max_heave_rate_mm_day=0.2, tolerance_mm_day=2.0)
    assert status_g == "green"

    status_y, _ = evaluate_module3_status(max_heave_rate_mm_day=1.2, tolerance_mm_day=2.0)
    assert status_y == "yellow"

    status_r, _ = evaluate_module3_status(max_heave_rate_mm_day=2.5, tolerance_mm_day=2.0)
    assert status_r == "red"

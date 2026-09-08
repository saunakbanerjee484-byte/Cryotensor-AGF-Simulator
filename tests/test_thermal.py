"""
tests/test_thermal.py
=======================
Regression tests for core_physics/phase_change.py. Kept fast (small grids,
short duration) so this suite can run on every commit over the project's
lifetime without needing a GPU or long wall-clock time.
"""

import numpy as np
import pytest

from core_physics.phase_change import (
    ThermalParams,
    liquid_fraction,
    dliquid_fraction_dT,
    apparent_heat_capacity,
    thermal_conductivity,
    build_2d_laplacian,
    run_stefan_simulation,
)


def test_liquid_fraction_bounds():
    T = np.array([-10.0, -0.3, 0.0, 0.3, 10.0])
    fl = liquid_fraction(T, T_freeze=0.0, delta_T=0.6)
    assert np.all(fl >= 0.0) and np.all(fl <= 1.0)
    assert fl[0] == 0.0          # fully frozen far below band
    assert fl[-1] == 1.0         # fully unfrozen far above band
    assert fl[2] == pytest.approx(0.5, abs=1e-9)  # at T_freeze, midpoint of band


def test_liquid_fraction_monotonic():
    T = np.linspace(-5, 5, 200)
    fl = liquid_fraction(T, T_freeze=0.0, delta_T=0.6)
    assert np.all(np.diff(fl) >= -1e-12)  # non-decreasing


def test_dliquid_fraction_dT_zero_outside_band():
    T = np.array([-10.0, 10.0])
    dfl = dliquid_fraction_dT(T, T_freeze=0.0, delta_T=0.6)
    assert np.allclose(dfl, 0.0)


def test_apparent_heat_capacity_peaks_in_band():
    p = ThermalParams()
    T_band = np.array([0.0])          # center of phase band
    T_outside = np.array([-10.0])
    c_band = apparent_heat_capacity(T_band, p)
    c_outside = apparent_heat_capacity(T_outside, p)
    assert c_band[0] > c_outside[0]   # latent heat spike inside the band


def test_thermal_conductivity_bounds():
    p = ThermalParams()
    T = np.array([-50.0, 0.0, 50.0])
    k = thermal_conductivity(T, p)
    assert np.isclose(k[0], p.k_frozen, atol=1e-6)
    assert np.isclose(k[-1], p.k_unfrozen, atol=1e-6)


def test_laplacian_shape_and_symmetry():
    nx, ny = 5, 4
    L = build_2d_laplacian(nx, ny, dx=0.1, dy=0.1)
    assert L.shape == (nx * ny, nx * ny)
    # Discrete Laplacian (Neumann-flavored) should be symmetric
    diff = (L - L.T)
    assert np.max(np.abs(diff.toarray())) < 1e-9


def test_run_stefan_simulation_small_grid_energy_direction():
    """
    Sanity/regression test: on a small grid with a cold pipe embedded in warm
    ground, the domain must cool monotonically overall (mean temperature at
    final frame <= mean temperature at initial frame) and no NaNs/Infs appear.
    """
    p = ThermalParams(
        Lx=1.0, Ly=1.0, nx=21, ny=21,
        pipe_radius=0.05, pipe_cx=0.5, pipe_cy=0.5, pipe_temp=-20.0,
        initial_ground_temp=10.0, far_field_temp=10.0,
        dt=1800.0, total_time=1800.0 * 20, picard_iters=3,
    )
    result = run_stefan_simulation(p, record_every=5)

    T0 = result["T_history"][0]
    Tf = result["T_history"][-1]

    assert np.all(np.isfinite(Tf))
    assert np.mean(Tf) <= np.mean(T0) + 1e-9
    # Freeze front radius should be non-negative and finite for every recorded frame
    assert np.all(result["freeze_front_radius_m"] >= 0.0)
    assert np.all(np.isfinite(result["freeze_front_radius_m"]))
    # Frozen fraction should be non-decreasing on this short, monotonic cooling run
    assert np.all(np.diff(result["frozen_fraction"]) >= -1e-9)


def test_pipe_boundary_stays_at_coolant_temperature():
    p = ThermalParams(
        Lx=1.0, Ly=1.0, nx=21, ny=21,
        pipe_radius=0.06, pipe_cx=0.5, pipe_cy=0.5, pipe_temp=-18.0,
        initial_ground_temp=8.0, far_field_temp=8.0,
        dt=1800.0, total_time=1800.0 * 6, picard_iters=3,
    )
    result = run_stefan_simulation(p, record_every=1)
    X, Y = result["X"], result["Y"]
    pipe_mask = (X - p.pipe_cx) ** 2 + (Y - p.pipe_cy) ** 2 <= p.pipe_radius ** 2
    T_final = result["T_final"]
    assert np.allclose(T_final[pipe_mask], p.pipe_temp, atol=1e-6)

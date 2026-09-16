"""
tests/test_cryosuction.py
============================
Regression tests for Module 2 (SFCC Cryogenic Suction Solver), following the
pattern established in test_thermal.py: small grids, short durations, fast.
"""

import numpy as np
import pytest

from core_physics.cryosuction import (
    SFCCParams, BoundaryParams, SolverParams,
    cryogenic_suction_kpa, van_genuchten_theta, psi_from_theta,
    specific_moisture_capacity, hydraulic_conductivity, diffusivity,
    run_cryosuction_simulation, evaluate_module2_status,
)


def test_cryogenic_suction_zero_above_freezing():
    T = np.array([0.0, 1.0, 5.0])
    assert np.all(cryogenic_suction_kpa(T) == 0.0)


def test_cryogenic_suction_matches_generalized_clausius_clapeyron_order_of_magnitude():
    # The generalized Clausius-Clapeyron slope for freezing-point depression of soil
    # water is a well-known geotechnical constant: ~1200 kPa of suction per degC.
    psi = cryogenic_suction_kpa(np.array([-1.0]))[0]
    assert 1100.0 < psi < 1300.0


def test_van_genuchten_psi_theta_round_trip():
    p = SFCCParams()
    psi_in = np.array([0.5, 5.0, 50.0, 500.0])
    theta = van_genuchten_theta(psi_in, p)
    psi_out = psi_from_theta(theta, p)
    # Below the Air Entry Value the curve is flat (theta = theta_s), so the round-trip
    # only holds in the active (psi > AEV) branch.
    active = psi_in > p.air_entry_kpa
    assert np.allclose(psi_in[active], psi_out[active], rtol=1e-3)


def test_van_genuchten_theta_bounds():
    p = SFCCParams()
    psi = np.linspace(0.0, 5000.0, 50)
    theta = van_genuchten_theta(psi, p)
    assert np.all(theta >= p.theta_r - 1e-9)
    assert np.all(theta <= p.theta_s + 1e-9)
    assert theta[0] == pytest.approx(p.theta_s, abs=1e-6)   # psi=0 -> fully saturated


def test_hydraulic_conductivity_monotonic_in_theta():
    p = SFCCParams()
    theta = np.linspace(p.theta_r + 0.01, p.theta_s, 20)
    K = hydraulic_conductivity(theta, p)
    assert np.all(np.diff(K) >= -1e-12)   # K should be non-decreasing with theta
    assert K[-1] == pytest.approx(p.k_sat, rel=1e-6)   # fully saturated -> K = k_sat


def test_diffusivity_finite_and_positive():
    p = SFCCParams()
    theta = np.linspace(p.theta_r + 0.01, p.theta_s, 20)
    D = diffusivity(theta, p)
    assert np.all(np.isfinite(D))
    assert np.all(D > 0.0)


def test_run_cryosuction_simulation_small_grid_stable():
    sfcc = SFCCParams()
    bnd = BoundaryParams()
    solver = SolverParams(n_nodes=15, dt_hours=3.0, total_days=2.0, picard_iters=4, record_every=3)

    result = run_cryosuction_simulation(sfcc, bnd, solver)

    theta_final = result["theta_history"][-1]
    psi_final = result["psi_history"][-1]

    assert np.all(np.isfinite(theta_final))
    assert np.all(np.isfinite(psi_final))
    assert np.all(theta_final >= sfcc.theta_r - 1e-6)
    assert np.all(theta_final <= sfcc.theta_s + 1e-6)
    # Bottom boundary (groundwater table) should remain saturated (Dirichlet theta_s).
    assert theta_final[-1] == pytest.approx(sfcc.theta_s, abs=1e-6)
    assert len(result["t"]) == len(result["theta_history"])


def test_evaluate_module2_status_all_three_states_reachable():
    # GREEN: no active suction gradient at all -> closed system.
    status_g, _ = evaluate_module2_status(mean_influx_velocity=2.0e-8, k_sat=2.0e-8, mean_suction_gradient=0.0)
    assert status_g == "green"

    # YELLOW: an active gradient present, but throttled by a low (capillary-barrier) k_sat.
    status_y, _ = evaluate_module2_status(mean_influx_velocity=4.0e-7, k_sat=1.0e-9, mean_suction_gradient=130.0)
    assert status_y == "yellow"

    # RED: sustained high flux -> continuous water-table feeding.
    status_r, _ = evaluate_module2_status(mean_influx_velocity=1.5e-5, k_sat=5.0e-8, mean_suction_gradient=7000.0)
    assert status_r == "red"

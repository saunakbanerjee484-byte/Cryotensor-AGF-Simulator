# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
tests/test_thaw_consolidation.py
===================================
Regression tests for Module 5 (Thaw Consolidation Simulator), following the
pattern established in test_thermal.py / test_frost_heave.py /
test_restrained_stress.py: small grids, fast, deterministic. Covers thaw-
front extraction (Step 1), the sparse diffusion operator (Step 2), the
Backward-Euler moving-boundary time march (Step 3), the vectorized
settlement integral (Step 4), and the three "essential physics" features
(stratified parameters, dynamic cv, dynamic surcharge).
"""

import numpy as np

from core_physics.thaw_consolidation import (
    ConsolidationParams, DynamicConsolidationParams,
    extract_thaw_front_indices, thaw_front_depth_m,
    active_thawed_mask, newly_thawed_mask,
    dynamic_cv, staggered_midpoint_average,
    void_ratio_from_effective_stress, settlement_from_void_ratio_history,
    run_thaw_consolidation_simulation, build_sparse_diffusion_operator,
    evaluate_module5_status,
)
from core_physics.thaw_consolidation.physics import instantaneous_thaw_void_ratio


def _make_mock_column(n_time: int = 25, n_y: int = 31, total_days: float = 15.0, front_rate: float = 0.3, T_thaw: float = 3.0, T_frozen: float = -5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(0.0, 3.0, n_y)
    t = np.linspace(0.0, total_days * 86400.0, n_time)
    front_depth_true = np.clip(front_rate * np.sqrt(t / 86400.0), 0.0, 3.0)
    T_col = np.where(y[None, :] <= front_depth_true[:, None], T_thaw, T_frozen)
    return t, y, T_col


def test_consolidation_params_accept_scalar_and_array() -> None:
    p_scalar = ConsolidationParams()
    p_array = ConsolidationParams(compression_index_Cc=np.array([0.1, 0.2, 0.3]))
    assert isinstance(p_scalar.compression_index_Cc, float)
    assert isinstance(p_array.compression_index_Cc, np.ndarray)


# ----------------------------------------------------------------------------
# Step 1: Vectorized thaw-front extraction
# ----------------------------------------------------------------------------
def test_extract_thaw_front_indices_grows_with_time() -> None:
    t, _, T_col = _make_mock_column()
    front_idx = extract_thaw_front_indices(T_col)
    assert front_idx.shape == (len(t),)
    assert front_idx[0] <= front_idx[-1]           # front advances (non-decreasing) over time
    assert np.all(np.diff(front_idx) >= 0)


def test_extract_thaw_front_indices_all_frozen_gives_front_at_surface():
    T_col = np.full((5, 10), -5.0)
    front_idx = extract_thaw_front_indices(T_col)
    assert np.all(front_idx == 0)


def test_extract_thaw_front_indices_all_thawed_gives_front_at_bottom():
    T_col = np.full((5, 10), 3.0)
    front_idx = extract_thaw_front_indices(T_col)
    assert np.all(front_idx == 9)


def test_thaw_front_depth_m_matches_indices():
    z = np.linspace(0, 3.0, 10)
    idx = np.array([0, 3, 9])
    depths = thaw_front_depth_m(idx, z)
    assert np.allclose(depths, z[idx])


def test_active_and_newly_thawed_masks():
    n = 10
    active = active_thawed_mask(3, n)
    assert active.sum() == 4              # indices 0,1,2,3
    newly = newly_thawed_mask(1, 3, n)
    assert newly.sum() == 2               # indices 2,3
    assert not newly[0] and not newly[1]


# ----------------------------------------------------------------------------
# Step 2: sparse diffusion operator, staggered averaging
# ----------------------------------------------------------------------------
def test_staggered_midpoint_average():
    arr = np.array([1.0, 3.0, 5.0, 9.0])
    mid = staggered_midpoint_average(arr)
    assert np.allclose(mid, [2.0, 4.0, 7.0])


def test_build_sparse_diffusion_operator_shape_and_symmetry():
    n = 15
    cv_mid = np.full(n - 1, 1.0e-7)
    dz = 0.1
    L = build_sparse_diffusion_operator(cv_mid, dz, n)
    assert L.shape == (n, n)
    dense = L.toarray()
    assert np.allclose(dense, dense.T)     # constant cv -> symmetric operator
    assert np.allclose(dense.sum(axis=1), 0.0, atol=1e-12)   # conservative (row sums to zero)


# ----------------------------------------------------------------------------
# Step 4: settlement integral
# ----------------------------------------------------------------------------
def test_settlement_from_void_ratio_history_zero_when_e_equals_e0():
    z = np.linspace(0, 3.0, 20)
    e0 = 0.85
    e_history = np.full((5, 20), e0)
    S = settlement_from_void_ratio_history(e_history, e0, z)
    assert np.allclose(S, 0.0)


def test_settlement_from_void_ratio_history_positive_when_e_below_e0():
    z = np.linspace(0, 3.0, 20)
    e0 = 0.85
    e_history = np.full((5, 20), 0.60)
    S = settlement_from_void_ratio_history(e_history, e0, z)
    expected = (e0 - 0.60) / (1.0 + e0) * 3.0
    assert np.allclose(S, expected, rtol=1e-6)


# ----------------------------------------------------------------------------
# Physics building blocks
# ----------------------------------------------------------------------------
def test_dynamic_cv_matches_cv0_when_e_equals_e0():
    e0 = np.full(10, 0.85)
    cv = dynamic_cv(e0, e0, 1.2e-7, 0.5)
    assert np.allclose(cv, 1.2e-7)


def test_dynamic_cv_decreases_as_void_ratio_collapses():
    e0 = np.full(10, 0.85)
    e_collapsed = np.full(10, 0.50)
    cv_e0 = dynamic_cv(e0, e0, 1.2e-7, 0.5)
    cv_collapsed = dynamic_cv(e_collapsed, e0, 1.2e-7, 0.5)
    assert np.all(cv_collapsed < cv_e0)    # permeability drops as soil densifies


def test_instantaneous_thaw_void_ratio_below_e0():
    e0 = np.full(5, 0.85)
    e_thaw = instantaneous_thaw_void_ratio(e0, alpha=0.08)
    assert np.all(e_thaw < e0)


def test_void_ratio_from_effective_stress_decreases_with_stress():
    sigma_ref = np.full(5, 100.0)
    e_thaw = np.full(5, 0.78)
    e_low = void_ratio_from_effective_stress(np.full(5, 100.0), sigma_ref, e_thaw, Cc=0.3)
    e_high = void_ratio_from_effective_stress(np.full(5, 10000.0), sigma_ref, e_thaw, Cc=0.3)
    assert np.allclose(e_low, e_thaw)      # at sigma_ref itself, no additional collapse
    assert np.all(e_high < e_low)          # higher effective stress -> more collapse


# ----------------------------------------------------------------------------
# End-to-end + the three essential physics features
# ----------------------------------------------------------------------------
def test_run_thaw_consolidation_simulation_end_to_end():
    t_m1, y_m1, T_col = _make_mock_column()
    p = ConsolidationParams()
    dyn = DynamicConsolidationParams(n_nodes=25, domain_depth_m=3.0, dt_hours=4.0,
                                      total_days=15.0, record_every=4)
    res = run_thaw_consolidation_simulation(T_col, t_m1, y_m1, p, dyn)
    assert np.isfinite(res["e_history"]).all()
    assert np.isfinite(res["u_history"]).all()
    assert res["e_history"].min() >= 0.30 - 1e-9      # residual void-ratio floor respected
    assert res["e_history"].max() <= p.initial_void_ratio_e0 + 1e-9
    assert np.all(res["u_history"] >= -1e-9)
    assert np.all(np.diff(res["settlement_m"]) >= -1e-9)   # monotonic settlement
    assert res["final_settlement_mm"] >= 0.0


def test_feature_stratified_params_changes_result():
    t_m1, y_m1, T_col = _make_mock_column()
    dyn = DynamicConsolidationParams(n_nodes=25, domain_depth_m=3.0, dt_hours=4.0,
                                      total_days=15.0, record_every=4)
    p_uniform = ConsolidationParams(compression_index_Cc=0.30)
    Cc_profile = np.where(np.linspace(0, 3.0, 25) < 1.0, 0.05, 0.5)
    p_strat = ConsolidationParams(compression_index_Cc=Cc_profile)

    res_uniform = run_thaw_consolidation_simulation(T_col, t_m1, y_m1, p_uniform, dyn)
    res_strat = run_thaw_consolidation_simulation(T_col, t_m1, y_m1, p_strat, dyn)
    assert np.isfinite(res_strat["e_history"]).all()
    assert res_strat["final_settlement_mm"] != res_uniform["final_settlement_mm"]


def test_feature_dynamic_surcharge_increases_settlement():
    t_m1, y_m1, T_col = _make_mock_column(front_rate=0.3)
    p = ConsolidationParams(compression_index_Cc=0.12, thaw_strain_alpha=0.03)
    dyn = DynamicConsolidationParams(n_nodes=25, domain_depth_m=3.0, dt_hours=4.0,
                                      total_days=15.0, record_every=4)
    n_steps = int(round(dyn.total_days * 86400.0 / (dyn.dt_hours * 3600.0)))
    surcharge = np.zeros(n_steps + 1)
    surcharge[n_steps // 2:] = 50_000.0
    dyn_sur = DynamicConsolidationParams(n_nodes=25, domain_depth_m=3.0, dt_hours=4.0,
                                          total_days=15.0, record_every=4,
                                          surcharge_history_pa=surcharge)

    res_base = run_thaw_consolidation_simulation(T_col, t_m1, y_m1, p, dyn)
    res_sur = run_thaw_consolidation_simulation(T_col, t_m1, y_m1, p, dyn_sur)
    assert np.isfinite(res_sur["u_history"]).all()
    assert res_sur["final_settlement_mm"] > res_base["final_settlement_mm"]


def test_evaluate_module5_status_all_three_states_reachable():
    red_status, red_msg = evaluate_module5_status(50.0)
    assert red_status == "red" and "CRITICAL" in red_msg

    yellow_status, yellow_msg = evaluate_module5_status(12.0)
    assert yellow_status == "yellow" and "MODERATE" in yellow_msg

    green_status, green_msg = evaluate_module5_status(1.0)
    assert green_status == "green" and "STABLE" in green_msg
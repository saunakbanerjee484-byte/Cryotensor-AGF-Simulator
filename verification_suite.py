#!/usr/bin/env python3
"""
Reproducible verification and stress-test suite for CryoTensor-AGF-Simulator.

Runs nominal physics/regression scenarios and boundary/stress scenarios that
exercise numerical guardrails and failure envelopes. Produces a JSON report and
a concise console summary.

Usage:
    python verification_suite.py
    python verification_suite.py --json verification_report.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_physics.phase_change import ThermalParams, run_stefan_simulation
from core_physics.cryosuction import (
    SFCCParams, BoundaryParams, SolverParams, run_cryosuction_simulation,
)
from core_physics.frost_heave import HeaveParams, run_frost_heave_simulation, generate_mock_temperature_fields
from core_physics.restrained_stress import (
    StressParams, run_restrained_stress_simulation, generate_mock_strain_field, generate_mock_temperature_field,
)
from core_physics.thaw_consolidation import (
    ConsolidationParams, DynamicConsolidationParams, run_thaw_consolidation_simulation,
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


def _finite_stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr)
    finite = np.isfinite(a)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite_fraction": float(finite.mean()) if a.size else 1.0,
        "min": float(np.nanmin(a)) if a.size else None,
        "max": float(np.nanmax(a)) if a.size else None,
        "mean": float(np.nanmean(a)) if a.size else None,
    }


def _case(name: str, category: str, fn: Callable[[], dict[str, Any]], expect: str = "pass") -> dict[str, Any]:
    t0 = time.perf_counter()
    record: dict[str, Any] = {
        "name": name,
        "category": category,
        "expect": expect,
        "status": "fail",
        "duration_s": None,
    }
    try:
        details = fn()
        record["status"] = "pass"
        record["details"] = details
    except Exception as exc:  # noqa: BLE001 - verification harness records failures
        record["status"] = "expected_fail" if expect == "fail" else "fail"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
        record["traceback_tail"] = traceback.format_exc().splitlines()[-5:]
    record["duration_s"] = round(time.perf_counter() - t0, 4)
    return record


def nominal_phase_change() -> dict[str, Any]:
    p = ThermalParams(Lx=1.0, Ly=1.0, nx=25, ny=25, dt=1800.0, total_time=1800.0 * 24, picard_iters=3)
    r = run_stefan_simulation(p, record_every=6)
    T_final = r["T_final"]
    assert np.isfinite(T_final).all()
    assert float(np.mean(T_final)) <= float(np.mean(r["T_history"][0])) + 1e-9
    return {"T_final": _finite_stats(T_final), "frozen_fraction_final": float(r["frozen_fraction"][-1])}


def nominal_cryosuction() -> dict[str, Any]:
    r = run_cryosuction_simulation(SFCCParams(), BoundaryParams(), SolverParams(n_nodes=21, dt_hours=2.0, total_days=4.0, picard_iters=4, record_every=4))
    th = r["theta_history"][-1]
    ps = r["psi_history"][-1]
    assert np.isfinite(th).all() and np.isfinite(ps).all()
    assert th.min() >= r["params"]["sfcc"].theta_r - 1e-6
    assert th.max() <= r["params"]["sfcc"].theta_s + 1e-6
    return {"theta_final": _finite_stats(th), "psi_final": _finite_stats(ps), "max_influx_mps": float(np.max(r["influx_velocity"]))}


def nominal_frost_heave() -> dict[str, Any]:
    x, y, Tp, Tc, dt_s = generate_mock_temperature_fields(nx=31, ny=31)
    r = run_frost_heave_simulation(Tc, Tp, dt_s, x, y, HeaveParams())
    for k in ["h_dot_field_mps", "strain_field", "uplift_rate_profile_mm_day"]:
        assert np.isfinite(np.asarray(r[k])).all()
    assert r["max_heave_rate_mm_day"] >= 0.0
    return {"max_heave_rate_mm_day": float(r["max_heave_rate_mm_day"]), "fringe_cells": int(r["fringe_mask"].sum())}


def nominal_restrained_stress() -> dict[str, Any]:
    x, y, eps = generate_mock_strain_field(nx=21, ny=21)
    Tm = generate_mock_temperature_field(x, y)
    r = run_restrained_stress_simulation(eps, x, y, StressParams(), T_matrix=Tm)
    for k in ["sigma_xx_mpa", "sigma_1_mpa", "D_field"]:
        assert np.isfinite(np.asarray(r[k])).all()
    assert 0.0 <= r["max_damage"] <= 1.0
    return {"max_lateral_pressure_mpa": float(r["max_lateral_pressure_mpa"]), "max_damage": float(r["max_damage"])}


def nominal_thaw_consolidation() -> dict[str, Any]:
    y = np.linspace(0.0, 3.0, 31)
    t = np.linspace(0.0, 15.0 * 86400.0, 25)
    front = np.clip(0.35 * np.sqrt(t / 86400.0), 0.0, 3.0)
    T_col = np.where(y[None, :] <= front[:, None], 3.0, -8.0)
    p = ConsolidationParams()
    dyn = DynamicConsolidationParams(n_nodes=21, domain_depth_m=3.0, dt_hours=4.0, total_days=15.0, record_every=4)
    r = run_thaw_consolidation_simulation(T_col, t, y, p, dyn)
    assert np.isfinite(r["e_history"]).all()
    assert np.all(np.diff(r["settlement_m"]) >= -1e-9)
    return {"final_settlement_mm": float(r["final_settlement_mm"]), "max_settlement_rate_mm_day": float(r["max_settlement_rate_mm_day"])}


def stress_phase_change_extreme_dt() -> dict[str, Any]:
    p = ThermalParams(Lx=1.0, Ly=1.0, nx=21, ny=21, dt=3600.0 * 24, total_time=3600.0 * 24 * 10, picard_iters=2)
    r = run_stefan_simulation(p, record_every=2)
    assert np.isfinite(r["T_final"]).all()
    return {"T_final": _finite_stats(r["T_final"]), "note": "Large timestep remains finite due to implicit Backward Euler."}


def stress_phase_change_invalid_dt() -> dict[str, Any]:
    p = ThermalParams(dt=0.0)
    run_stefan_simulation(p)
    return {}


def stress_cryosuction_invalid_params() -> dict[str, Any]:
    SFCCParams(vg_n=1.0)
    return {}


def stress_frost_heave_zero_dt() -> dict[str, Any]:
    x, y, Tp, Tc, _ = generate_mock_temperature_fields(nx=11, ny=11)
    run_frost_heave_simulation(Tc, Tp, 0.0, x, y, HeaveParams())
    return {}


def stress_restrained_stress_singular_nu() -> dict[str, Any]:
    x, y, eps = generate_mock_strain_field(nx=11, ny=11)
    run_restrained_stress_simulation(eps, x, y, StressParams(poisson_ratio_frozen=0.5))
    return {}


def stress_thaw_invalid_dt() -> dict[str, Any]:
    y = np.linspace(0, 1, 10)
    t = np.linspace(0, 86400, 5)
    T = np.zeros((5, 10))
    run_thaw_consolidation_simulation(T, t, y, ConsolidationParams(), DynamicConsolidationParams(n_nodes=10, domain_depth_m=1.0, dt_hours=0.0, total_days=1.0))
    return {}


def stress_thaw_high_surcharge() -> dict[str, Any]:
    y = np.linspace(0.0, 3.0, 21)
    t = np.linspace(0.0, 10.0 * 86400.0, 20)
    front = np.clip(0.4 * np.sqrt(t / 86400.0), 0.0, 3.0)
    T_col = np.where(y[None, :] <= front[:, None], 2.0, -5.0)
    n_steps = int(round(10.0 * 24 / 4.0))
    q = np.zeros(n_steps + 1); q[n_steps // 3:] = 500_000.0
    dyn = DynamicConsolidationParams(n_nodes=21, domain_depth_m=3.0, dt_hours=4.0, total_days=10.0, record_every=3, surcharge_history_pa=q)
    r = run_thaw_consolidation_simulation(T_col, t, y, ConsolidationParams(), dyn)
    assert np.isfinite(r["e_history"]).all()
    assert r["final_settlement_mm"] >= 0
    return {"final_settlement_mm": float(r["final_settlement_mm"]), "max_u_kpa": float(np.max(r["u_history"]) / 1000.0)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()

    cases = [
        _case("nominal_phase_change", "nominal", nominal_phase_change),
        _case("nominal_cryosuction", "nominal", nominal_cryosuction),
        _case("nominal_frost_heave", "nominal", nominal_frost_heave),
        _case("nominal_restrained_stress", "nominal", nominal_restrained_stress),
        _case("nominal_thaw_consolidation", "nominal", nominal_thaw_consolidation),
        _case("stress_phase_change_extreme_dt", "stress", stress_phase_change_extreme_dt),
        _case("stress_phase_change_invalid_dt", "stress", stress_phase_change_invalid_dt, expect="fail"),
        _case("stress_cryosuction_invalid_params", "stress", stress_cryosuction_invalid_params, expect="fail"),
        _case("stress_frost_heave_zero_dt", "stress", stress_frost_heave_zero_dt, expect="fail"),
        _case("stress_restrained_stress_singular_nu", "stress", stress_restrained_stress_singular_nu, expect="fail"),
        _case("stress_thaw_invalid_dt", "stress", stress_thaw_invalid_dt, expect="fail"),
        _case("stress_thaw_high_surcharge", "stress", stress_thaw_high_surcharge),
    ]

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "versions": {
            "numpy": np.__version__,
        },
        "cases": cases,
    }
    try:
        import scipy, pandas, matplotlib, seaborn  # type: ignore
        report["versions"].update({
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
        })
    except Exception:
        pass

    summary = {
        "total": len(cases),
        "pass": sum(1 for c in cases if c["status"] == "pass"),
        "expected_fail": sum(1 for c in cases if c["status"] == "expected_fail"),
        "fail": sum(1 for c in cases if c["status"] == "fail"),
    }
    report["summary"] = summary

    print(json.dumps(summary, indent=2))
    for c in cases:
        print(f"[{c['status'].upper():13}] {c['category']:7} {c['name']} ({c['duration_s']}s)")
        if c["status"] == "fail":
            print(f"  error: {c.get('error_type')}: {c.get('error')}")

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
        print(f"Wrote {args.json_path}")

    return 0 if summary["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

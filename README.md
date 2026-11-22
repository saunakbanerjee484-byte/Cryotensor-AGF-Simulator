# CryoTensor-AGF-Simulator

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-62%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Verification](https://img.shields.io/badge/verification-nominal%20%2B%20stress%20suite-blue)](verification_suite.py)
[![Stack](https://img.shields.io/badge/stack-NumPy%20%7C%20SciPy%20%7C%20Streamlit%20%7C%20Matplotlib-111827)](requirements.txt)

**CryoTensor-AGF-Simulator** is a deterministic, CPU-bound thermo-hydro-mechanical (THM) simulator for Artificial Ground Freezing (AGF). It models transient Stefan phase change, SFCC-driven cryosuction, frost-heave strain generation, restrained structural stress with THMC degradation, and thaw consolidation settlement in a modular, auditable Python codebase.

> Scope boundary: this repository targets frozen-soil AGF mechanics and phase-change geotechnics. It is not a CFD, hydrodynamic, or machine-learning surrogate platform.

---

## Why This Repository Matters

- **Deterministic physics core:** classical PDE and constitutive models, no trained black-box predictors.
- **CPU-first numerics:** NumPy/SciPy sparse linear algebra; no GPU runtime required.
- **Modular AGF pipeline:** thermal field -> suction/moisture -> heave strain -> restrained stress -> thaw settlement.
- **Inspectable outputs:** Streamlit command-center UI, raw matrices/CSV exports, and rule-based engineering alerts.
- **Reproducible verification:** pytest regression suite plus a dedicated nominal/stress verification harness.

---

## Physics & Architecture Overview

| Module | Physical Model | Numerical Method | Status |
|---|---|---|---|
| 1. Transient Stefan Phase-Change | Apparent Heat Capacity Method (AHCM), latent heat smoothing, temperature-dependent `k(T), c_app(T)` | Implicit Backward Euler, 2D 5-point Laplacian via `scipy.sparse.kron`, Picard iteration | Implemented |
| 2. SFCC Cryosuction | van Genuchten SFCC, Clausius-Clapeyron cryogenic suction, Richards moisture migration | 1D implicit Backward Euler, Picard-linearized tridiagonal sparse solve | Implemented |
| 3. Volumetric Frost Heave | 9% in-situ phase expansion + Konrad-Morgenstern segregation potential in active fringe | Vectorized field operations, finite-difference front velocity proxy, strain/uplift integrals | Implemented |
| 4. Restrained Stress & THMC | Plane-strain thermo-elastic restraint, J2/von Mises creep driver, continuum damage, advective/solute coupling | Sparse block-diagonal constitutive contraction, sparse derivative operator, temporal creep sub-steps | Implemented |
| 5. Thaw Consolidation | Morgenstern-Nixon excess pore pressure generation, `cv(e)` dynamic consolidation, surcharge response | Moving thaw boundary, implicit Backward Euler sparse tridiagonal solve, trapezoidal settlement integral | Implemented |

### Repository Layout

```text
Cryotensor-AGF-Simulator/
├── app.py                          # Streamlit command center entry point
├── registry.py                     # Module registry / dispatch
├── state_pipeline.py               # Shared in-memory handoff between modules
├── verification_suite.py           # Reproducible nominal + stress verification harness
├── core_physics/
│   ├── phase_change.py             # Module 1 AHCM Stefan solver
│   ├── cryosuction.py              # Module 2 SFCC/Richards solver
│   ├── frost_heave.py              # Module 3 heave/strain tensor
│   ├── restrained_stress.py        # Module 4 THMC restrained stress
│   └── thaw_consolidation/         # Module 5 consolidation package
│       ├── param.py
│       ├── boundaries.py
│       ├── physics.py
│       └── solver.py
├── modules/                        # Streamlit presentation per module
├── ui_components/                  # Marquee, alerts, dual visualizer
└── tests/                          # Pytest regression suite
```

---

## Installation & Quickstart

### Prerequisites

- Python 3.10+ (verified on Python 3.14.5, Windows 11)
- `pip`

### Install

```powershell
git clone https://github.com/saunakbanerjee484-byte/Cryotensor-AGF-Simulator.git
cd Cryotensor-AGF-Simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run the Command Center

```powershell
streamlit run app.py
```

### Run Tests

```powershell
pip install pytest
python -m pytest tests -v
```

### Run Verification & Stress Suite

```powershell
python verification_suite.py --json verification_report.json
```

The verification suite executes nominal physics scenarios and invalid-input/stress-boundary scenarios, returning a non-zero exit code if any non-expected failure occurs.

---

## Engineering Design Principles

| Principle | Implementation |
|---|---|
| CPU-bound execution | NumPy/SciPy sparse operations; no PyTorch/CUDA/GPU dependencies. |
| Vectorized spatial operators | Sparse Laplacian/derivative assembly and NumPy broadcasting; no spatial Python node loops in core solvers. |
| Deterministic assessment | Rule-based red/yellow/green thresholds; no ML classifiers. |
| Separation of concerns | `core_physics/` has no Streamlit UI logic; `modules/` and `ui_components/` handle presentation. |
| Defensive numerical guards | Explicit validation for invalid timestep/grid/parameter domains; fail fast with `ValueError` instead of silent NaN/divergence. |

---

## Verification, Test Conditions & Failure Envelope

### Test Environment & Run Conditions

The verification evidence below was produced on **2026-11-22** in the local workspace environment.

| Item | Value |
|---|---|
| OS / Platform | Windows 11, `Windows-11-10.0.26300-SP0`, AMD64 |
| Python | `3.14.5 (MSC v.1944 64 bit)` |
| Git | `2.55.0.windows.2` |
| NumPy / SciPy | `2.4.6 / 1.18.0` |
| Pandas / Matplotlib / Seaborn / Streamlit | `3.0.3 / 3.11.0 / 0.13.2 / 1.60.0` |
| Pytest | `9.1.1` |
| Commands executed | `python -m pytest tests -v`; `python verification_suite.py --json verification_report.json` |

#### Nominal Verification Cases (`verification_suite.py`)

| Case | Key Inputs / Conditions | Pass Criteria | Observed Result |
|---|---|---|---|
| `nominal_phase_change` | `Lx=Ly=1.0 m`, `nx=ny=25`, `dt=1800 s`, `total=24 h`, `picard_iters=3`, coolant `-25 C`, ground `12 C` | Finite `T_final`; mean final temperature <= mean initial temperature | Pass; `frozen_fraction_final=0.4352` |
| `nominal_cryosuction` | `n_nodes=21`, `dt=2 h`, `total=4 days`, `picard_iters=4`, default van Genuchten/SFCC params | Finite `theta, psi`; `theta_r <= theta <= theta_s` | Pass; `max_influx_mps=1.835e-09` |
| `nominal_frost_heave` | Mock thermal pair `31x31`, `dt` from mock generator, default `HeaveParams` | Finite heave/strain/uplift; non-negative peak heave | Pass; `max_heave_rate_mm_day=0.01020`, `fringe_cells=212` |
| `nominal_restrained_stress` | Mock strain `21x21`, mock thermal field, default `StressParams` | Finite stresses/damage; `0 <= D <= 1` | Pass; `max_lateral_pressure_mpa=0.06298`, `max_damage=0.1310` |
| `nominal_thaw_consolidation` | `n_nodes=21`, `domain_depth=3 m`, `dt=4 h`, `total=15 days`, sqrt-time thaw column, no surcharge | Finite histories; settlement monotonic non-decreasing | Pass; `final_settlement_mm=134.78`, `max_settlement_rate_mm_day=30.84` |

#### Stress / Boundary Cases (`verification_suite.py`)

| Case | Condition | Expected Outcome | Observed Outcome | Scientific / Numerical Interpretation |
|---|---|---|---|---|
| `stress_phase_change_extreme_dt` | `dt=24 h`, `total=10 days`, coarse `21x21` grid, `picard_iters=2` | Pass (finite) | Pass | Backward Euler remains unconditionally stable for large timesteps; accuracy/coarsening risk increases, but no blow-up. |
| `stress_phase_change_invalid_dt` | `dt=0 s` | Controlled failure | `ValueError: dt must be > 0 seconds.` | Prevents singular time discretization and undefined transient update. |
| `stress_cryosuction_invalid_params` | `vg_n=1.0` | Controlled failure | `ValueError: vg_n must be > 1.0.` | van Genuchten parameterization requires `n>1` so `m=1-1/n` is physically/numerically valid. |
| `stress_frost_heave_zero_dt` | `dt_s=0` | Controlled failure | `ValueError: dt_s must be > 0 seconds.` | Front velocity is `|dT/dt|/|grad(T)|`; zero timestep makes rate undefined. |
| `stress_restrained_stress_singular_nu` | `nu=0.5` | Controlled failure | `ValueError: poisson ratio must satisfy -1 < nu < 0.5 for plane strain.` | Plane-strain constitutive denominator `(1+nu)(1-2nu)` becomes singular at `nu=0.5`. |
| `stress_thaw_invalid_dt` | `dt_hours=0` | Controlled failure | `ValueError: dt_hours must be > 0.` | Backward Euler consolidation update requires positive temporal discretization. |
| `stress_thaw_high_surcharge` | 500 kPa surcharge step at 1/3 duration, `dt=4 h`, 10 days | Pass (finite, physically bounded trend) | Pass; `final_settlement_mm=190.95`, `max_u_kpa=545.00` | Surcharge increases effective stress target and transient excess pore pressure; solver remains finite under aggressive loading. |

### Validated Operating Regime

- **Phase change (Module 1):** Stable on small-to-moderate grids (`21x21` to `25x25` in verification), implicit timesteps from hourly to daily, positive phase band width and physical thermal properties.
- **Cryosuction (Module 2):** Stable for valid van Genuchten/SFCC domains (`vg_n>1`, `0<=theta_r<theta_s`, positive `k_sat`, positive timestep), with bounded moisture content.
- **Frost heave (Module 3):** Stable for positive `dt`, positive grid spacing, and finite temperature fields; fringe-isolated heave remains finite/non-negative.
- **Restrained stress (Module 4):** Stable for admissible plane-strain Poisson ratio (`-1 < nu < 0.5`), finite strain/temperature inputs, and toggled THMC submodels.
- **Thaw consolidation (Module 5):** Stable for positive timestep/grid spacing, monotonic settlement accumulation in nominal run, and finite response under aggressive surcharge.

### Exact Failure Points & Limitations

The following are **validated guardrail failures** (intentional, controlled) or **model limitations**. They are documented to define the current reliability envelope.

1. **Invalid temporal discretization (`dt <= 0`)**  
   - **Affected modules:** 1, 3, 5 (and any solver requiring time stepping).  
   - **Observed behavior:** Immediate `ValueError` before numerical solve.  
   - **Root cause:** Time derivatives and implicit update matrices require positive `dt`; zero timestep creates division-by-zero/singular transient operators.

2. **Invalid SFCC parameterization (`vg_n <= 1`, `theta_r >= theta_s`, `k_sat <= 0`, `vg_alpha <= 0`)**  
   - **Affected module:** 2.  
   - **Observed behavior:** `ValueError` at parameter construction.  
   - **Root cause:** van Genuchten/Mualem relations become undefined or nonphysical outside admissible parameter domains; hydraulic conductivity/capacity lose monotonic meaning.

3. **Plane-strain constitutive singularity (`nu -> 0.5` or invalid Poisson ratio domain)**  
   - **Affected module:** 4.  
   - **Observed behavior:** `ValueError` for `nu=0.5`.  
   - **Root cause:** The plane-strain constitutive scaling includes `(1+nu)(1-2nu)` in the denominator; incompressible limit requires a different formulation.

4. **Grid/spacing degeneracy (`nx<2`, `ny<2`, `dx<=0`, `dy<=0`, `n_nodes<2`)**  
   - **Affected modules:** 1-5 where applicable.  
   - **Observed behavior:** `ValueError` guard failures.  
   - **Root cause:** Finite-difference stencils, sparse operators, and gradient estimates are undefined on degenerate grids.

5. **Model-form limitations (not code crashes, but scientific envelope limits)**  
   - Module 1 uses scalar-effective conductivity in Picard linearization for speed/stability; strongly heterogeneous conductivity contrasts may require nonlinear operator updates.  
   - Module 3 computes heave from two temperature frames and fringe masking; unresolved fringes on coarse grids can under-estimate local heave (UI emits warning).  
   - Module 4 uses isotropic scalar damage and plane-strain assumptions; anisotropic fracture and 3D confinement effects are outside scope.  
   - Module 5 is a 1D consolidation abstraction with moving thaw boundary; lateral drainage, 3D deformation, and fully coupled large-strain poromechanics are not represented.

### Verification Artifacts

- `tests/` - deterministic pytest regression suite (62 passing tests at verification time).
- `verification_suite.py` - nominal + stress/boundary harness with JSON report output.
- `verification_report.json` - machine-readable run record (generated locally when `--json` is supplied).

---

## Security & Reliability Notes

- No hardcoded secrets, credentials, or API keys were identified in repository source files during audit.
- No use of `eval`, `exec`, unsafe deserialization, or shell-command injection patterns was identified in core Python modules.
- The simulator performs in-memory numerical computation; it does not accept untrusted network payloads or execute external commands.
- Input validation guards were added to fail fast on invalid numerical domains (e.g., non-positive timesteps, singular constitutive parameters, invalid SFCC bounds) to prevent silent NaN propagation.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

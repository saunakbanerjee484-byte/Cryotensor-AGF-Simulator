# CryoTensor-AGF-Simulator

**Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine**

A standalone, deterministic, CPU-bound engineering simulator for Artificial
Ground Freezing (AGF) design: transient phase-change heat transfer,
cryogenic suction, frost heave, restrained structural stress, and thaw
consolidation. This project is **not** a surface-water hydrodynamics
platform — it is exclusively concerned with deep sub-surface solid
mechanics, phase-change thermodynamics, and frost heave.

---

## 1. Design Philosophy

| Constraint | How it is enforced |
|---|---|
| **CPU-bound, runs on a Core i3** | Pure NumPy / SciPy sparse linear algebra only. No PyTorch, no CUDA, no GPU code path anywhere in `core_physics/`. |
| **Pure vectorization** | Spatial operators (Laplacians, mixing laws, boundary masks) are assembled with `scipy.sparse.diags` / `scipy.sparse.kron` and NumPy broadcasting. There is **no Python `for` loop over spatial grid nodes** anywhere in the codebase. The only sequential loop is the outer transient time-march, which is physically unavoidable for an implicit scheme (each timestep depends on the previous one). |
| **No machine learning** | 100% deterministic classical PDEs and closed-form geotechnical equations (Fourier's Law, the Stefan condition, Richards' equation, thermo-elasticity, Morgenstern–Nixon thaw consolidation). No trained models, no learned thresholds. |
| **Separation of concerns** | `core_physics/` contains only math (dataclasses + vectorized functions) — zero `import streamlit`. `ui_components/` contains only Streamlit presentation helpers — zero physics. `app.py` wires the two together and holds no equations of its own. |

---

## 2. Architecture

```
CryoTensor-AGF-Simulator/
├── app.py                     # Streamlit Entry Point (The Command Center UI)
├── core_physics/              # CPU-bound, pure vectorized math — NO UI logic
│   ├── phase_change.py        # Module 1: Stefan formulation, Apparent Heat Capacity Method  [IMPLEMENTED]
│   ├── cryosuction.py         # Module 2: SFCC, moisture migration, ice lens growth           [stub / contract]
│   ├── frost_heave.py         # Module 3: 9% volumetric expansion tensor                      [stub / contract]
│   ├── restrained_stress.py   # Module 4: Thermo-elastic structural stress (MPa)               [stub / contract]
│   └── thaw_consolidation.py  # Module 5: Melt-down excess pore pressure & settlement           [stub / contract]
├── ui_components/             # Streamlit layouts & UX engine — NO physics
│   ├── marquee_engine.py      # "No Black Box" scrolling HTML/CSS live-equation banner
│   ├── alert_system.py        # Deterministic Red/Yellow/Green threshold logic
│   └── dual_visualizer.py     # Simultaneous Matplotlib line + Seaborn heatmap + DataFrame
└── tests/
    ├── test_thermal.py        # Regression tests for Module 1 physics
    └── test_mechanical.py     # Contract tests for Modules 2–5 (params + NotImplementedError guards)
```

Stub modules (2–5) are **not empty placeholders**: each one already defines
its final parameter dataclass (the design-basis constants an engineer would
need to specify) and its planned public function signatures, documented
inline with the governing equations. This means `app.py` and the UI layer
never need to be re-architected as each module lands — only the function
bodies change from `raise NotImplementedError(...)` to real vectorized
solvers.

---

## 3. Module 1 — Transient Stefan Phase-Change Matrix

**Status: fully implemented.**

### Governing physics

2D transient heat conduction with latent heat release, via the **Apparent
Heat Capacity Method (AHCM)**:

```
ρ · c_app(T) · ∂T/∂t = ∇·(k(T) · ∇T)
```

```
c_app(T) = c(T) + L · d(f_l)/dT              (apparent/effective heat capacity)

f_l(T)   = clip[ (T − (T_f − ΔT/2)) / ΔT , 0, 1 ]     (smoothed liquid/unfrozen fraction)

k(T)     = f_l · k_unfrozen + (1 − f_l) · k_frozen    (arithmetic conductivity mixing)
```

The phase-change (Stefan) condition is captured **without explicit front
tracking** by smoothing the latent-heat release over a narrow temperature
band `[T_f − ΔT/2, T_f + ΔT/2]` — the standard "apparent heat capacity" or
"equivalent heat capacity" formulation used in AGF engineering practice.
This avoids a moving-mesh or level-set front-tracking scheme entirely,
which is what keeps the solver cheap enough for a Core i3.
## 4. Module 2 — SFCC Cryogenic Suction Solver

**Status: fully implemented.**

### Governing physics
Couples the Soil Freezing Characteristic Curve (SFCC) to a 1D vertical Richards' equation (moisture-content / diffusive form) through the Generalized Clausius-Clapeyron relation.
*   **Cryogenic Suction:** `psi_cryo(T) = [rho_w * L_f / (T_f,K * 1000)] * max(T_f - T, 0)`
*   **SFCC:** van Genuchten formulation with an Air-Entry-Value (AEV) plateau.
*   **Moisture Migration:** Richards' equation tracking unfrozen-water migration from a groundwater source toward an advancing freezing front.

### Numerics
*   **CPU-Bound Vectorization:** The spatial diffusion operator is assembled ONCE per Picard sub-iteration as a tridiagonal `scipy.sparse.diags` matrix.
*   **No Spatial Loops:** Built from harmonic-mean nodal diffusivities via pure array slicing (`D[:-1]`, `D[1:]`) — absolutely no loop over depth nodes.
*   **Time Integration:** Backward-Euler (fully implicit) time integration for stability near saturation. Nonlinearity is resolved with a Picard fixed-point iteration (`scipy.sparse.linalg.spsolve`).

### Deterministic engineering assessment (Module 2)
*   🔴 **Red** — CRITICAL: Sustained moisture influx velocity continuous feeding from the water table. Massive ice lens formation in progress.
*   🟡 **Yellow** — MODERATE: Active suction gradient exists, but ice lens growth is restricted by a capillary barrier (low `k_sat`).
*   🟢 **Green** — OPTIMAL: Closed-system freezing confirmed. Minimal moisture migration.


## 5. Module 3 — Volumetric Frost Heave Tensor

**Status: Architecture defined, dynamic UI routing implemented.**

### Governing physics
Transitions the engine into a thermo-hydro-mechanical solver by calculating the total Volumetric Frost Heave Rate (`h_dot`) as a summation of two localized strain components:
*   **In-Situ Expansion:** `h_dot_in-situ = 0.09 * n * (dz/dt)` (9% volumetric expansion of existing pore water).
*   **Segregation Heave (Ice Lensing):** Utilizes the Konrad & Morgenstern Segregation Potential (SP) model where water intake flux is `v_s = SP * grad(T_f)`. 
*   **Overburden Decay:** Segregation Potential decays exponentially under structural overburden pressure (`SP = SP_0 * e^{-aP}`).

### Numerics
*   **Boolean Masking:** The active freezing fringe (e.g., `0°C` to `-0.5°C`) is isolated instantly across the 2D matrix using pure NumPy boolean masking `(T <= 0) & (T > -0.5)`.
*   **Vectorized Tensor Assembly:** `np.gradient(T)` calculates temperature vectors across the grid in one CPU cycle, multiplying with the fringe mask to instantly generate the spatial Volumetric Strain Tensor field. Zero explicit Python `for` loops.

### Deterministic engineering assessment (Module 3)
*   Triggers deterministic threshold alerts based on structural tolerances. If the localized differential heave rate (mm/day) inside the spatial tensor exceeds the maximum allowable foundation uplift, the system flags a structural failure risk.
### Numerics

- **Fully implicit (Backward Euler)** time integration — unconditionally
  stable, required because the freeze/thaw problem is numerically stiff
  near `T_f`.
- **Spatial operator**: 2D five-point Laplacian assembled once via
  `scipy.sparse.kron` of two 1D tridiagonal second-difference matrices
  (`scipy.sparse.diags`). No loop over the `nx * ny` grid nodes.
- **Nonlinearity** (`c_app(T)`, `k(T)` depend on `T`) resolved via a
  **Picard fixed-point iteration** — a handful of sparse linear solves
  (`scipy.sparse.linalg.spsolve`) per timestep, all vectorized.
- **Boundary conditions**: Dirichlet freeze-pipe (circular, brine/coolant
  temperature) and Dirichlet far-field, enforced by vectorized sparse
  row-replacement (boolean mask indexing — no per-node loop).
- **Only genuine Python-level loop**: the outer transient time-march
  (`for step in range(1, n_steps + 1)`). This is inherent to any implicit
  transient PDE solve and is explicitly *not* a spatial loop.

### Sub-modules (single-screen Command Center)

| Sub-module | Contents |
|---|---|
| **1A — Thermal Matrix Parameters** | Domain/grid size, unfrozen/frozen thermal conductivity, specific heat, bulk density, latent heat of fusion, gravimetric water content. |
| **1B — Boundary Setup** | Freeze-pipe geometry & coolant temperature (Dirichlet), initial ground temperature, far-field boundary temperature, target freeze-wall closure radius (design basis for the alert system). |
| **1C — Phase Engine** | Phase-change band width, Backward-Euler timestep & duration, Picard iteration count, alert thresholds, and the **Run** control. Outputs the live temperature matrix, freeze-front radius history, cooling-rate history, and frozen-fraction history. |

### Deterministic engineering assessment (Module 1)

- 🔴 **Red** — peak nodal cooling rate ≥ critical thermal-shock threshold
  (cracking risk from excessive thermal gradient).
- 🟢 **Green** — freeze-wall closure radius has reached the target design
  radius **and** cooling rate is within the safe band.
- 🟡 **Yellow** — anything in between (freeze wall still developing, or
  cooling rate in the moderate band).

All thresholds are named, numeric, `st.number_input`-editable constants —
fully auditable, no hidden scoring model.

---

## 4. UI/UX — The "Command Center" Paradigm

- **[7, 3] single-screen column split.** Left 70% = Live Matrix
  (visualizations). Right 30% = Control Panel (inputs), organized as
  Sub-module tabs (1A / 1B / 1C) so all of Module 1 runs seamlessly on one
  screen without page navigation.
- **No sliders.** Every engineering parameter is an exact `st.number_input`
  for professional numerical precision.
- **Dual (triple) visualization**, simultaneously, for every module:
  1. **Matplotlib line chart** — exact gradients / drop-off points (e.g.
     freeze-front radius vs. time).
  2. **Seaborn heatmap** — 2D spatial contours / thermal hotspots.
  3. **Pandas DataFrame** — raw, extractable numerical matrix, with a CSV
     download button.
- **"No Black Box" marquee** — a scrolling HTML/CSS banner at the top of
  every module screen, printing the exact differential equations and
  mixing laws currently driving the backend, sourced from a single
  canonical equation list per sub-module (`ui_components/marquee_engine.py`)
  so the UI text can never drift from the actual physics implementation.
- **Deterministic Red/Yellow/Green status boxes** below the graphs,
  computed by pure threshold logic in `ui_components/alert_system.py`.

---

## 5. Running the app

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 6. Running the tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v
```

`tests/test_thermal.py` is a fast regression suite (small grids, short
duration) covering the liquid-fraction smoothing function, apparent heat
capacity, conductivity mixing, Laplacian assembly/symmetry, and an
end-to-end small-grid transient run (monotonic cooling, finite fields,
Dirichlet pipe boundary held exactly).

`tests/test_mechanical.py` locks in the parameter contracts (default
values, physical bounds) for Modules 2–5 and asserts each stub fails loudly
(`NotImplementedError`) rather than returning fabricated numbers — this is
intentional so no downstream Module can silently produce numbers before its
physics is actually implemented.

---

## 7. Roadmap (Modules 2–6)

| Module | Scope | Status |
|---|---|---|
| 1 | Transient Stefan Phase-Change Matrix (AHCM) | ✅ Implemented |
| 2 | SFCC Cryogenic Suction Solver — Richards' equation coupled to the Soil Freezing Characteristic Curve, ice-lens growth via Clausius–Clapeyron suction | Implemented ✅✅|
| 3 | Volumetric Frost Heave Tensor — 9% water→ice expansion + segregation-potential ice-lens heave | 🔲 Contract defined (`core_physics/frost_heave.py`) |
| 4 | Thermo-Elastic Restrained Stress — lateral crushing pressure (MPa) on retaining structures | 🔲 Contract defined (`core_physics/restrained_stress.py`) |
| 5 | Thaw Consolidation Simulator — Morgenstern–Nixon void-ratio collapse, excess pore pressure, settlement | 🔲 Contract defined (`core_physics/thaw_consolidation.py`) |
| 6 | *(reserved — TBD scope)* | 🔲 Not started |

Each future module will consume the prior module's output (e.g. Module 2
consumes Module 1's `T_history`) and plug into the same Command Center
layout, marquee engine, dual visualizer, and alert system with zero
UI-layer rework.

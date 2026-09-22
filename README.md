# CryoTensor-AGF-Simulator

**Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine**

A standalone, deterministic, CPU-bound engineering simulator for Artificial Ground Freezing (AGF) design: transient phase-change heat transfer, cryogenic suction, frost heave, restrained structural stress, and thaw consolidation. This project is **not** a surface-water hydrodynamics platform — it is exclusively concerned with deep sub-surface solid mechanics, phase-change thermodynamics, and frost heave.

---

## 1. Design Philosophy

| Constraint | How it is enforced |
| --- | --- |
| **CPU-bound, runs on a Core i3** | Pure NumPy / SciPy sparse linear algebra only. No PyTorch, no CUDA, no GPU code path anywhere in `core_physics/`. |
| **Pure vectorization** | Spatial operators (Laplacians, mixing laws, boundary masks) are assembled with `scipy.sparse.diags` / `scipy.sparse.kron` and NumPy broadcasting. There is **no Python `for` loop over spatial grid nodes** anywhere in the codebase. The only sequential loop is the outer transient time-march, which is physically unavoidable for an implicit scheme (each timestep depends on the previous one). |
| **No machine learning** | 100% deterministic classical PDEs and closed-form geotechnical equations (Fourier's Law, the Stefan condition, Richards' equation, thermo-elasticity, Morgenstern–Nixon thaw consolidation). No trained models, no learned thresholds. |
| **Separation of concerns** | `core_physics/` contains only math (dataclasses + vectorized functions) — zero `import streamlit`. `ui_components/` contains only Streamlit presentation helpers — zero physics. `app.py` wires the two together and holds no equations of its own. |
| **Strict Static Typing** | Comprehensive type hinting (`dict[str, Any]`, Matplotlib structural types) and strict `dataclass` parameters resolve Pylance diagnostics, ensuring maximum stability without runtime type errors. |

---

## 2. Architecture

```text
CryoTensor-AGF-Simulator/
├── app.py                     # Streamlit Entry Point (The Command Center UI)
├── core_physics/              # CPU-bound, pure vectorized math — NO UI logic
│   ├── types.py               # Data structures & typed dicts (ConsolidationParams, etc.)
│   ├── phase_change.py        # Module 1: Stefan formulation, Apparent Heat Capacity Method  [IMPLEMENTED]
│   ├── cryosuction.py         # Module 2: SFCC, moisture migration, ice lens growth           [stub / contract]
│   ├── frost_heave.py         # Module 3: 9% volumetric expansion tensor                      [IMPLEMENTED]
│   ├── restrained_stress.py   # Module 4: Thermo-elastic structural stress (MPa)              [stub / contract]
│   └── thaw_consolidation.py  # Module 5: Melt-down excess pore pressure & settlement         [IMPLEMENTED]
├── ui_components/             # Streamlit layouts & UX engine — NO physics
│   ├── marquee_engine.py      # "No Black Box" scrolling HTML/CSS live-equation banner
│   ├── alert_system.py        # Deterministic Red/Yellow/Green threshold logic
│   └── dual_visualizer.py     # Simultaneous Matplotlib line + Seaborn heatmap + DataFrame
└── tests/
    ├── test_thermal.py        # Regression tests for Module 1 physics
    └── test_mechanical.py     # Contract tests for Modules 2–5 (params + NotImplementedError guards)

```

---

## 3. Module 1 — Transient Stefan Phase-Change Matrix

**Status: fully implemented.**

### Governing physics

2D transient heat conduction with latent heat release, via the **Apparent Heat Capacity Method (AHCM)**:

* $\rho \cdot c_{app}(T) \cdot \frac{\partial T}{\partial t} = \nabla \cdot (k(T) \cdot \nabla T)$
* $c_{app}(T) = c(T) + L \cdot \frac{df_l}{dT}$ (apparent/effective heat capacity)
* $f_l(T) = \text{clip}[ (T - (T_f - \frac{\Delta T}{2})) / \Delta T , 0, 1 ]$ (smoothed liquid/unfrozen fraction)
* $k(T) = f_l \cdot k_{unfrozen} + (1 - f_l) \cdot k_{frozen}$ (arithmetic conductivity mixing)

The phase-change (Stefan) condition is captured **without explicit front tracking** by smoothing the latent-heat release over a narrow temperature band `[T_f − ΔT/2, T_f + ΔT/2]`. This avoids a moving-mesh or level-set front-tracking scheme entirely, keeping the solver cheap enough for a Core i3.

### Numerics

* **Fully implicit (Backward Euler)** time integration — unconditionally stable, required because the freeze/thaw problem is numerically stiff near $T_f$.
* **Spatial operator**: 2D five-point Laplacian assembled once via `scipy.sparse.kron` of two 1D tridiagonal second-difference matrices. No loop over the `nx * ny` grid nodes.
* **Nonlinearity** resolved via a **Picard fixed-point iteration** — a handful of sparse linear solves per timestep, all vectorized.
* **Boundary conditions**: Dirichlet freeze-pipe and Dirichlet far-field, enforced by vectorized sparse row-replacement.

---

## 4. Module 5 — Thaw Consolidation Simulator

**Status: fully implemented.**

### Governing Physics & Geotechnical Mechanics

Module 5 handles the critical post-freeze phase, simulating infrastructure settlement driven by the thaw-weakened soil matrix utilizing the classical **Morgenstern-Nixon thaw consolidation framework**.

* **Excess Pore Pressure Generation:** Calculates the instantaneous hydrostatic pressure spike ($u$) at the active thaw boundary caused by rapid void ratio collapse.
* **Dynamic Permeability:** Employs the non-linear $e-\log(k)$ relationship to continuously update the coefficient of consolidation ($c_v$) as escaping water shrinks the available drainage paths.
* **Heterogeneous Loading Limits:** Supports multi-layered soil stratigraphy via localized parameter arrays, alongside a dynamic surface surcharge vector $q(t)$ to model massive overlying equipment or embankments.

### Numerics & Computational Logic

To satisfy the zero-spatial-loop requirement on standard CPUs, this module operates purely on sparse tensor mathematics:

* **Vectorized Moving Boundaries:** The advancing thaw front position $X_f(t)$ is isolated across all timesteps simultaneously using boolean matrix masking (`T_history > 0.0`) and `np.argmax`, entirely bypassing iterative spatial searches.
* **Staggered-Grid Harmonic Assembly:** The 1D diffusion governing equation ($\frac{\partial u}{\partial t} = \frac{\partial}{\partial z} ( c_v \frac{\partial u}{\partial z} )$) is mapped into a tridiagonal finite difference Laplacian using `scipy.sparse.diags`. The variable $c_v$ field is resolved using array-based arithmetic midpoint averaging to maintain absolute numerical precision.
* **Implicit Integration:** Stability is guaranteed via an unconditionally stable Backward Euler solver (`scipy.sparse.linalg.spsolve`), enabling the engine to resolve thousands of depth nodes in milliseconds.
* **Zero-Loop Settlement:** The macroscopic structural settlement time-series $S(t)$ is computed by passing the 2D void-ratio matrix through a discrete spatial integral (`np.trapz`), avoiding any sequential $z$-axis iterations.

---

## 5. UI/UX — The "Command Center" Paradigm

* **[7, 3] single-screen column split.** Left 70% = Live Matrix (visualizations). Right 30% = Control Panel (inputs), organized as Sub-module tabs so all workflows run seamlessly on one screen.
* **No sliders.** Every engineering parameter is an exact `st.number_input` for professional numerical precision.
* **Dual (triple) visualization**, simultaneously, for every module:
1. **Matplotlib line chart** — exact gradients / drop-off points.
2. **Seaborn heatmap** — spatial contours / field visualization.
3. **Pandas DataFrame** — raw, extractable numerical matrix, with a CSV download button.


* **"No Black Box" marquee** — a scrolling HTML/CSS banner printing the exact differential equations currently driving the backend.
* **Deterministic Red/Yellow/Green status boxes** computed by pure threshold logic.

---

## 6. Architecture Log: Resolving Pylance & Typing Complexity

Building a mathematically dense system required neutralizing widespread Pylance diagnostics and unknown type resolution errors that emerged when bridging Matplotlib components, Streamlit UI states, and generic Python dictionaries.

* **Strict Parameter Schemas:** Replaced generic dynamic structures with dedicated `ConsolidationParams` and `DynamicConsolidationParams` dataclasses/TypedDicts grouped into `core_physics/types.py` to prevent cyclic dependencies.
* **Graphics Typing Hardening:** Injected explicit matplotlib structural imports (`Figure`, `Axes`, `Line2D`, `Text`, `Legend`) across the visualization pipeline (`module4.py`, `module5.py`), ensuring the frontend engine strictly understands the layout outputs.
* **Registry Type Assurance:** Upgraded the application registry (`registry.py`) utilizing explicit tuple routing (`ModuleInfo = Tuple[str, Optional[str], bool]`). All generic `dict` outputs from the simulation solvers were upgraded to `dict[str, Any]` to eliminate ambiguity in the visualization routing.

---

## 7. Running the app

```bash
pip install -r requirements.txt
streamlit run app.py

```

## 8. Running the tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v

```

`tests/test_thermal.py` is a fast regression suite covering the liquid-fraction smoothing function, apparent heat capacity, conductivity mixing, Laplacian assembly/symmetry, and an end-to-end small-grid transient run.

`tests/test_mechanical.py` locks in the parameter contracts (default values, physical bounds) for all subsequent modules and asserts each stub fails loudly (`NotImplementedError`) rather than returning fabricated numbers.

---

## 9. Roadmap (Modules 1–6)

| Module | Scope | Status |
| --- | --- | --- |
| 1 | Transient Stefan Phase-Change Matrix (AHCM) | ✅ Implemented |
| 2 | SFCC Cryogenic Suction Solver — Richards' equation coupled to the Soil Freezing Characteristic Curve, ice-lens growth | 🔲 Contract defined |
| 3 | Volumetric Frost Heave Tensor — 9% water→ice expansion + segregation-potential ice-lens heave | ✅ Implemented |
| 4 | Thermo-Elastic Restrained Stress — lateral crushing pressure (MPa) on retaining structures | 🔲 Contract defined |
| 5 | Thaw Consolidation Simulator — Morgenstern–Nixon void-ratio collapse, excess pore pressure, settlement | ✅ Implemented |
| 6 | *(reserved — TBD scope)* | 🔲 Not started |

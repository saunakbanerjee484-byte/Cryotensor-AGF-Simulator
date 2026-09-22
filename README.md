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
│   ├── types.py               # Data structures & typed dicts (StressParams, HeaveParams, etc.)
│   ├── phase_change.py        # Module 1: Stefan formulation, Apparent Heat Capacity Method  [IMPLEMENTED]
│   ├── cryosuction.py         # Module 2: SFCC, moisture migration, ice lens growth           [IMPLEMENTED]
│   ├── frost_heave.py         # Module 3: 9% volumetric expansion tensor                      [IMPLEMENTED]
│   ├── restrained_stress.py   # Module 4: Thermo-elastic structural stress (MPa)              [IMPLEMENTED]
│   └── thaw_consolidation.py  # Module 5: Melt-down excess pore pressure & settlement         [IMPLEMENTED]
├── modules/
│   ├── module2.py             # UI presentation for Module 2
│   ├── module3.py             # UI presentation for Module 3
│   └── module4.py             # UI presentation for Module 4 (Physics toggles & Diagnostics)
├── ui_components/             # Streamlit layouts & UX engine — NO physics
│   ├── marquee_engine.py      # "No Black Box" scrolling HTML/CSS live-equation banner
│   ├── alert_system.py        # Deterministic Red/Yellow/Green threshold logic
│   └── dual_visualizer.py     # Simultaneous Matplotlib line + Seaborn heatmap + DataFrame
└── tests/
    ├── test_thermal.py        # Regression tests for Module 1 physics
    ├── test_frost_heave.py    # Regression tests for Module 3 physics
    ├── test_restrained_stress.py # Regression tests for Module 4 physics
    └── test_mechanical.py     # Contract tests for Modules 2–5

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

The phase-change (Stefan) condition is captured **without explicit front tracking** by smoothing the latent-heat release over a narrow temperature band.

### Numerics

* **Fully implicit (Backward Euler)** time integration — unconditionally stable.
* **Spatial operator**: 2D five-point Laplacian assembled once via `scipy.sparse.kron` of two 1D tridiagonal second-difference matrices. No loop over the `nx * ny` grid nodes.
* **Nonlinearity** resolved via a **Picard fixed-point iteration** — a handful of sparse linear solves per timestep, all vectorized.

---

## 4. Module 2 — SFCC Cryogenic Suction Solver

**Status: fully implemented.**

### Governing Physics & Geotechnical Mechanics

Module 2 models the highly non-linear hydrogeological phenomena occurring at the active freeze front.

* **Soil Freezing Characteristic Curve (SFCC):** Translates thermal gradients into volumetric unfrozen water content ($\theta_u$). Utilizes the **van Genuchten** empirical framework alongside Clausius-Clapeyron cryogenic suction mechanics.
* **Cryogenic Suction Generation:** As temperature drops below 0°C, the drastic reduction in unfrozen water creates immense matric suction ($\psi$).
* **Richards' Equation:** Transient unsaturated moisture flow is governed by the 1D Richards' equation.

### Numerics & Computational Logic

* **Vectorized Richards' Engine:** 1D implicit Backward-Euler time integration over the spatial grid.
* **Picard-Linearized Updates:** Solves highly stiff, non-linear dependencies ($K(\psi)$ and $C(\psi)$) via fixed-point Picard iterations without Python `for` loops across the depth tensor.

---

## 5. Module 3 — Volumetric Frost Heave Tensor

**Status: fully implemented.**

### Governing Physics & Geotechnical Mechanics

Module 3 converts a 2D temperature field into a spatially resolved volumetric heave-rate field and volumetric strain tensor field.

* **In-situ expansion (primary heave):** Represents the 9% volumetric expansion of pore water converting to ice. Scaled by porosity and local phase-change velocity ($h_{insitu\_dot} = 0.09 \cdot n \cdot (dz/dt)$).
* **Segregation heave (secondary heave / ice lensing):** Governed by the Konrad & Morgenstern Segregation Potential (SP) model. Intake velocity is proportional to the temperature gradient in the frozen fringe ($v_s = SP \cdot \nabla T$).
* **Active Freezing Fringe Limits:** Mechanisms are physically restricted to the sub-zero band (e.g., 0°C to -0.5°C) isolated via boolean matrix masking.

### Numerics & Computational Logic

* **Vectorized Fields:** Isotherm velocity, segregation flux, in-situ expansion, and strain fields execute as single vectorized expressions without spatial `for` loops.
* **Fallback Generator:** Dynamically consumes the 2D temperature field from Module 1 via the shared session-state pipeline. If unavailable, falls back to a synthetic mock temperature generator.

---

## 6. Module 4 — Thermo-Elastic Restrained Stress & THMC Degradation

**Status: fully implemented.**

### Governing Physics & Geotechnical Mechanics

Module 4 models the complex lateral earth pressures and multi-physics degradation of the frozen soil matrix against retaining structures.

* **Restrained Thermo-Elastic Core:** Translates the volumetric strain tensor from Module 3 into a multi-axial stress field via eigenstrain decomposition. The structural restraint factor ($\beta$) decays exponentially into the free field.


* **Visco-Plastic Creep Tensor:** Accumulates temperature-coupled Vialov/Ladanyi secondary creep based on the J2-invariant von Mises equivalent stress. Visco-plastic flow dynamically relaxes the elastic strain constraints.


* **Continuum Damage & Hydro-Chemical Degradation:** Unifies mechanical micro-cracking (tensile strain limits) and advective/solute thermal erosion into a single damage scalar ($D \in [0,1]$). This globally degrades the soil's stiffness matrix.



### Numerics & Computational Logic

* **Vectorized Tensor Assembly:** The core constitutive integration utilizes a single `scipy.sparse` block-diagonal matrix-vector product to convert the entire 2D strain field into the damaged stress tensor simultaneously.


* **Sparse Derivative Operators:** Advective seepage and solute rejection mechanisms are resolved using a vectorized `scipy.sparse.diags` first-derivative spatial operator.


* **Zero Spatial Loops:** The only Python-level loop is the highly restricted outer time-march (sub-steps) driving creep accumulation; every operation within the loop evaluates the entire spatial grid simultaneously via NumPy broadcasting.



### UI/UX (Module 4C Advanced Control Panel)

* **THMC Physics Toggles:** Allows engineers to independently toggle Visco-Plastic Creep, Advective Seepage, Solute Rejection, and Continuum Damage to compare pure elasticity against full THMC degradation.


* **Pipeline Diagnostics & Mock Data Mode:** Actively monitors the state pipeline for Module 1, 2, and 3 handoffs. Includes a "Mock Data Mode" that generates synthetic radial strain and temperature fields if upstream modules have not yet been executed in the session.


* **Dual Visualization:** Generates an interactive Matplotlib 1D profile of Lateral Earth Pressure vs. Depth alongside a toggleable Seaborn heatmap mapping either the Principal Stress Field ($\sigma_1$) or the Continuum Damage Scalar ($D$).



---

## 7. Module 5 — Thaw Consolidation Simulator

**Status: fully implemented.**

### Governing Physics & Geotechnical Mechanics

Module 5 handles the critical post-freeze phase utilizing the classical **Morgenstern-Nixon thaw consolidation framework**.

* **Excess Pore Pressure Generation:** Calculates the instantaneous hydrostatic pressure spike ($u$) at the active thaw boundary caused by rapid void ratio collapse.
* **Dynamic Permeability:** Employs the non-linear $e-\log(k)$ relationship to continuously update the coefficient of consolidation ($c_v$).

### Numerics & Computational Logic

* **Vectorized Moving Boundaries:** The advancing thaw front position $X_f(t)$ is isolated using boolean matrix masking (`T_history > 0.0`) and `np.argmax`.
* **Staggered-Grid Harmonic Assembly:** The 1D diffusion governing equation is mapped into a tridiagonal finite difference Laplacian using `scipy.sparse.diags`.
* **Zero-Loop Settlement:** Structural settlement $S(t)$ is computed by passing the 2D void-ratio matrix through a discrete spatial integral (`np.trapz`).

---

## 8. Architecture Log: Resolving Pylance & Typing Complexity

Building a mathematically dense system required neutralizing widespread Pylance diagnostics and unknown type resolution errors that emerged when bridging Matplotlib components, Streamlit UI states, and generic Python dictionaries.

* **Strict Parameter Schemas:** Replaced generic dynamic structures with dedicated `StressParams`, `HeaveParams`, and `ConsolidationParams` dataclasses grouped into `core_physics/types.py` to prevent cyclic dependencies.
* **Graphics Typing Hardening:** Injected explicit matplotlib structural imports (`Figure`, `Axes`, `Line2D`) across the visualization pipeline, ensuring the frontend engine strictly understands the layout outputs.

---

## 9. Running the app

```bash
pip install -r requirements.txt
streamlit run app.py

```

## 10. Running the tests

```bash
pip install -r requirements.txt pytest
pytest tests/ -v

```

All test suites follow a strict architectural pattern: small grid domains, fast execution, and deterministic assertions.

* `tests/test_thermal.py`: covers the liquid-fraction smoothing function, apparent heat capacity, and conductivity mixing.
* `tests/test_frost_heave.py`: asserts volumetric strain scaling and exponential segregation potential decay.
* `tests/test_restrained_stress.py`: validates the sparse constitutive matrix assemblies, confirms von Mises non-negativity, and verifies that disabling all THMC toggles gracefully collapses the solution back to pure restrained elasticity.


* `tests/test_mechanical.py`: locks in the parameter contracts for all downstream modules.

---

## 11. Roadmap (Modules 1–6)

| Module | Scope | Status |
| --- | --- | --- |
| 1 | Transient Stefan Phase-Change Matrix (AHCM) | ✅ Implemented |
| 2 | SFCC Cryogenic Suction Solver | ✅ Implemented |
| 3 | Volumetric Frost Heave Tensor | ✅ Implemented |
| **4** | **Thermo-Elastic Restrained Stress & THMC Degradation** | **✅ Implemented** |
| 5 | Thaw Consolidation Simulator | ✅ Implemented |
| 6 | *(reserved — TBD scope)* | 🔲 Not started |

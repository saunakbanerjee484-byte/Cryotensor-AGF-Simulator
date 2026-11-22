<div align="center">

# 🧊 CryoTensor-AGF-Simulator

### Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-62%20passed-brightgreen?logo=pytest&logoColor=white)](tests/)
[![Verification](https://img.shields.io/badge/verification-nominal%20%2B%20stress%20suite-blue)](verification_suite.py)
[![Numerics](https://img.shields.io/badge/Numerics-NumPy%20%7C%20SciPy-013243?logo=numpy&logoColor=white)](requirements.txt)
[![UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](app.py)
[![Deterministic](https://img.shields.io/badge/Physics-Deterministic%20%7C%20No%20ML-6A1B9A)](#1-design-philosophy)
[![CPU Bound](https://img.shields.io/badge/Runtime-CPU--Bound%20%7C%20Core%20i3%20Ready-0A7E8C)](#1-design-philosophy)

<br/>

> **A deterministic, CPU-bound thermo-hydro-mechanical simulator for Artificial Ground Freezing (AGF)** — from transient Stefan phase change to thaw consolidation, with auditable physics, reproducible verification, and publication-grade engineering traceability.

<br/>

`Stefan Phase Change` · `Cryosuction / Richards` · `Frost Heave Tensor` · `THMC Restrained Stress` · `Thaw Consolidation`

</div>

---

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
python -m pytest tests -v

# Reproducible nominal + stress/boundary verification harness
python verification_suite.py --json verification_report.json

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

## Contents

<table>
<tr>
<td width="50%" valign="top">

### Core Engineering
- [1. Design Philosophy](#1-design-philosophy)
- [2. Architecture](#2-architecture)
- [3–7. Physics Modules](#3-module-1--transient-stefan-phase-change-matrix)
- [8. Architecture Log](#8-architecture-log-resolving-pylance--typing-complexity)
- [9. Running the app](#9-running-the-app)
- [10. Running the tests](#10-running-the-tests)
- [11. Roadmap](#11-roadmap-modules-16)

</td>
<td width="50%" valign="top">

### Deep Dives & Assurance
- [How the Physics Actually Works (ELI5)](#how-the-physics-actually-works-the-eli5-edition)
- [FAQ: Mathematics & Physics Demystified](#-faq-the-mathematics--physics-demystified)
- [For Tech Enthusiasts](#for-tech-enthusiasts)
- [FAQ: Software Engineering & Architecture](#-faq-the-software-engineering--architecture)
- [Verification, Test Conditions & Failure Envelope](#verification-test-conditions--failure-envelope)
- [Security & Reliability Notes](#security--reliability-notes)

</td>
</tr>
</table>

---

<div align="center">

# 🧊 How the Physics Actually Works (The ELI5 Edition)

*From liquid nitrogen to thaw settlement — the full AGF lifecycle, explained plainly.*

</div>

If you are wondering, *"What is Artificial Ground Freezing (AGF) and what does this math actually do?"* — read this.

**The Core Concept:** When engineers need to dig a deep tunnel or mine, but the ground is too soft or full of groundwater, they can't just dig (it will collapse). So, they pump liquid nitrogen into the ground to freeze the water inside the soil. The soil turns into a temporary, solid wall of ice-rock. You can dig safely.

But freezing ground is violent. It moves, expands, and breaks things. This simulator predicts exactly how that happens.

Here is how the 5 modules work in plain English:

## 🌡️ Module 1: The Cold Front (Heat Transfer)

**What it does:** Predicts how fast the ground freezes.

**The Physics:** When water turns into ice, it actually releases a little bit of heat (Latent Heat). So, the ground fights back against the freezing. Instead of tracking an exact, sharp ice-boundary (which crashes computers), we use the **Apparent Heat Capacity Method (AHCM)**. Think of it like a blurry line where water is gradually turning into ice over a 1-degree temperature drop.

> **🔬 Academic Rigor & Theoretical Justification**
> The non-linear transient heat equation is formulated using the Apparent Heat Capacity Method (AHCM) to bypass explicit front-tracking (the classical Stefan problem). By approximating the Dirac delta function of latent heat release over a narrow temperature domain ΔT, we maintain strict energy continuity. The resulting highly stiff non-linear PDEs are solved unconditionally via a fully implicit Backward Euler time-marching scheme. Spatial discretization utilizes a 5-point finite difference Laplacian assembled via sparse Kronecker products (`scipy.sparse.kron`). The temperature-dependent thermal conductivity k(T) and apparent heat capacity matrix are iteratively converged at each timestep utilizing a fixed-point Picard iteration, ensuring stability without grid-loop overhead.

## 🧽 Module 2: The Ice Sponge (Cryogenic Suction)

**What it does:** Explains why freezing ground sucks up water from below.

**The Physics:** Ice wants to grow. As the temperature drops below 0°C, the ice "pulls" surrounding unfrozen water towards itself (Cryogenic Suction). Imagine a dry sponge placed on a puddle of water. The freezing front acts exactly like that sponge, violently sucking up groundwater from the warmer soil below.

> **🔬 Academic Rigor & Theoretical Justification**
> Sub-zero matric suction is governed by the generalized Clausius-Clapeyron equality, coupling thermal gradients directly to pore-water thermodynamics. Moisture migration through the unsaturated frozen fringe is dictated by the 1D Richards' equation. The soil freezing characteristic curve (SFCC) is parameterized via van Genuchten formulations, yielding highly non-linear hydraulic conductivity K(ψ) and specific moisture capacity C(ψ) tensors. To resolve extreme stiffness at the active freezing front, the implicit numerical formulation employs chord-slope approximations and Picard linearizations. This guarantees monotonic convergence of the pressure head profile during dynamic ice segregation without oscillatory instability.

## 💥 Module 3: Ground Swelling (Frost Heave)

**What it does:** Calculates how much the ground bulges upwards.

**The Physics:** You know how a water bottle bursts if you leave it in the freezer? Water expands by 9% when it freezes. But it gets worse: remember the "sponge" effect from Module 2? That extra water being sucked in also freezes, creating thick bands of pure ice called **Ice Lenses**. The ground swells up massively. This is called **Frost Heave**.

> **🔬 Academic Rigor & Theoretical Justification**
> Total volumetric expansion encompasses both in-situ pore-water phase change (the 9% volume expansion constraint) and secondary segregation heave. The latter is mathematically bounded by the Konrad-Morgenstern Segregation Potential (SP) model, where intake velocity v_s is linearly proportional to the temperature gradient ∇T across the active fringe. Using boolean matrix masking to isolate the thermodynamically active sub-zero zone, the total heave rate integrates both fluxes spatially. The resulting 1D heave displacement is mapped into a fully vectorized, time-dependent volumetric strain tensor ε_v, which acts as the primary kinematic boundary condition for the downstream thermo-mechanical module.

## 🏗️ Module 4: The Pressure Cooker (Thermo-Elastic Stress)

**What it does:** Calculates how hard the swelling ground pushes against concrete walls.

**The Physics:** The ground is expanding (Module 3), but there are concrete retaining walls in the way. So, the soil generates massive internal pressure (**Restrained Stress**). Over time, under this immense pressure, the ice starts to slowly deform and flow like a glacier (**Visco-Plastic Creep**), and the soil structure gets permanently damaged (**THMC Degradation**).

> **🔬 Academic Rigor & Theoretical Justification**
> Stresses are resolved via a coupled thermo-poro-mechanical framework. The volumetric strain tensor translates into a multi-axial effective stress field assuming restrained boundary conditions. The yielding mechanism employs a J2-invariant von Mises equivalent stress approach, coupled with Vialov and Ladanyi's empirical formulations for temperature-dependent visco-plastic secondary creep. Furthermore, continuum damage mechanics (D ∈ [0,1]) is integrated to model hydro-chemical degradation and micro-cracking. The global stiffness matrix is dynamically relaxed through a single sparse block-diagonal matrix-vector product, accurately representing stress redistribution and strain-softening without explicit local mesh iterations.

## 🧊 Module 5: The Meltdown (Thaw Consolidation)

**What it does:** Predicts the disaster when the ice finally melts.

**The Physics:** The tunnel is built, and the refrigeration is turned off. The ice melts. Suddenly, all those thick ice lenses turn back into water. You now have soil that is completely destroyed (from Module 4) and drowning in excess water. The ground rapidly collapses and sinks (**Settlement**), and water squirts out under high pressure.

> **🔬 Academic Rigor & Theoretical Justification**
> Post-freezing structural settlement is treated as a highly non-linear moving boundary problem governed by the Morgenstern-Nixon thaw consolidation theory. The instantaneous collapse of the ice matrix generates massive excess pore pressures u, dictated by a coupled fluid-solid diffusion equation. To account for large-strain deformations, the coefficient of consolidation c_v is dynamically updated at every timestep using the non-linear e-log(k) void ratio-permeability relationship. The active spatial domain updates dynamically via `np.argmax` thresholding, while total settlement S(t) is derived from the discrete spatial integral (`np.trapz`) of the collapsing void-ratio tensor.

---

<div align="center">

# 🧠 FAQ: The Mathematics & Physics Demystified

*ELI5 · Theoretical Rigor · Challenge & Defense*

</div>

Each entry has an ELI5 explanation, the theoretical rigor, and a prepared defense against the obvious challenge.

## 1. Apparent Heat Capacity Method (AHCM) & Dirac Delta Function

**👶 ELI5:** When water turns into ice, it releases a sudden burst of heat. Tracking the exact micro-millimeter where this happens crashes computers. So, we "smear" or spread this heat burst over a tiny temperature range (like 1°C). The sudden burst is the "Dirac Delta", and the smearing is the "AHCM".

**🔬 Theoretical Rigor:** The latent heat of fusion behaves as a discontinuous Dirac delta function δ(T − T_f) at the phase boundary. AHCM bypasses explicit moving-boundary front tracking (classical Stefan problem) by approximating this delta function as a continuous peak over a narrow temperature interval ΔT. This is added to the volumetric heat capacity, ensuring energy conservation without mesh distortion.

> **⚠️ Challenge:** "Smearing the latent heat artificially alters the thermal diffusivity. How do you ensure you don't 'lose' energy and violate thermodynamics?"
>
> **🛡️ Defense:** By integrating the apparent heat capacity curve exactly over ΔT. The area under the smoothed peak mathematically equals the latent heat L. Furthermore, I use fully implicit time-stepping with small Δt, ensuring the energy balance closes at each iteration.

## 2. Kronecker Products (`scipy.sparse.kron`)

**👶 ELI5:** Imagine making a 2D chessboard out of two 1D rulers. The Kronecker product multiplies two 1D lines to instantly build a 2D grid. We do this to avoid writing slow Python `for` loops that visit every single square one by one.

**🔬 Theoretical Rigor:** It is a tensor product used to construct the 2D discrete Laplacian matrix. By taking the Kronecker product of 1D finite difference matrices (I⊗D_xx + D_yy⊗I), we assemble the entire spatial operator globally.

> **⚠️ Challenge:** "Kronecker products generate massive N×N matrices. For a 10,000 node grid, isn't a 100,000,000 element matrix highly inefficient for a CPU-bound solver?"
>
> **🛡️ Defense:** It would be, if it were dense. But I am using `scipy.sparse.kron`. The resulting matrix is a sparse block-tridiagonal matrix. It only stores the non-zero diagonals, keeping memory complexity at O(N). It runs seamlessly on a Core i3.

## 3. Picard Iteration & Non-linear Hydraulic Conductivity K(ψ)

**👶 ELI5:** Water flows easily through wet soil but stops when it freezes or dries out. The flow speed K(ψ) drops wildly and unpredictably (non-linear). To solve this puzzle, the computer makes a smart guess, checks how wrong it is, adjusts, and repeats until the guess is perfect. That loop is a Picard iteration.

**🔬 Theoretical Rigor:** Under the van Genuchten framework, hydraulic conductivity K(ψ) and moisture capacity are highly non-linear functions of matric suction. To solve the stiff Richards' equation, Picard iteration (successive substitution) evaluates the non-linear coefficients at the previous iteration level m to solve for state m+1, looping until |h^(m+1) − h^m| < ε.

> **⚠️ Challenge:** "Picard iteration converges linearly and often fails for highly stiff infiltration/freezing fronts. Why not use Newton-Raphson which has quadratic convergence?"
>
> **🛡️ Defense:** Newton-Raphson requires assembling a Jacobian matrix at every step, which is computationally expensive to vectorize purely in SciPy sparse formats. Picard is Jacobian-free. I maintain stability by dynamically stepping down the timestep size (Δt) if the iteration struggles to converge.

## 4. J2-Invariant & von Mises Equivalent Stress

**👶 ELI5:** The frozen ground is being squeezed from 3 different directions (X, Y, Z). Instead of checking all 3 pressures separately to see if the ice breaks, we combine them into one master "Stress Number". If this number crosses a limit, the ground yields.

**🔬 Theoretical Rigor:** The second invariant of the deviatoric stress tensor (J₂) isolates shear stress from hydrostatic pressure. By using the von Mises yield criterion (√(3J₂)), complex 3D thermo-mechanical states are mapped into a single equivalent scalar to compute visco-plastic yielding.

> **⚠️ Challenge:** "Soil is a frictional material whose strength depends on confining pressure (Mohr-Coulomb). von Mises assumes yield is independent of confinement. Why use it for soil?"
>
> **🛡️ Defense:** Standard soil is frictional, yes. But artificially frozen soil behaves like ice — a cohesive, crystalline material. Ice creep and yield are almost entirely governed by deviatoric shear stresses, not confining pressure, making von Mises highly accurate for the frozen matrix.

## 5. Vialov & Ladanyi's Visco-Plastic Secondary Creep

**👶 ELI5:** If you push hard against a block of ice, it doesn't shatter instantly; it slowly flows and deforms over months like thick honey. This formula calculates exactly how fast the frozen wall will bend depending on how cold it is.

**🔬 Theoretical Rigor:** It is an empirical constitutive law linking steady-state (secondary) creep strain rate to applied equivalent stress and temperature. The strain rate is proportional to σₑⁿ and relies on temperature-dependent viscosity parameters, accurately capturing the rheological flow of the frozen earth retaining structure.

> **⚠️ Challenge:** "This only models secondary steady-state creep. What about primary (transient) or tertiary (accelerating failure) creep?"
>
> **🛡️ Defense:** For AGF projects, the design life is dictated by steady-state secondary creep. Primary creep exhausts rapidly. Tertiary creep means catastrophic structural failure — the exact scenario my AGF design engine operates to prevent by keeping stresses well within the secondary threshold.

## 6. Morgenstern-Nixon Thaw Consolidation Theory

**👶 ELI5:** When the project is done and the ice melts, the soil suddenly has too much water and loses its strength. It acts like a leaking water balloon being stepped on, shrinking rapidly and squirting water out. This theory calculates exactly how fast the ground will sink.

**🔬 Theoretical Rigor:** It is a classical moving-boundary consolidation theory. It couples the Stefan thermal phase-change moving boundary with Terzaghi's 1D consolidation. It analytically predicts the generation and dissipation of excess pore water pressures (u) as the thaw boundary X_f(t) advances into the frozen mass.

> **⚠️ Challenge:** "Morgenstern-Nixon assumes small strains. But thaw settlement of ice-rich soil causes massive volume collapse. Doesn't this violate the core assumption of the theory?"
>
> **🛡️ Defense:** Classical M-N is indeed small-strain. I bypassed this limitation by dynamically updating the void ratio e and the coefficient of consolidation c_v at every timestep using the non-linear e-log(k) relationship. This effectively simulates large-strain behavior within an updated grid framework.

## 7. Implicit vs. Explicit Time-Stepping (Backward Euler)

**👶 ELI5:** Imagine driving a car blindly. "Explicit" means taking a step forward based on where you are right now. If you take too big a step, you might walk off a cliff (the code crashes). "Implicit" means looking at where you want to end up and stepping safely towards it. It takes more brainpower (math), but you can take massive steps without ever crashing.

**🔬 Theoretical Rigor:** The simulator uses a fully implicit Backward Euler scheme rather than a Forward Euler explicit scheme. Explicit methods are conditionally stable and restricted by the Courant-Friedrichs-Lewy (CFL) condition, meaning highly stiff diffusion equations require microscopically small time steps (Δt). The implicit scheme requires solving a system of linear equations at each step but guarantees unconditional numerical stability.

> **⚠️ Challenge:** "Implicit schemes introduce artificial numerical diffusion (smearing of the solution). Doesn't this degrade the accuracy of the sharp freezing front?"
>
> **🛡️ Defense:** Yes, first-order Backward Euler introduces numerical dissipation proportional to the time step Δt. However, the physical thermal diffusivity of soil is inherently dissipative. By keeping Δt reasonably tight during rapid temperature gradients, the numerical diffusion remains mathematically negligible compared to the physical thermal conduction.

## 8. Tensor Fields vs. Scalar Fields

**👶 ELI5:** A scalar is just a single number, like the temperature in your room (25°C). A tensor is a number with a direction and complex shape, like the stress inside a twisted metal rod. Our simulator doesn't just calculate how cold the ground is (scalar); it calculates the forces tearing the ground apart (tensor).

**🔬 Theoretical Rigor:** While temperature and pore pressure are modelled as scalar fields φ(x,y), the mechanical interactions — strain ε_ij and stress σ_ij — are modeled as 2nd-order tensors. The simulator maps volumetric scalar outputs (like frost heave expansion) into the diagonal components of the strain tensor, which is then integrated via the constitutive matrix to yield the effective stress tensor field.

> **⚠️ Challenge:** "You claim to solve tensors, but you are running a 2D domain. Aren't you ignoring the out-of-plane (Z-axis) stress interactions?"
>
> **🛡️ Defense:** The model strictly enforces Plane Strain conditions (ε_zz = 0). In deep tunnel excavations or long retaining walls, the longitudinal dimension is massive compared to the cross-section. The out-of-plane strain is zero, but the out-of-plane stress σ_zz = ν(σ_xx + σ_yy) is rigorously computed and included in the von Mises yield criterion.

## 9. Vectorization vs. Python `for` Loops

**👶 ELI5:** Imagine a teacher grading 10,000 exam papers. A Python `for` loop is like grading them one by one — it takes forever. Vectorization is like having 10,000 teachers grade all papers at the exact same second. It's how we make complex math run lightning fast on a normal laptop.

**🔬 Theoretical Rigor:** Standard Python `for` loops over spatial arrays invoke massive interpreter overhead due to dynamic typing and object instantiation per iteration. Vectorization delegates operations down to highly optimized, pre-compiled C/C++ and Fortran libraries (NumPy/SciPy/BLAS/LAPACK). The spatial grid is processed as continuous memory blocks using Single Instruction, Multiple Data (SIMD) paradigms.

> **⚠️ Challenge:** "If vectorization is so powerful, why did you state that the time-march (outer loop) is sequential? Why not vectorize time as well?"
>
> **🛡️ Defense:** Because physics dictates causality. The temperature at t+Δt depends absolutely on the temperature at t. You cannot compute the future without first resolving the present. Spatial nodes exist simultaneously and can be vectorized; time is strictly sequential.

## 10. Continuum Damage Mechanics (Damage Scalar D)

**👶 ELI5:** As the frozen ground swells and cracks, it gets weaker. Instead of tracking thousands of microscopic cracks, we use a single number from 0 to 1. D = 0 means perfect, solid ground. D = 1 means the ground is completely pulverized. The closer to 1, the softer the ground acts in the math.

**🔬 Theoretical Rigor:** Rather than employing discrete fracture mechanics (DFM), the engine utilizes isotropic continuum damage mechanics. The scalar damage variable D represents the volumetric degradation of the material structure due to micro-cracking and thermal erosion. The global elasticity tensor is degraded via the effective stress concept: σ̃ = σ / (1 − D), smoothly reducing the macroscopic stiffness matrix.

> **⚠️ Challenge:** "Isotropic damage assumes micro-cracks form equally in all directions. In restrained freezing, cracks are highly directional (anisotropic). Isn't this an oversimplification?"
>
> **🛡️ Defense:** Yes, it is a homogenization. Implementing a full anisotropic damage tensor (D_ijkl) would require tracking crack opening vectors, exponentially increasing the degrees of freedom and destroying the CPU-bound constraint. For macroscopic stability assessment of an AGF wall, isotropic scalar degradation conservatively bounds the structural yielding behavior without extreme computational overhead.

---

<div align="center">

# 💻 For Tech Enthusiasts

*Each module through three lenses: plain-English, CS/software view, and academic rigor.*

</div>

Each module in three lenses: plain-English, CS/software view, and academic rigor.

## Module 1: Transient Stefan Phase-Change Matrix

**👶 ELI5 & CS View:** Imagine water freezing. It releases a sudden burst of heat. Tracking the exact millimeter of this ice boundary crashes normal code. So, we "smear" this heat burst over a tiny 1°C range. From a software perspective: instead of writing a slow Python `for` loop to check every single pixel on the screen, we multiply two 1D rulers (using Kronecker products) to update the entire 2D grid instantly in RAM. No AI, just pure matrix math.

**🔬 Academic Rigor:** The non-linear transient heat equation is formulated using the Apparent Heat Capacity Method (AHCM) to bypass explicit front-tracking. By approximating the Dirac delta function of latent heat release over a narrow temperature domain ΔT, we maintain strict energy continuity. The resulting highly stiff non-linear PDEs are solved unconditionally via a fully implicit Backward Euler time-marching scheme. Spatial discretization utilizes a 5-point finite difference Laplacian assembled via sparse Kronecker products (`scipy.sparse.kron`). The temperature-dependent thermal conductivity k(T) and apparent heat capacity matrix are iteratively converged at each timestep utilizing a fixed-point Picard iteration, ensuring absolute stability.

## Module 2: SFCC Cryogenic Suction Solver

**👶 ELI5 & CS View:** Freezing ground acts like a dry sponge, violently sucking up groundwater from below due to dropping temperatures. For the CS folks: the flow of water is wildly unpredictable and non-linear. Instead of a Machine Learning model guessing the flow, the engine makes a smart mathematical guess, checks the error, and self-corrects until perfect (Picard iteration). All of this executes via vectorized NumPy arrays — zero lag and zero deep-learning overhead.

**🔬 Academic Rigor:** Sub-zero matric suction is governed by the generalized Clausius-Clapeyron equality, coupling thermal gradients directly to pore-water thermodynamics. Moisture migration through the unsaturated frozen fringe is dictated by the 1D Richards' equation. The soil freezing characteristic curve (SFCC) is parameterized via van Genuchten formulations, yielding highly non-linear hydraulic conductivity K(ψ). To resolve extreme stiffness at the active freezing front, the implicit numerical formulation employs chord-slope approximations and Picard linearizations. This guarantees monotonic convergence of the pressure head profile during dynamic ice segregation without oscillatory instability.

## Module 3: Volumetric Frost Heave Tensor

**👶 ELI5 & CS View:** Water expands by 9% when freezing. Combine this with the extra water sucked up in Module 2, and the ground swells massively with ice lenses. For the CS folks: how do we track this without loops? We use **Boolean Matrix Masking**. We instantly filter the 2D grid to isolate only the sub-zero zones (T < 0) and apply the expansion math strictly there. It executes as a single, lightning-fast operation across thousands of data points.

**🔬 Academic Rigor:** Total volumetric expansion encompasses both in-situ pore-water phase change and secondary segregation heave. The latter is mathematically bounded by the Konrad-Morgenstern Segregation Potential (SP) model, where intake velocity v_s is linearly proportional to the temperature gradient ∇T. Using boolean matrix masking to isolate the thermodynamically active sub-zero zone, the total heave rate integrates both fluxes spatially. The resulting 1D heave displacement is mapped into a fully vectorized, time-dependent volumetric strain tensor ε_v, forming the rigorous kinematic boundary condition for the stress analysis module.

## Module 4: Thermo-Elastic Restrained Stress & THMC Degradation

**👶 ELI5 & CS View:** The swelling ground hits concrete walls, creating massive pressure until the frozen soil starts cracking and flowing like thick honey. For the CS folks: instead of simulating a million individual micro-cracks (which requires supercomputers), we use a single **Damage** variable from 0 to 1 for the whole grid. We convert complex 3D forces into one master scalar and solve the structural yielding using one giant sparse block-diagonal matrix.

**🔬 Academic Rigor:** Stresses are resolved via a coupled thermo-poro-mechanical framework. The volumetric strain tensor translates into a multi-axial effective stress field assuming restrained boundary conditions. The yielding mechanism employs a J2-invariant von Mises equivalent stress approach, coupled with Vialov-Ladanyi empirical formulations for temperature-dependent visco-plastic secondary creep. Continuum damage mechanics (D ∈ [0,1]) is integrated to model hydro-chemical degradation and micro-cracking. The global stiffness matrix is dynamically relaxed through a single sparse block-diagonal matrix-vector product, accurately representing stress redistribution and strain-softening without explicit local mesh iterations.

## Module 5: Thaw Consolidation Simulator

**👶 ELI5 & CS View:** When the refrigeration stops, the ice melts, and the ground collapses rapidly, squirting out trapped water under high pressure. For the CS folks: this is a classic **Moving Boundary Problem**. As the melt-line moves, the code dynamically updates the permeability (how fast water escapes) at every single timestep, using `np.argmax` to track the boundary instantly. No heavy 3D physics engines needed — just pure, staggered-grid numerical diffusion.

**🔬 Academic Rigor:** Post-freezing structural settlement is treated as a highly non-linear moving boundary problem governed by the Morgenstern-Nixon thaw consolidation theory. The instantaneous collapse of the ice matrix generates massive excess pore pressures u, dictated by a coupled fluid-solid diffusion equation. To account for large-strain deformations, the coefficient of consolidation c_v is dynamically updated at every timestep using the non-linear e-log(k) void ratio-permeability relationship. The active spatial domain updates via `np.argmax` thresholding, while total settlement S(t) is derived from the discrete spatial integral (`np.trapz`) of the collapsing void-ratio tensor.

---

<div align="center">

# 🏗️ FAQ: The Software Engineering & Architecture

*For CS students, code reviewers, and tech enthusiasts.*

</div>

*(For CS students, code reviewers, and tech enthusiasts)*

## 11. Why NO Machine Learning / AI?

**👶 ELI5:** Machine Learning is like predicting the weather by looking at past records — it's a very smart guess. But if you want to build a bridge, you don't "guess" gravity; you calculate it using exact laws. This engine doesn't guess. It uses pure laws of physics to calculate exactly what happens to the ice and soil.

**💻 Technical:** This is a deterministic computational engine, not a probabilistic black box. AI/ML models (like neural networks) map inputs to outputs based on training data. But geotechnical failures under cryogenic freezing are highly sensitive to boundary conditions. We cannot rely on learned approximations; we need mathematical truth governed by classical PDEs.

> **⚠️ Challenge:** "Physics-Informed Neural Networks (PINNs) can solve PDEs now. Why not use PyTorch instead of manual numerical methods?"
>
> **🛡️ Defense:** PINNs require heavy GPU acceleration and thousands of training epochs just to converge on a single boundary condition. My numerical solver (Backward Euler + Picard) computes the solution deterministically on a standard CPU in seconds. For this specific domain, classical numerical methods are vastly more efficient than deep learning.

## 12. Separation of Concerns (Zero UI in the Physics Folder)

**👶 ELI5:** The Chef (Physics Engine) stays in the kitchen. The Waiter (Streamlit UI) serves the food. If the Chef starts serving tables, the restaurant becomes chaotic. Here, the math files do zero web design, and the web design files do zero math.

**💻 Technical:** The repository follows the MVC (Model-View-Controller) paradigm. The `core_physics/` directory contains purely mathematical dataclasses and vectorized functions. It has no knowledge that Streamlit exists. `app.py` acts as the controller, wiring the UI inputs to the backend engine.

> **⚠️ Challenge:** "Streamlit is reactive and re-runs from top to bottom on every click. Isn't it easier to just put the math functions inside the Streamlit file to share variables easily?"
>
> **🛡️ Defense:** That creates spaghetti code. If I mix physics with UI logic, the physics engine becomes untestable and locked to Streamlit. By decoupling them, I can run my entire pytest suite on the physics engine via terminal without spinning up a web server. If I want to migrate to FastAPI tomorrow, the physics core remains 100% untouched.

## 13. Strict Static Typing in Python (Why Dataclasses?)

**👶 ELI5:** Python normally lets you put a shoe in the fridge, and it won't complain until you try to eat it (and crash). "Static typing" is like putting a lock on the fridge that says "FOOD ONLY". It catches stupid mistakes before the code even runs.

**💻 Technical:** Python is dynamically typed, which is fast for small scripts but a liability for complex physics engines. Passing generic `dict` objects between 5 different modules leads to `KeyError` and silent type failures. The codebase uses strict type hinting and dataclasses (e.g., `StressParams`, `HeaveParams`), resolving Pylance diagnostics and creating a rigid contract between modules.

> **⚠️ Challenge:** "Python is meant to be flexible and dynamic. Aren't you just trying to write Java/C++ in Python and slowing down your coding speed?"
>
> **🛡️ Defense:** Dynamic typing is fast when writing a 50-line script. But when piping a 2D volumetric strain tensor into a visco-plastic creep solver, flexibility is a liability. Explicit dataclasses enable autocomplete, prevent runtime crashes, and serve as self-documenting code. It slows down writing by 5%, but speeds up debugging by 90%.

## 14. Session State vs. Databases for Pipeline Data

**👶 ELI5:** Like a relay race, Module 1 calculates the temperature and directly hands the baton (data) to Module 2 in memory. We don't stop the race to write the data down in a notebook (database) and read it back later. It's instant.

**💻 Technical:** The modules pass complex 2D NumPy arrays between each other sequentially. Instead of an external database like SQLite or Redis, the pipeline leverages Streamlit's `st.session_state` to hold these arrays in active RAM. This allows instant memory-pointer passing without serialization/deserialization overhead.

> **⚠️ Challenge:** "In-memory session state is volatile. If the user refreshes the browser or the app sleeps, all simulation data is wiped. Why not save to a database for persistence?"
>

> **🛡️ Defense:** This is a CPU-bound engineering workstation, not a distributed social media app. Storing 100,000-element floating-point matrices into a relational database at every timestep would introduce massive I/O bottlenecks. Volatility is an acceptable trade-off for zero-latency data visualization.


> **🛡️ Defense:** This is a CPU-bound engineering workstation, not a distributed social media app. Storing 100,000-element floating-point matrices into a relational database at every timestep would introduce massive I/O bottlenecks. Volatility is an acceptable trade-off for zero-latency data visualization.

---

## ✅ Verification, Test Conditions & Failure Envelope

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

---

## 🔒 Security & Reliability Notes

- No hardcoded secrets, credentials, or API keys were identified in repository source files during audit.
- No use of `eval`, `exec`, unsafe deserialization, or shell-command injection patterns was identified in core Python modules.
- The simulator performs in-memory numerical computation; it does not accept untrusted network payloads or execute external commands.
- Input validation guards were added to fail fast on invalid numerical domains (e.g., non-positive timesteps, singular constitutive parameters, invalid SFCC bounds) to prevent silent NaN propagation.

---

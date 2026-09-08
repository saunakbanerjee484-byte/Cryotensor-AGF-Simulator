"""
core_physics
============
Pure, CPU-bound, vectorized (NumPy/SciPy) physics engines for the
CryoTensor-AGF-Simulator. NO Streamlit / UI / plotting code lives here --
see ui_components/ for presentation logic. NO machine learning, NO GPU
dependencies: every module is a deterministic classical-physics solver.

Modules:
    phase_change        - Module 1: Transient Stefan phase-change (Apparent Heat Capacity Method)
    cryosuction          - Module 2: SFCC-coupled cryogenic suction / ice lens growth (stub)
    frost_heave           - Module 3: Volumetric frost heave tensor (stub)
    restrained_stress      - Module 4: Thermo-elastic restrained structural stress (stub)
    thaw_consolidation      - Module 5: Thaw consolidation / excess pore pressure (stub)
"""

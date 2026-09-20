"""
registry.py
=============
The CryoTensor-AGF-Simulator Module Registry -- the single source of truth for
which modules exist, their sidebar display order, and which render() function
each maps to.

Adding a new module to the Command Center means adding exactly one tuple to
MODULES below. app.py's sidebar generation and dispatch loop never need to
change -- they simply unpack this list.

Each entry is (display_name, render_fn_or_None, is_implemented). Modules not
yet implemented pass `None` for render_fn; app.py falls through to
`render_placeholder()`, which preserves the standard [7, 3] Command Center
layout so the UI never jumps or degrades as new modules land.
"""

from __future__ import annotations
import streamlit as st

from modules import module1, module2, module3, module4

MODULES: list[tuple[str, "callable | None", bool]] = [
    ("1. Transient Stefan Phase-Change Matrix", module1.render, True),
    ("2. SFCC Cryosuction Solver", module2.render, True),
    ("3. Volumetric Frost Heave Tensor", module3.render, True),
    ("4. Thermo-Elastic Restrained Stress", module4.render, True),
    ("5. Thaw Consolidation Simulator", None, False),
    ("6. Command Center Master Export", None, False),
]


def render_placeholder(module_name: str) -> None:
    """
    Standard "under construction" screen for a not-yet-implemented module.
    Rigidly preserves the [7, 3] column split so the Command Center's UI/UX
    never degrades as Modules 4-6 are stubbed in ahead of their physics.
    """
    st.caption(f"Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — {module_name}")
    col_sim, col_input = st.columns([7, 3], gap="medium")

    with col_input:
        st.subheader("⚙️ Control Panel")
        st.info(
            "This module's parameter inputs will appear here once its physics "
            "engine is implemented. Its contract (parameter dataclass and planned "
            "public API) already exists in `core_physics/`, so no downstream "
            "rework will be needed when it lands."
        )

    with col_sim:
        st.warning(
            f"🚧 **{module_name}** is under construction. The physics engine and "
            "UI for this module are scheduled for a future development phase."
        )

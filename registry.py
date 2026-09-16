from __future__ import annotations
import streamlit as st
from typing import Callable, Any

from modules import module1, module2, module3

# FIXED: Replaced "callable | None" with Callable[..., Any] | None
MODULES: list[tuple[str, Callable[..., Any] | None, bool]] = [
    ("1. Transient Stefan Phase-Change Matrix", module1.render, True),
    ("2. SFCC Cryosuction Solver", module2.render, True),
    ("3. Volumetric Frost Heave Tensor", module3.render, True),
    ("4. Thermo-Elastic Restrained Stress", lambda: render_placeholder("4. Thermo-Elastic Restrained Stress"), True),
    ("5. Thaw Consolidation Simulator", lambda: render_placeholder("5. Thaw Consolidation Simulator"), True),
    ("6. Command Center Master Export", lambda: render_placeholder("6. Command Center Master Export"), True),
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
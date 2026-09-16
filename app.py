"""
app.py
======
CryoTensor-AGF-Simulator -- Streamlit Entry Point ("The Command Center").

This file is intentionally thin: it owns page config, initializes the shared
cross-module data pipeline (state_pipeline.py), and dispatches to whichever
module the sidebar selects, per the registry defined in registry.py. It does
not contain any physics or per-module UI layout -- those live in
core_physics/ and modules/ respectively.

Adding Module 4, 5, 6, etc. never requires editing this file: add one tuple
to registry.MODULES and implement modules/moduleN.render().
"""

from __future__ import annotations
import streamlit as st

from registry import MODULES, render_placeholder
from state_pipeline import init_shared_state


# ------------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="CryoTensor-AGF-Simulator",
    page_icon="🧊",
    layout="wide",
)

# ------------------------------------------------------------------------------------
# Shared cross-module data pipeline (see state_pipeline.py for the memory-efficiency
# and hand-off standard this enforces).
# ------------------------------------------------------------------------------------
init_shared_state()

# ------------------------------------------------------------------------------------
# SYSTEM NAVIGATION CONSOLE (SIDEBAR) -- dynamically generated from registry.MODULES
# ------------------------------------------------------------------------------------
_labels = [name for name, _render_fn, _implemented in MODULES]
active_module = st.sidebar.radio("⚙️ SYSTEM NAVIGATION CONSOLE", _labels)

st.title("🧊 CryoTensor-AGF-Simulator")

# ------------------------------------------------------------------------------------
# Dispatch -- app.py never branches on module identity beyond this loop. New modules
# are added to registry.MODULES only; this code is stable regardless of module count.
# ------------------------------------------------------------------------------------
for name, render_fn, implemented in MODULES:
    if name == active_module:
        if implemented and render_fn is not None:
            render_fn()
        else:
            render_placeholder(name)
        break

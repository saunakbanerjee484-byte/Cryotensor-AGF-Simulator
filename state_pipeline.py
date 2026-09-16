"""
state_pipeline.py
====================
The CryoTensor-AGF-Simulator's shared data-handoff standard between modules,
built on `st.session_state`.

Why this exists
-----------------
Module 3 (and eventually 4 and 5) need Module 1's temperature field, and later
modules will need Module 2's suction/moisture field. Without a single agreed
contract, every module would invent its own session_state keys, and each new
module would need to know the internal result-dict shape of every module
upstream of it. This file is that single contract.

Memory-efficiency standard (i3-class hardware)
-------------------------------------------------
`st.session_state` already persists exactly one copy of whatever object is
assigned to it -- assigning a NumPy array (or a list of them) to a
session_state key does NOT duplicate the underlying buffer. The rule this
module enforces to keep that true instead of accidentally regressing it:

    1. PUBLISH BY REFERENCE, NEVER BY COPY.
       `publish_thermal_field()` stores the exact `T_history` list (and its
       array elements) that Module 1 already built for its own plotting --
       it does not `.copy()` them. Downstream modules read the same
       in-memory buffers.

    2. CONSUMERS NEVER MUTATE WHAT THEY READ.
       Every function in `core_physics/frost_heave.py` (and, by convention,
       any future consumer) builds new arrays via vectorized NumPy
       expressions (`np.where`, arithmetic, `np.gradient`, ...) rather than
       writing into `T_curr` / `T_prev` in place. As long as consumers keep
       this rule, "publish by reference" is safe -- nothing downstream can
       corrupt Module 1's own copy of its results.

    3. ONE SHARED SLOT PER FIELD TYPE, OVERWRITTEN ON RE-RUN.
       Re-running Module 1 overwrites the single `thermal` slot below rather
       than accumulating a growing history of past runs -- old field data is
       released for garbage collection immediately, so the app's peak memory
       stays bounded by "one run's worth of grids" regardless of how many
       times the user clicks Run over a session.

    4. HAND OFF THE MINIMUM, NOT THE MAXIMUM.
       Only the last two recorded frames are actually needed downstream
       (Module 3's front-velocity estimate is a two-frame finite difference).
       Modules should slice what they need (`T_history[-1]`, `T_history[-2]`)
       at the point of use rather than requesting the full history be
       duplicated into a second container.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import streamlit as st

_THERMAL_KEY = "cryotensor_shared_thermal"
_SUCTION_KEY = "cryotensor_shared_suction"


def init_shared_state() -> None:
    """Call once near the top of app.py. Idempotent -- safe to call every rerun."""
    if _THERMAL_KEY not in st.session_state:
        st.session_state[_THERMAL_KEY] = None
    if _SUCTION_KEY not in st.session_state:
        st.session_state[_SUCTION_KEY] = None


# =====================================================================================
# Module 1 -> downstream: thermal field (temperature history)
# =====================================================================================
def publish_thermal_field(T_history: list[np.ndarray], t: np.ndarray,
                           x: np.ndarray, y: np.ndarray) -> None:
    """
    Called by modules/module1.py immediately after a successful Stefan solve.
    Stores REFERENCES to Module 1's own result arrays (see module docstring,
    point 1) -- this call does not copy any grid data.
    """
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    st.session_state[_THERMAL_KEY] = {
        "T_history": T_history,
        "t": t,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "source_module": "module1",
    }


def get_thermal_field() -> Optional[dict[str, Any]]:
    """
    Returns the shared thermal-field dict (see `publish_thermal_field`), or
    `None` if Module 1 has not yet produced one in this session. Consumers
    must not mutate any array in the returned dict (see module docstring,
    point 2).
    """
    return st.session_state.get(_THERMAL_KEY)


def has_thermal_field() -> bool:
    return get_thermal_field() is not None


# =====================================================================================
# Module 2 -> downstream: suction / moisture field (reserved for Modules 4/5)
# =====================================================================================
def publish_suction_field(psi_history: list[np.ndarray], theta_history: list[np.ndarray],
                           t: np.ndarray, z: np.ndarray) -> None:
    """Called by modules/module2.py after a successful Richards' equation solve."""
    dz = float(z[1] - z[0])
    st.session_state[_SUCTION_KEY] = {
        "psi_history": psi_history,
        "theta_history": theta_history,
        "t": t,
        "z": z,
        "dz": dz,
        "source_module": "module2",
    }


def get_suction_field() -> Optional[dict[str, Any]]:
    return st.session_state.get(_SUCTION_KEY)


def has_suction_field() -> bool:
    return get_suction_field() is not None

"""
core_physics/thaw_consolidation/boundaries.py
=================================================
Vectorized thaw-front extraction and moving-boundary logic for Module 5.

Zero spatial Python loops anywhere in this file: front extraction across ALL
recorded timesteps happens in one `np.argmax(..., axis=1)` call, and the
per-step active/frozen partition used by solver.py is a single boolean mask.
"""

from __future__ import annotations

import numpy as np


def extract_thaw_front_indices(T_history_2d: np.ndarray, thaw_temp_c: float = 0.0) -> np.ndarray:
    """
    Extract the thaw-front node index X_f simultaneously for every recorded
    timestep -- no loop over time or depth.

    `T_history_2d` : (n_time, n_z) array, temperature [degC] vs. depth node,
    one row per recorded time (already resampled/extracted from Module 1's
    2D spatial `T_history` by the caller -- see modules/module5.py).

    Depth index 0 is the ground surface, which thaws first; the front
    advances downward. A node is "thawed" once T > thaw_temp_c. The front is
    the first still-FROZEN node (the boundary between the thawed layer above
    it and the frozen mass below) -- NOT the first thawed node, since index 0
    is thawed from the very first moment thaw begins and `argmax` on the
    thawed mask alone would trivially return 0 forever.
    """
    is_frozen = T_history_2d <= thaw_temp_c                 # (n_time, n_z) boolean, one op
    any_frozen = is_frozen.any(axis=1)                       # (n_time,)
    first_frozen_idx = np.argmax(is_frozen, axis=1)            # (n_time,) -- first True per row
    n_z = T_history_2d.shape[1]
    return np.where(any_frozen, first_frozen_idx, n_z - 1)      # fully thawed row -> front at bottom


def thaw_front_depth_m(front_idx: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Convert front node indices to physical depth [m], one vectorized gather."""
    return z[front_idx]


def resample_front_to_solver_grid(front_depth_m_module1: np.ndarray, t_module1: np.ndarray,
                                   t_solver: np.ndarray, z_solver: np.ndarray) -> np.ndarray:
    """
    Module 1's recorded frames live on their own time grid AND their own depth
    resolution (Module 1's y-array), which generally differs from Module 5's
    own Backward-Euler solver grid. To stay dimensionally correct, interpolate
    the front's PHYSICAL DEPTH [m] (not a raw node index, which would silently
    assume matching resolutions) onto the solver's time grid in one vectorized
    `np.interp` call, then convert to the solver's own nearest node index.
    """
    depth_interp_m = np.interp(t_solver, t_module1, front_depth_m_module1)
    dz_solver = float(z_solver[1] - z_solver[0])
    idx = np.round(depth_interp_m / dz_solver).astype(int)
    return np.clip(idx, 0, len(z_solver) - 1)


def active_thawed_mask(front_idx_at_step: int, n_nodes: int) -> np.ndarray:
    """
    Boolean mask, length n_nodes: True for nodes at or above the current thaw
    front (already thawed, participating in consolidation), False below it
    (still frozen -- rigid, no excess pore pressure yet). One vectorized
    comparison, no loop.
    """
    node_idx = np.arange(n_nodes)
    return node_idx <= front_idx_at_step


def newly_thawed_mask(front_idx_prev: int, front_idx_curr: int, n_nodes: int) -> np.ndarray:
    """
    Boolean mask for nodes that crossed from frozen to thawed THIS step -- these
    are where Morgenstern-Nixon instantaneous excess-pore-pressure generation
    is applied (solver.py). One vectorized comparison, no loop.
    """
    node_idx = np.arange(n_nodes)
    return (node_idx > front_idx_prev) & (node_idx <= front_idx_curr)
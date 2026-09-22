"""
core_physics/thaw_consolidation/physics.py
==============================================
Dynamic coefficient-of-consolidation, void-ratio-collapse, and staggered-grid
averaging for Module 5. Every function is a single vectorized NumPy
expression over the whole depth array -- no spatial loops.

Governing relations
---------------------
e-log(k) permeability model (feature 2):
    cv(e) = cv0 * 10 ** [(e - e0) / Ck]

Void-ratio collapse (standard e-log(sigma') consolidation line), driven by
the effective stress that develops as excess pore pressure dissipates:
    e(sigma') = e0 - Cc * log10(sigma' / sigma'_ref) ,  sigma' >= sigma'_ref

Morgenstern-Nixon instantaneous excess-pore-pressure generation: the instant
a node thaws, its ice-matrix structure collapses under the full overburden
before any drainage can occur (an undrained-loading assumption), so:
    u_excess(z, t_thaw) = sigma'_0(z)          (full pre-thaw effective overburden)
"""

from __future__ import annotations

import numpy as np


def overburden_effective_stress_profile(z: np.ndarray, gamma_eff: float) -> np.ndarray:
    """sigma'_0(z) = gamma_eff * z [Pa], vectorized over the depth array."""
    return gamma_eff * z


def dynamic_cv(e_field: np.ndarray, e0: "float | np.ndarray", cv0: "float | np.ndarray",
                Ck: "float | np.ndarray") -> np.ndarray:
    """
    Feature 2: void-ratio-dependent permeability / consolidation coefficient.
        cv(e) = cv0 * 10 ** [(e - e0) / Ck]
    All three parameters may be scalars or per-node arrays (Feature 1,
    stratified profiles) -- ordinary NumPy broadcasting handles both.
    """
    return cv0 * np.power(10.0, (e_field - e0) / Ck)


def staggered_midpoint_average(field_array: np.ndarray) -> np.ndarray:
    """
    Arithmetic staggered-grid average onto cell midpoints, exactly as specified:
        cv_mid[i] = 0.5 * (cv[i] + cv[i+1])
    Pure array slicing -- no loop, length (n-1) from an input of length n.
    """
    return 0.5 * (field_array[:-1] + field_array[1:])


def instantaneous_thaw_void_ratio(e0_field: "float | np.ndarray", alpha: float) -> np.ndarray:
    """
    The instantaneous volumetric thaw strain (Morgenstern-Nixon): the moment a
    node crosses the thaw front, its ice-rich structure collapses by a strain
    of `alpha` before any pore-pressure dissipation has occurred -- tied
    directly to the same (e0-e)/(1+e0) strain convention used in the
    settlement integral, so alpha is exactly the local volumetric strain at
    the instant of thaw:
        e_thaw = e0 - alpha * (1 + e0)
    """
    return e0_field - alpha * (1.0 + np.asarray(e0_field, dtype=float))


def void_ratio_from_effective_stress(sigma_eff: np.ndarray, sigma_ref: np.ndarray,
                                      e_thaw_field: np.ndarray, Cc: "float | np.ndarray") -> np.ndarray:
    """
    Subsequent gradual collapse ALONG the e-log(sigma') compression line as
    effective stress rebuilds from its post-thaw near-zero value (sigma_ref)
    back up toward the full overburden (and beyond, under any surcharge):
        e(sigma') = e_thaw - Cc * log10(sigma'/sigma_ref) ,  sigma' >= sigma_ref
    Anchored at (e_thaw, sigma_ref) rather than (e0, sigma'_0) -- e0 is the
    FROZEN state, not a point on this compression line; e_thaw is.
    """
    ratio = np.clip(sigma_eff, sigma_ref, None) / np.clip(sigma_ref, 1.0e-3, None)
    return e_thaw_field - Cc * np.log10(np.clip(ratio, 1.0e-6, None))


def settlement_from_void_ratio_history(e_history_2d: np.ndarray, e0_field: "float | np.ndarray",
                                        z: np.ndarray) -> np.ndarray:
    """
    Feature/Step 4: the vectorized settlement integral.
        S(t) = integral_0^L  (e0 - e(z,t)) / (1 + e0)  dz
    `e_history_2d` has shape (n_frames, n_z). The whole 2D matrix collapses to
    the 1D settlement time-series S(t) in one vectorized trapezoidal-rule
    reduction (`np.diff` + a single `.sum(axis=1)`) -- no loop over frames,
    no loop over depth. (Implemented directly rather than via `np.trapz`,
    which was removed in newer NumPy releases in favor of `np.trapezoid`;
    this formulation is version-independent.)
    """
    integrand = (e0_field - e_history_2d) / (1.0 + e0_field)   # (n_frames, n_z), broadcasts e0_field
    dz = np.diff(z)                                              # (n_z-1,)
    segment_avg = 0.5 * (integrand[:, :-1] + integrand[:, 1:])   # (n_frames, n_z-1)
    return np.sum(segment_avg * dz[None, :], axis=1)              # (n_frames,)
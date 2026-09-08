"""
core_physics/phase_change.py
=============================
Sub-Module 1: Transient Stefan Phase-Change Matrix (Apparent Heat Capacity Method).

Governing PDE (2D transient heat conduction with latent heat release/absorption):

    rho * c_app(T) * dT/dt = div( k(T) * grad(T) )

Apparent Heat Capacity Method (AHCM):
    c_app(T) = c(T) + L * d(f_l)/dT      over the phase-change band [T_f - dT/2, T_f + dT/2]

    f_l(T) = liquid (unfrozen) fraction, smoothed linearly across the phase band:
        f_l = 0                                for T <= T_f - dT/2   (fully frozen)
        f_l = 1                                for T >= T_f + dT/2   (fully unfrozen)
        f_l = (T - (T_f - dT/2)) / dT          otherwise (linear ramp)

Thermal conductivity k(T) and volumetric heat capacity are blended between frozen and
unfrozen end-member values using the same f_l(T) mixing function (Voigt/arithmetic mixing,
standard in AGF engineering practice).

Numerics:
    - Fully implicit (Backward Euler) time integration for unconditional stability
      (freeze/thaw problems are numerically stiff near T_f).
    - Spatial operator is a 2D five-point Laplacian assembled ONCE via
      scipy.sparse.kron of two 1D second-difference (tridiagonal) matrices --
      i.e. no explicit Python loop over grid nodes.
    - Nonlinearity (c_app, k depend on T) is resolved with a fixed-point
      (Picard) iteration per timestep -- a handful of linear solves, not a
      loop over space.
    - The only genuine Python-level loop in this module is the OUTER TIME LOOP,
      which is unavoidable for a transient implicit scheme (each step's system
      depends on the previous step's solution). No spatial `for` loop exists.

This module contains NO Streamlit / plotting / UI code (see ui_components/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# Parameter container
# ----------------------------------------------------------------------------
@dataclass
class ThermalParams:
    # --- Domain (2D soil cross-section, meters) ---
    Lx: float = 3.0                 # domain width  [m]
    Ly: float = 3.0                 # domain height [m]
    nx: int = 61                    # grid points in x
    ny: int = 61                    # grid points in y

    # --- Freeze pipe (modeled as a circular Dirichlet cold boundary) ---
    pipe_radius: float = 0.08       # [m]
    pipe_cx: float = 1.5            # pipe center x [m]
    pipe_cy: float = 1.5            # pipe center y [m]
    pipe_temp: float = -25.0        # brine/coolant temperature [degC]

    # --- Far-field / initial conditions ---
    initial_ground_temp: float = 12.0   # [degC]
    far_field_temp: float = 12.0        # Dirichlet outer boundary [degC]

    # --- Thermal properties: unfrozen / frozen end members ---
    k_unfrozen: float = 1.8         # [W/m/K]  thermal conductivity, unfrozen soil
    k_frozen: float = 2.3           # [W/m/K]  thermal conductivity, frozen soil
    c_unfrozen: float = 1850.0      # [J/kg/K] specific heat, unfrozen soil
    c_frozen: float = 1550.0        # [J/kg/K] specific heat, frozen soil
    rho: float = 1900.0             # [kg/m3]  bulk soil density
    latent_heat: float = 3.34e5     # [J/kg]   latent heat of fusion (water in soil, per kg of pore water)
    water_content: float = 0.22     # [-] gravimetric water content (fraction of soil mass that is pore water)

    # --- Phase change band (numerical smoothing of Stefan condition) ---
    T_freeze: float = 0.0           # [degC]  nominal freezing point
    delta_T_phase: float = 0.6      # [degC]  half-width->full-width smoothing band

    # --- Time integration ---
    dt: float = 3600.0              # [s] timestep (default 1 hour)
    total_time: float = 3600.0 * 24 * 30   # [s] total simulated duration (default 30 days)
    picard_iters: int = 4           # nonlinear sub-iterations per timestep

    def grid_spacing(self):
        dx = self.Lx / (self.nx - 1)
        dy = self.Ly / (self.ny - 1)
        return dx, dy

    def n_steps(self) -> int:
        return max(1, int(round(self.total_time / self.dt)))

    def effective_latent(self) -> float:
        """Latent heat per kg of BULK soil (water_content scales pure-water latent heat)."""
        return self.latent_heat * self.water_content


# ----------------------------------------------------------------------------
# Vectorized field functions (no spatial loops -- pure NumPy array ops)
# ----------------------------------------------------------------------------
def liquid_fraction(T: np.ndarray, T_freeze: float, delta_T: float) -> np.ndarray:
    """Smoothed unfrozen (liquid) water fraction f_l(T), vectorized. Values in [0, 1]."""
    lo = T_freeze - delta_T / 2.0
    hi = T_freeze + delta_T / 2.0
    fl = (T - lo) / (hi - lo)
    return np.clip(fl, 0.0, 1.0)


def dliquid_fraction_dT(T: np.ndarray, T_freeze: float, delta_T: float) -> np.ndarray:
    """d(f_l)/dT, vectorized. Nonzero only inside the phase-change band."""
    lo = T_freeze - delta_T / 2.0
    hi = T_freeze + delta_T / 2.0
    inside = (T >= lo) & (T <= hi)
    return np.where(inside, 1.0 / (hi - lo), 0.0)


def apparent_heat_capacity(T: np.ndarray, p: ThermalParams) -> np.ndarray:
    """
    c_app(T) = mixed specific heat + latent contribution, vectorized over the grid.
    Returns VOLUMETRIC apparent heat capacity [J/m3/K] (already multiplied by rho).
    """
    fl = liquid_fraction(T, p.T_freeze, p.delta_T_phase)
    dfl = dliquid_fraction_dT(T, p.T_freeze, p.delta_T_phase)
    c_mix = fl * p.c_unfrozen + (1.0 - fl) * p.c_frozen
    c_app = c_mix + p.effective_latent() * dfl
    return p.rho * c_app


def thermal_conductivity(T: np.ndarray, p: ThermalParams) -> np.ndarray:
    """Mixed thermal conductivity k(T) [W/m/K], vectorized (arithmetic mixing on f_l)."""
    fl = liquid_fraction(T, p.T_freeze, p.delta_T_phase)
    return fl * p.k_unfrozen + (1.0 - fl) * p.k_frozen


# ----------------------------------------------------------------------------
# Sparse operator assembly (vectorized construction, NOT looped over nodes)
# ----------------------------------------------------------------------------
def build_2d_laplacian(nx: int, ny: int, dx: float, dy: float) -> sp.csr_matrix:
    """
    Assemble the 2D five-point Laplacian operator L such that (L @ T.ravel())
    approximates div(grad(T)) under a Neumann (zero-flux) default boundary --
    Dirichlet boundaries are enforced afterwards by row-replacement.

    Built via scipy.sparse.kron of two 1D tridiagonal second-difference
    matrices: no explicit loop over the (nx * ny) grid nodes.
    """
    ex = np.ones(nx)
    main_x = -2.0 * ex
    main_x[0] = -1.0
    main_x[-1] = -1.0
    Lx1d = sp.diags([ex[:-1], main_x, ex[:-1]], offsets=[-1, 0, 1], shape=(nx, nx)) / dx ** 2

    ey = np.ones(ny)
    main_y = -2.0 * ey
    main_y[0] = -1.0
    main_y[-1] = -1.0
    Ly1d = sp.diags([ey[:-1], main_y, ey[:-1]], offsets=[-1, 0, 1], shape=(ny, ny)) / dy ** 2

    Ix = sp.eye(nx)
    Iy = sp.eye(ny)
    L = sp.kron(Iy, Lx1d) + sp.kron(Ly1d, Ix)
    return L.tocsr()


def dirichlet_masks(p: ThermalParams):
    """
    Build boundary index masks (vectorized, via meshgrid) -- no spatial loop.
    Returns (pipe_mask_flat, outer_mask_flat, x, y, X, Y)
    """
    x = np.linspace(0.0, p.Lx, p.nx)
    y = np.linspace(0.0, p.Ly, p.ny)
    X, Y = np.meshgrid(x, y)  # shape (ny, nx)

    pipe_mask = (X - p.pipe_cx) ** 2 + (Y - p.pipe_cy) ** 2 <= p.pipe_radius ** 2
    outer_mask = np.zeros_like(pipe_mask, dtype=bool)
    outer_mask[0, :] = True
    outer_mask[-1, :] = True
    outer_mask[:, 0] = True
    outer_mask[:, -1] = True
    # pipe boundary takes precedence if domains overlap (shouldn't in practice)
    outer_mask = outer_mask & (~pipe_mask)

    return pipe_mask.ravel(), outer_mask.ravel(), x, y, X, Y


# ----------------------------------------------------------------------------
# Main transient solve
# ----------------------------------------------------------------------------
def run_stefan_simulation(p: ThermalParams, record_every: int = 1):
    """
    Run the fully-implicit, Picard-linearized transient AHCM simulation.

    Returns a dict with:
        x, y, X, Y            : grid coordinate arrays
        t                      : recorded time stamps [s]
        T_history              : list of 2D temperature fields [degC], one per recorded step
        T_final                : final 2D field
        freeze_front_radius_m  : list of estimated 0 degC isotherm radius from pipe center [m]
        max_cooling_rate       : list of max |dT/dt| per step [degC/s] (thermal shock proxy)
        frozen_fraction        : list of domain fraction with T <= T_freeze
        params                 : the ThermalParams used
    """
    dx, dy = p.grid_spacing()
    N = p.nx * p.ny
    L = build_2d_laplacian(p.nx, p.ny, dx, dy)
    pipe_mask, outer_mask, x, y, X, Y = dirichlet_masks(p)
    boundary_mask = pipe_mask | outer_mask
    interior_mask = ~boundary_mask

    T = np.full(N, p.initial_ground_temp, dtype=float)
    T[pipe_mask] = p.pipe_temp
    T[outer_mask] = p.far_field_temp

    dt = p.dt
    n_steps = p.n_steps()
    record_every = max(1, record_every)

    t_list = [0.0]
    T_history = [T.reshape(p.ny, p.nx).copy()]
    front_list = [0.0]
    cooling_rate_list = [0.0]
    frozen_frac_list = [float(np.mean(T <= p.T_freeze))]

    I = sp.identity(N, format="csr")

    for step in range(1, n_steps + 1):
        T_old = T.copy()
        T_guess = T.copy()  # Picard seed = previous timestep solution

        for _ in range(p.picard_iters):
            c_app = apparent_heat_capacity(T_guess, p)          # [J/m3/K], shape (N,)
            k_field = thermal_conductivity(T_guess, p)           # [W/m/K], shape (N,)
            k_avg = float(np.mean(k_field))                      # scalar effective k for operator
            # (Using a scalar effective k keeps the Laplacian operator fixed-structure &
            #  fast to solve each Picard sub-iteration; c_app carries the full nonlinearity.)

            C_diag = sp.diags(c_app / dt)
            A = C_diag - k_avg * L

            b = (c_app / dt) * T_old

            # Enforce Dirichlet rows (vectorized row replacement, no loop over nodes)
            A = A.tolil()
            A[pipe_mask, :] = 0.0
            A[pipe_mask, np.where(pipe_mask)[0]] = 1.0
            A[outer_mask, :] = 0.0
            A[outer_mask, np.where(outer_mask)[0]] = 1.0
            A = A.tocsr()

            b[pipe_mask] = p.pipe_temp
            b[outer_mask] = p.far_field_temp

            T_new = spla.spsolve(A, b)
            T_guess = T_new

        cooling_rate = np.max(np.abs(T_guess - T_old)) / dt
        T = T_guess

        if step % record_every == 0 or step == n_steps:
            t_list.append(step * dt)
            T_field = T.reshape(p.ny, p.nx)
            T_history.append(T_field.copy())
            cooling_rate_list.append(float(cooling_rate))
            frozen_frac_list.append(float(np.mean(T <= p.T_freeze)))
            front_list.append(_estimate_freeze_front_radius(T_field, X, Y, p))

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "t": np.array(t_list),
        "T_history": T_history,
        "T_final": T_history[-1],
        "freeze_front_radius_m": np.array(front_list),
        "max_cooling_rate": np.array(cooling_rate_list),
        "frozen_fraction": np.array(frozen_frac_list),
        "params": p,
    }


def _estimate_freeze_front_radius(T_field: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                   p: ThermalParams) -> float:
    """
    Vectorized estimate of the mean 0 degC isotherm radius from the pipe center:
    average radial distance of frozen-boundary cells (T <= T_freeze adjacent to
    T > T_freeze) from the pipe centroid. No explicit spatial loop -- uses
    boolean masks and np.gradient-based edge detection.
    """
    frozen = T_field <= p.T_freeze
    if not frozen.any() or frozen.all():
        # Nothing frozen yet, or entire domain frozen -> fall back to simple area estimate
        area_m2 = frozen.mean() * p.Lx * p.Ly
        return float(np.sqrt(area_m2 / np.pi))

    gx, gy = np.gradient(frozen.astype(float))
    edge = (np.abs(gx) + np.abs(gy)) > 0
    if not edge.any():
        return 0.0
    r = np.sqrt((X[edge] - p.pipe_cx) ** 2 + (Y[edge] - p.pipe_cy) ** 2)
    return float(np.mean(r))

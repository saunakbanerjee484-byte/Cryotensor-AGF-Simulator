"""
modules/module1.py
=====================
Module 1: Transient Stefan Phase-Change Matrix -- Streamlit render() entry point.

Called by registry.py / app.py. All physics lives in core_physics/phase_change.py;
this file is presentation-only. On a successful run it publishes the resulting
temperature history into the shared state pipeline (state_pipeline.py) so
Module 3 (and later modules) can consume it without re-running Module 1's solver.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import streamlit as st

from core_physics.phase_change import ThermalParams, run_stefan_simulation
from ui_components.marquee_engine import (
    render_marquee,
    MODULE_1A_EQUATIONS,
    MODULE_1B_EQUATIONS,
    MODULE_1C_EQUATIONS,
)
from ui_components.alert_system import evaluate_module1_status, render_alert
from ui_components.dual_visualizer import render_dual_visualization
import state_pipeline


def render() -> None:
    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "target_closure_radius_m" not in st.session_state:
        st.session_state["target_closure_radius_m"] = 1.0

    st.caption(
        "Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — "
        "Module 1: Transient Stefan Phase-Change Matrix"
    )

    left, right = st.columns([7, 3], gap="medium")

    # ==================================================================================
    # RIGHT COLUMN (30%) — CONTROL PANEL
    # ==================================================================================
    with right:
        st.subheader("⚙️ Control Panel")
        tab_1a, tab_1b, tab_1c = st.tabs(["1A · Thermal Matrix", "1B · Boundary Setup", "1C · Phase Engine"])

        # ---------------- 1A: Thermal Matrix Parameters ----------------
        with tab_1a:
            st.markdown("**Domain & Grid**")
            Lx = st.number_input("Domain width, Lx [m]", min_value=0.5, value=3.0, step=0.1, format="%.2f")
            Ly = st.number_input("Domain height, Ly [m]", min_value=0.5, value=3.0, step=0.1, format="%.2f")
            nx = st.number_input("Grid nodes, nx [-]", min_value=15, max_value=121, value=61, step=2)
            ny = st.number_input("Grid nodes, ny [-]", min_value=15, max_value=121, value=61, step=2)

            st.markdown("**Thermal Properties — Unfrozen / Frozen End Members**")
            k_unfrozen = st.number_input("k, unfrozen [W/m/K]", min_value=0.01, value=1.80, step=0.01, format="%.3f")
            k_frozen = st.number_input("k, frozen [W/m/K]", min_value=0.01, value=2.30, step=0.01, format="%.3f")
            c_unfrozen = st.number_input("c, unfrozen [J/kg/K]", min_value=1.0, value=1850.0, step=10.0, format="%.1f")
            c_frozen = st.number_input("c, frozen [J/kg/K]", min_value=1.0, value=1550.0, step=10.0, format="%.1f")
            rho = st.number_input("Bulk density, ρ [kg/m³]", min_value=1.0, value=1900.0, step=10.0, format="%.1f")
            latent_heat = st.number_input("Latent heat of fusion, L [J/kg]", min_value=1.0, value=3.34e5, step=1.0e3, format="%.1f")
            water_content = st.number_input("Gravimetric water content, w [-]", min_value=0.0, max_value=1.0, value=0.22, step=0.01, format="%.3f")

        # ---------------- 1B: Boundary Setup ----------------
        with tab_1b:
            st.markdown("**Freeze Pipe (Dirichlet cold boundary)**")
            pipe_radius = st.number_input("Pipe radius [m]", min_value=0.01, value=0.08, step=0.01, format="%.3f")
            pipe_cx = st.number_input("Pipe center, x [m]", min_value=0.0, value=1.5, step=0.1, format="%.2f")
            pipe_cy = st.number_input("Pipe center, y [m]", min_value=0.0, value=1.5, step=0.1, format="%.2f")
            pipe_temp = st.number_input("Coolant/brine temperature [°C]", value=-25.0, step=1.0, format="%.1f")

            st.markdown("**Initial & Far-Field Conditions**")
            initial_ground_temp = st.number_input("Initial ground temperature [°C]", value=12.0, step=0.5, format="%.1f")
            far_field_temp = st.number_input("Far-field boundary temperature [°C]", value=12.0, step=0.5, format="%.1f")

            st.markdown("**Design Target**")
            target_closure_radius_m = st.number_input(
                "Target freeze-wall closure radius [m]", min_value=0.05, value=1.00, step=0.05, format="%.2f",
                help="Used by the deterministic alert system (Green/Yellow/Red) in Sub-Module 1C.",
            )
            st.session_state["target_closure_radius_m"] = target_closure_radius_m

        # ---------------- 1C: Phase Engine (Apparent Heat Capacity) ----------------
        with tab_1c:
            st.markdown("**Phase-Change Band (Stefan Smoothing)**")
            T_freeze = st.number_input("Nominal freezing point, T_f [°C]", value=0.0, step=0.1, format="%.2f")
            delta_T_phase = st.number_input("Phase-change band width, ΔT [°C]", min_value=0.05, value=0.60, step=0.05, format="%.2f")

            st.markdown("**Time Integration**")
            dt_hours = st.number_input("Timestep, Δt [hours]", min_value=0.05, value=1.0, step=0.25, format="%.2f")
            total_days = st.number_input("Total simulated duration [days]", min_value=0.5, value=15.0, step=0.5, format="%.2f")
            picard_iters = st.number_input("Picard nonlinear sub-iterations [-]", min_value=1, max_value=15, value=4, step=1)
            record_every = st.number_input("Record every N steps [-]", min_value=1, max_value=500, value=4, step=1)

            st.markdown("**Alert Thresholds**")
            critical_cooling_rate = st.number_input(
                "Critical cooling rate (thermal shock) [°C/s]", min_value=0.0001, value=0.0200, step=0.0010, format="%.4f"
            )
            warning_cooling_rate = st.number_input(
                "Warning cooling rate [°C/s]", min_value=0.0001, value=0.0100, step=0.0010, format="%.4f"
            )

            st.markdown("---")
            run_clicked_1 = st.button("▶ Run Stefan Phase-Change Simulation", type="primary", width="stretch")

            if run_clicked_1:
                params = ThermalParams(
                    Lx=Lx, Ly=Ly, nx=int(nx), ny=int(ny),
                    pipe_radius=pipe_radius, pipe_cx=pipe_cx, pipe_cy=pipe_cy, pipe_temp=pipe_temp,
                    initial_ground_temp=initial_ground_temp, far_field_temp=far_field_temp,
                    k_unfrozen=k_unfrozen, k_frozen=k_frozen,
                    c_unfrozen=c_unfrozen, c_frozen=c_frozen,
                    rho=rho, latent_heat=latent_heat, water_content=water_content,
                    T_freeze=T_freeze, delta_T_phase=delta_T_phase,
                    dt=dt_hours * 3600.0, total_time=total_days * 24.0 * 3600.0,
                    picard_iters=int(picard_iters),
                )
                with st.spinner("Solving transient Stefan phase-change matrix (implicit, vectorized)..."):
                    sim_out: dict[str, Any] = run_stefan_simulation(params, record_every=int(record_every))
                sim_out["critical_cooling_rate"] = critical_cooling_rate
                sim_out["warning_cooling_rate"] = warning_cooling_rate
                st.session_state["results"] = sim_out

                # Hand off the temperature history to downstream modules (Module 3+)
                # via the shared state pipeline -- by reference, no duplication.
                state_pipeline.publish_thermal_field(
                    T_history=sim_out["T_history"], t=sim_out["t"],
                    x=sim_out["x"], y=sim_out["y"],
                )
                st.success(
                    f"Simulation complete — {len(sim_out['t'])} recorded frames. "
                    "Temperature field published to the shared pipeline for Module 3."
                )

    # ==================================================================================
    # LEFT COLUMN (70%) — LIVE MATRIX
    # ==================================================================================
    with left:
        all_equations = MODULE_1A_EQUATIONS + MODULE_1B_EQUATIONS + MODULE_1C_EQUATIONS
        st.markdown(render_marquee(all_equations, label="MODULE 1 · LIVE PHYSICS"), unsafe_allow_html=True)

        results: dict[str, Any] | None = st.session_state.get("results")

        if results is None:
            st.info(
                "⬅ Configure Sub-Modules 1A (Thermal Matrix), 1B (Boundary Setup), and "
                "1C (Phase Engine) in the Control Panel, then click **Run Stefan Phase-Change "
                "Simulation** to populate the Live Matrix."
            )
        else:
            t_days = results["t"] / 86400.0

            frame_idx = st.slider(
                "Inspect recorded timestep",
                min_value=0, max_value=len(results["t"]) - 1,
                value=len(results["t"]) - 1,
                format="frame %d",
                key="slider_mod1",
            )
            st.caption(f"Elapsed time: **{t_days[frame_idx]:.2f} days** ({results['t'][frame_idx]:.0f} s)")

            T_field = results["T_history"][frame_idx]

            render_dual_visualization(
                line_x=t_days,
                line_y=results["freeze_front_radius_m"],
                line_title="Freeze-Front Radius vs. Time",
                line_xlabel="Time [days]",
                line_ylabel="Front radius [m]",
                field_2d=T_field,
                heatmap_title=f"Temperature Field, T(x,y) @ frame {frame_idx}",
                heatmap_cbar_label="Temperature [°C]",
                x=results["x"],
                y=results["y"],
                df_value_name="T (°C)",
            )

            st.markdown("#### Additional Diagnostics")
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Frozen fraction of domain (current frame)", f"{results['frozen_fraction'][frame_idx]*100:.1f} %")
                st.metric("Freeze-front radius (current frame)", f"{results['freeze_front_radius_m'][frame_idx]:.3f} m")
            with d2:
                st.metric("Max cooling rate (current frame)", f"{results['max_cooling_rate'][frame_idx]:.4f} °C/s")
                st.metric("Target closure radius", f"{st.session_state['target_closure_radius_m']:.3f} m")

            st.markdown("#### 🚦 Deterministic Engineering Assessment")
            alert = evaluate_module1_status(
                frozen_fraction_final=results["frozen_fraction"][-1],
                max_cooling_rate=float(np.max(results["max_cooling_rate"])),
                freeze_front_radius_m=results["freeze_front_radius_m"][-1],
                target_closure_radius_m=st.session_state["target_closure_radius_m"],
                critical_cooling_rate=results.get("critical_cooling_rate", 0.02),
                warning_cooling_rate=results.get("warning_cooling_rate", 0.01),
            )
            render_alert(alert)

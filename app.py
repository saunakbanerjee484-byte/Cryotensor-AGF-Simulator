"""
app.py
======
CryoTensor-AGF-Simulator -- Streamlit Entry Point ("The Command Center").

Layout: single-screen [7, 3] column split.
    - Left  (70%): Live Matrix -- marquee, dual visualization (line + heatmap + dataframe),
                   deterministic engineering assessment (Green/Yellow/Red).
    - Right (30%): Control Panel -- exact numerical inputs (no sliders) for
                   Sub-Modules.
"""

# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ---- Module 1 Imports ----
from core_physics.phase_change import ThermalParams, run_stefan_simulation
from ui_components.marquee_engine import (
    render_marquee,
    MODULE_1A_EQUATIONS,
    MODULE_1B_EQUATIONS,
    MODULE_1C_EQUATIONS,
)
from ui_components.alert_system import evaluate_module1_status, render_alert
from ui_components.dual_visualizer import render_dual_visualization

# ---- Module 2 Imports ----
from core_physics.cryosuction import (
    SFCCParams, 
    BoundaryParams, 
    SolverParams, 
    run_cryosuction_simulation, 
    evaluate_module2_status, 
    render_cryosuction_marquee
)


# ------------------------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------------------------
st.set_page_config(
    page_title="CryoTensor-AGF-Simulator",
    page_icon="🧊",
    layout="wide",
)

# ------------------------------------------------------------------------------------
# SYSTEM NAVIGATION CONSOLE (SIDEBAR)
# ------------------------------------------------------------------------------------
active_module = st.sidebar.radio(
    "⚙️ SYSTEM NAVIGATION CONSOLE", 
    [
        "1. Stefan Phase-Change Matrix", 
        "2. SFCC Cryosuction Solver"
    ]
)

if "results" not in st.session_state:
    st.session_state["results"] = None
if "target_closure_radius_m" not in st.session_state:
    st.session_state["target_closure_radius_m"] = 1.0
if "cryosuction_results" not in st.session_state:
    st.session_state["cryosuction_results"] = None


st.title("🧊 CryoTensor-AGF-Simulator")

# ======================================================================================
# MODULE 1 RUNTIME
# ======================================================================================
if active_module == "1. Stefan Phase-Change Matrix":
    st.caption(
        "Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — "
        "Module 1: Transient Stefan Phase-Change Matrix"
    )

    left, right = st.columns([7, 3], gap="medium")

    # ======================================================================================
    # RIGHT COLUMN (30%) — CONTROL PANEL
    # ======================================================================================
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
                    sim_out: dict[str, Any] = run_stefan_simulation(params, record_every=int(record_every))  # type: ignore
                sim_out["critical_cooling_rate"] = critical_cooling_rate
                sim_out["warning_cooling_rate"] = warning_cooling_rate
                st.session_state["results"] = sim_out
                st.success(f"Simulation complete — {len(sim_out['t'])} recorded frames.")  # type: ignore


    # ======================================================================================
    # LEFT COLUMN (70%) — LIVE MATRIX
    # ======================================================================================
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
            p = results["params"]
            t_days = results["t"] / 86400.0

            frame_idx = st.slider(
                "Inspect recorded timestep",
                min_value=0, max_value=len(results["t"]) - 1,
                value=len(results["t"]) - 1,
                format="frame %d",
                key="slider_mod1"
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

# ======================================================================================
# MODULE 2 RUNTIME
# ======================================================================================
elif active_module == "2. SFCC Cryosuction Solver":
    st.caption(
        "Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — "
        "Module 2: SFCC Cryogenic Suction Solver"
    )
    
    left, right = st.columns([7, 3], gap="medium")

    # -------------------------------------------------------------------------------------
    # RIGHT COLUMN (30%) -- CONTROL PANEL
    # -------------------------------------------------------------------------------------
    with right:
        st.subheader("⚙️ Control Panel")

        with st.expander("2A · SFCC Matrix (Soil Freezing Characteristic Curve)", expanded=True):
            air_entry_kpa = st.number_input("Air Entry Value, AEV [kPa]", min_value=0.0, value=2.0, step=0.1, format="%.2f")
            vg_n = st.number_input("van Genuchten n [-]", min_value=1.01, value=1.80, step=0.01, format="%.3f")
            vg_alpha = st.number_input("van Genuchten α [1/kPa]", min_value=0.0001, value=0.0500, step=0.0010, format="%.4f")
            theta_r = st.number_input("Residual unfrozen water content, θr [-]", min_value=0.0, max_value=0.9, value=0.05, step=0.01, format="%.3f")
            theta_s = st.number_input("Saturated water content, θs [-]", min_value=0.05, max_value=1.0, value=0.42, step=0.01, format="%.3f")
            k_sat = st.number_input("Saturated hydraulic conductivity, k_sat [m/s]", min_value=1.0e-12, value=5.0e-8, step=1.0e-9, format="%.2e")

        with st.expander("2B · Hydrogeological Boundary Setup", expanded=True):
            distance_to_water_table_m = st.number_input("Distance to groundwater table [m]", min_value=0.2, value=3.0, step=0.1, format="%.2f")
            initial_matric_suction_kpa = st.number_input("Initial matric suction [kPa]", min_value=0.0, value=5.0, step=0.5, format="%.2f")
            st.markdown("**Temperature Gradient (mock freezing-front array)**")
            surface_temp_c = st.number_input("Frozen-surface / pipe temperature [°C]", value=-12.0, step=0.5, format="%.1f")
            ground_temp_c = st.number_input("Undisturbed ground temperature [°C]", value=8.0, step=0.5, format="%.1f")
            temp_gradient_c_per_m = st.number_input("Temperature gradient, dT/dz [°C/m]", value=4.0, step=0.1, format="%.2f")
            front_advance_rate = st.number_input(
                "Freeze-front advance rate [m/√day]", min_value=0.01, value=0.35, step=0.01, format="%.2f",
                help="MOCK Stefan-type placeholder for the real T(z,t) field pending hand-off from Module 1.",
            )

        with st.expander("Solver Discretization", expanded=False):
            n_nodes = st.number_input("Depth nodes [-]", min_value=11, max_value=201, value=41, step=2)
            dt_hours = st.number_input("Timestep, Δt [hours]", min_value=0.1, value=2.0, step=0.1, format="%.2f")
            total_days_2 = st.number_input("Total simulated duration [days] (Mod 2)", min_value=0.5, value=10.0, step=0.5, format="%.2f")
            picard_iters_2 = st.number_input("Picard sub-iterations [-] (Mod 2)", min_value=1, max_value=15, value=5, step=1)
            record_every_2 = st.number_input("Record every N steps [-] (Mod 2)", min_value=1, max_value=200, value=6, step=1)

        st.markdown("---")
        st.markdown("### 2C · Vectorized Richards' Engine")
        run_clicked_2 = st.button("🚀 RUN CRYOSUCTION & ICE LENS SIMULATION", type="primary", width="stretch")

        if run_clicked_2:
            sfcc_params = SFCCParams(
                air_entry_kpa=air_entry_kpa, vg_n=vg_n, vg_alpha=vg_alpha,
                theta_r=theta_r, theta_s=theta_s, k_sat=k_sat,
            )
            bnd_params = BoundaryParams(
                distance_to_water_table_m=distance_to_water_table_m,
                initial_matric_suction_kpa=initial_matric_suction_kpa,
                surface_temp_c=surface_temp_c, ground_temp_c=ground_temp_c,
                temp_gradient_c_per_m=temp_gradient_c_per_m,
                front_advance_rate=front_advance_rate,
            )
            solver_params = SolverParams(
                n_nodes=int(n_nodes), dt_hours=dt_hours, total_days=total_days_2,
                picard_iters=int(picard_iters_2), record_every=int(record_every_2),
            )
            with st.spinner("Solving vectorized 1D Richards' equation (implicit, Picard-linearized)..."):
                sim_out: dict[str, Any] = run_cryosuction_simulation(sfcc_params, bnd_params, solver_params)  # type: ignore
            st.session_state["cryosuction_results"] = sim_out
            st.success(f"Simulation complete — {len(sim_out['t'])} recorded frames.")  # type: ignore

    # -------------------------------------------------------------------------------------
    # LEFT COLUMN (70%) -- LIVE SIMULATION / VISUALS
    # -------------------------------------------------------------------------------------
    with left:
        st.markdown(render_cryosuction_marquee(), unsafe_allow_html=True)

        results_mod2: dict[str, Any] | None = st.session_state.get("cryosuction_results")

        if results_mod2 is None:
            st.info(
                "⬅ Configure Sub-Modules 2A (SFCC Matrix), 2B (Boundary Setup), and the solver "
                "discretization in the Control Panel, then click **RUN CRYOSUCTION & ICE LENS "
                "SIMULATION** to populate the Live Simulation panel."
            )
        else:
            p_sfcc = results_mod2["params"]["sfcc"]
            z = results_mod2["z"]
            t_days_mod2 = results_mod2["t"] / 86400.0

            frame_idx_mod2 = st.slider(
                "Inspect recorded timestep", min_value=0, max_value=len(results_mod2["t"]) - 1,
                value=len(results_mod2["t"]) - 1, format="frame %d", key="slider_mod2"
            )
            st.caption(
                f"Elapsed time: **{t_days_mod2[frame_idx_mod2]:.2f} days** — "
                f"mock freeze-front depth: **{results_mod2['front_depth_history'][frame_idx_mod2]:.3f} m**"
            )

            psi_now = results_mod2["psi_history"][frame_idx_mod2]
            theta_now = results_mod2["theta_history"][frame_idx_mod2]

            # ---- Dual visualization: Matplotlib line chart(s) | Seaborn heatmap ----
            col_line, col_heat = st.columns(2)

            with col_line:
                fig_line, (ax_sfcc, ax_profile) = plt.subplots(2, 1, figsize=(6.0, 6.4))

                T_range_c, theta_curve = results_mod2["sfcc_curve"]
                ax_sfcc.plot(T_range_c, theta_curve, color="#1c7c4f", linewidth=1.8)
                ax_sfcc.set_title("Soil Freezing Characteristic Curve", fontsize=10, fontweight="bold")
                ax_sfcc.set_xlabel("Temperature [°C]", fontsize=9)
                ax_sfcc.set_ylabel("Unfrozen water content, θu [-]", fontsize=9)
                ax_sfcc.grid(alpha=0.3)

                ax_profile.plot(psi_now, z, color="#0a4d8c", linewidth=1.8, marker="o", markersize=2.5)
                ax_profile.set_title(f"Matric Suction Profile @ frame {frame_idx_mod2}", fontsize=10, fontweight="bold")
                ax_profile.set_xlabel("Matric suction, ψ [kPa]", fontsize=9)
                ax_profile.set_ylabel("Depth, z [m]", fontsize=9)
                ax_profile.invert_yaxis()
                ax_profile.grid(alpha=0.3)

                fig_line.tight_layout()
                st.pyplot(fig_line, clear_figure=True)
                fig_line.clf()
                plt.close(fig_line)

            with col_heat:
                psi_matrix = np.array(results_mod2["psi_history"]).T   # shape (N_depth, N_frames)
                fig_heat, ax_heat = plt.subplots(figsize=(6.2, 6.4))
                sns.heatmap(
                    psi_matrix, cmap="YlGnBu", cbar_kws={"label": "Matric suction ψ [kPa]"},
                    ax=ax_heat, xticklabels=False, yticklabels=False,
                )
                ax_heat.set_title("Spatio-Temporal Matric Suction Field", fontsize=10, fontweight="bold")
                ax_heat.set_xlabel("Time →", fontsize=9)
                ax_heat.set_ylabel("Depth (top=front, bottom=water table) →", fontsize=9)
                fig_heat.tight_layout()
                st.pyplot(fig_heat, clear_figure=True)
                fig_heat.clf()
                plt.close(fig_heat)

            # ---- Raw data grid: ultra-wide DataFrame + CSV download ----
            st.markdown("#### 📐 Raw Nodal Grid — Matric Suction ψ(z, t) [kPa]")
            df_wide = pd.DataFrame(
                np.round(psi_matrix, 3),
                index=np.round(z, 3),
                columns=[f"t={td:.2f}d" for td in t_days_mod2],
            )
            df_wide.index.name = "Depth z [m]"
            st.dataframe(df_wide, width="stretch", height=280)
            csv_bytes = df_wide.to_csv().encode("utf-8")
            st.download_button(
                "Download full suction/moisture tensor (CSV)", data=csv_bytes,
                file_name="cryosuction_psi_matrix.csv", mime="text/csv",
            )

            # ---- Diagnostics ----
            st.markdown("#### Additional Diagnostics")
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Unfrozen water content θu (current frame, top node)", f"{theta_now[0]:.4f}")
                st.metric("Matric suction ψ (current frame, top node)", f"{psi_now[0]:.2f} kPa")
            with d2:
                st.metric("Moisture influx velocity (current frame)", f"{results_mod2['influx_velocity'][frame_idx_mod2]:.3e} m/s")
                st.metric("Mean suction gradient (current frame)", f"{results_mod2['suction_gradient'][frame_idx_mod2]:.2f} kPa/m")

            # ---- Deterministic engineering assessment ----
            st.markdown("#### 🚦 Deterministic Engineering Assessment")
            mean_influx = float(np.mean(results_mod2["influx_velocity"][-5:]))
            mean_gradient = float(np.mean(results_mod2["suction_gradient"][-5:]))
            status, message = evaluate_module2_status(mean_influx, p_sfcc.k_sat, mean_gradient)

            if status == "red":
                st.error(message)
            elif status == "yellow":
                st.warning(message)
            else:
                st.success(message)
"""
modules/module2.py
=====================
Module 2: SFCC Cryogenic Suction Solver -- Streamlit render() entry point.

Called by registry.py / app.py. All physics lives in core_physics/cryosuction.py;
this file is presentation-only. On a successful run it publishes the resulting
suction/moisture history into the shared state pipeline (reserved for Modules 4/5).
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from core_physics.cryosuction import (
    SFCCParams, BoundaryParams, SolverParams,
    run_cryosuction_simulation, evaluate_module2_status, render_cryosuction_marquee,
)
import state_pipeline


def render() -> None:
    if "cryosuction_results" not in st.session_state:
        st.session_state["cryosuction_results"] = None

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
                sim_out: dict[str, Any] = run_cryosuction_simulation(sfcc_params, bnd_params, solver_params)
            st.session_state["cryosuction_results"] = sim_out

            # Hand off to the shared pipeline for downstream modules (reserved for 4/5).
            state_pipeline.publish_suction_field(
                psi_history=sim_out["psi_history"], theta_history=sim_out["theta_history"],
                t=sim_out["t"], z=sim_out["z"],
            )
            st.success(f"Simulation complete — {len(sim_out['t'])} recorded frames.")

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
                value=len(results_mod2["t"]) - 1, format="frame %d", key="slider_mod2",
            )
            st.caption(
                f"Elapsed time: **{t_days_mod2[frame_idx_mod2]:.2f} days** — "
                f"mock freeze-front depth: **{results_mod2['front_depth_history'][frame_idx_mod2]:.3f} m**"
            )

            psi_now = results_mod2["psi_history"][frame_idx_mod2]
            theta_now = results_mod2["theta_history"][frame_idx_mod2]

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
                psi_matrix = np.array(results_mod2["psi_history"]).T
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

            st.markdown("#### Additional Diagnostics")
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Unfrozen water content θu (current frame, top node)", f"{theta_now[0]:.4f}")
                st.metric("Matric suction ψ (current frame, top node)", f"{psi_now[0]:.2f} kPa")
            with d2:
                st.metric("Moisture influx velocity (current frame)", f"{results_mod2['influx_velocity'][frame_idx_mod2]:.3e} m/s")
                st.metric("Mean suction gradient (current frame)", f"{results_mod2['suction_gradient'][frame_idx_mod2]:.2f} kPa/m")

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

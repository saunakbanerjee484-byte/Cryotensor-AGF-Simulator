# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportMissingTypeArgument=false, reportUnknownParameterType=false
"""
modules/module5.py
=====================
Module 5: Thaw Consolidation Simulator -- Streamlit render() entry point.

Called by registry.py / app.py. All physics lives in
core_physics/thaw_consolidation/; this file is presentation-only. Consumes
Module 1's temperature field from the shared state pipeline
(state_pipeline.py) -- if Module 1 has not been run yet in this session,
falls back to a clearly-labeled synthetic MOCK temperature history so the
module remains fully explorable standalone.

Grid note: Module 1's `T_history` is a list of 2D (ny, nx) spatial snapshots
around a freeze pipe, whereas Module 5 (like Module 2) operates on a 1D
vertical soil column. This file extracts a representative vertical column --
the domain's horizontal midline -- from each of Module 1's recorded frames
and stacks them into the (n_time, n_y) matrix Module 5's physics expects.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from core_physics.thaw_consolidation import (
    ConsolidationParams,
    DynamicConsolidationParams,
    run_thaw_consolidation_simulation,
    evaluate_module5_status,
    render_thaw_consolidation_marquee,
)
import state_pipeline


def _generate_mock_thermal_column(n_time: int = 30, n_y: int = 41, domain_depth_m: float = 3.0,
                                   total_days: float = 15.0):
    """Synthetic post-freeze warming column (Stefan sqrt-time thaw front) for standalone use."""
    y = np.linspace(0.0, domain_depth_m, n_y)
    t = np.linspace(0.0, total_days * 86400.0, n_time)
    front_depth_true = np.clip(0.35 * np.sqrt(t / 86400.0), 0.0, domain_depth_m)
    T_col = np.where(y[None, :] <= front_depth_true[:, None], 3.0, -8.0)
    return t, y, T_col


def _extract_column_from_module1(thermal_bundle: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack Module 1's per-frame 2D fields into a (n_time, ny) column at x = midline."""
    x, y, t = thermal_bundle["x"], thermal_bundle["y"], thermal_bundle["t"]
    col_idx = int(np.argmin(np.abs(x - x.mean())))
    T_col = np.stack([frame[:, col_idx] for frame in thermal_bundle["T_history"]], axis=0)
    return t, y, T_col


def render() -> None:
    if "thaw_consolidation_results" not in st.session_state:
        st.session_state["thaw_consolidation_results"] = None

    st.caption(
        "Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — "
        "Module 5: Thaw Consolidation Simulator"
    )

    col_sim, col_input = st.columns([7, 3], gap="medium")

    # -------------------------------------------------------------------------------------
    # RIGHT COLUMN (30%) -- CONTROL PANEL
    # -------------------------------------------------------------------------------------
    with col_input:
        st.subheader("⚙️ Control Panel")

        thermal_bundle = state_pipeline.get_thermal_field()
        has_real_field = thermal_bundle is not None and len(thermal_bundle["T_history"]) >= 2
        if has_real_field:
            st.success("✅ Live temperature field received from Module 1.")
        else:
            st.warning(
                "⚠️ No Module 1 output found in the shared pipeline yet. Using a "
                "**MOCK** post-freeze thaw column so Module 5 can still be explored "
                "standalone. Run Module 1 first for physically coupled results."
            )

        with st.expander("5A · Consolidation Parameters", expanded=True):
            compression_index_Cc = st.number_input(
                "Compression Index, Cc [-]", min_value=0.01, value=0.30, step=0.01, format="%.3f"
            )
            initial_void_ratio_e0 = st.number_input(
                "Initial (frozen) Void Ratio, e₀ [-]", min_value=0.1, value=0.85, step=0.05, format="%.3f"
            )
            coeff_consolidation_cv0 = st.number_input(
                "Reference Coeff. of Consolidation, cv₀ [m²/s]", min_value=1.0e-9,
                value=1.2e-7, step=1.0e-8, format="%.3e",
            )
            permeability_index_Ck = st.number_input(
                "Permeability Index, Ck [-]", min_value=0.05, value=0.50, step=0.05, format="%.3f",
                help="e-log(k) slope: cv(e) = cv0 * 10^[(e-e0)/Ck]",
            )
            thaw_strain_alpha = st.number_input(
                "Thaw-Strain Coefficient, α [-]", min_value=0.0, max_value=0.5,
                value=0.08, step=0.01, format="%.3f",
            )
            effective_unit_weight = st.number_input(
                "Effective Unit Weight, γ' [N/m³]", min_value=1000.0,
                value=9000.0, step=500.0, format="%.0f",
            )

        with st.expander("5B · Solver & Loading", expanded=True):
            n_nodes = st.number_input("Depth nodes [-]", min_value=11, max_value=161, value=41, step=2)
            domain_depth_m = st.number_input("Domain depth [m]", min_value=0.5, value=3.0, step=0.5, format="%.2f")
            dt_hours = st.number_input("Timestep, Δt [hours]", min_value=0.5, value=4.0, step=0.5, format="%.2f")
            total_days = st.number_input("Total simulated duration [days]", min_value=1.0, value=15.0, step=1.0, format="%.1f")
            record_every = st.number_input("Record every N steps [-]", min_value=1, max_value=200, value=4, step=1)

            st.markdown("**Dynamic Surcharge Loading, q(t)**")
            enable_surcharge = st.checkbox("Enable surcharge (embankment/equipment load)", value=False)
            surcharge_mag_kpa = st.number_input(
                "Surcharge magnitude [kPa]", min_value=0.0, value=50.0, step=5.0, format="%.1f",
                disabled=not enable_surcharge,
            )
            surcharge_start_pct = st.number_input(
                "Applied at % of duration [-]", min_value=0, max_value=99, value=50, step=5,
                disabled=not enable_surcharge,
            )

        run_clicked = st.button("🌊 RUN THAW CONSOLIDATION SOLVER", type="primary", width="stretch")

        if run_clicked:
            p = ConsolidationParams(
                compression_index_Cc=compression_index_Cc,
                initial_void_ratio_e0=initial_void_ratio_e0,
                coeff_consolidation_cv0=coeff_consolidation_cv0,
                permeability_index_Ck=permeability_index_Ck,
                thaw_strain_alpha=thaw_strain_alpha,
                effective_unit_weight_n_per_m3=effective_unit_weight,
            )

            n_steps = max(1, int(round(total_days * 86400.0 / (dt_hours * 3600.0))))
            surcharge_history = None
            if enable_surcharge:
                surcharge_history = np.zeros(n_steps + 1)
                start_step = int((surcharge_start_pct / 100.0) * n_steps)
                surcharge_history[start_step:] = surcharge_mag_kpa * 1000.0

            dyn = DynamicConsolidationParams(
                n_nodes=int(n_nodes), domain_depth_m=domain_depth_m, dt_hours=dt_hours,
                total_days=total_days, record_every=int(record_every),
                surcharge_history_pa=surcharge_history,
            )

            if has_real_field and thermal_bundle is not None:
                t_m1, y_m1, T_col = _extract_column_from_module1(thermal_bundle)
                data_source = "Module 1 (live)"
            else:
                t_m1, y_m1, T_col = _generate_mock_thermal_column(domain_depth_m=domain_depth_m, total_days=total_days)
                data_source = "MOCK (Module 1 not yet run)"

            with st.spinner("Solving Backward-Euler thaw consolidation (sparse, moving boundary)..."):
                sim_out: dict[str, Any] = run_thaw_consolidation_simulation(T_col, t_m1, y_m1, p, dyn)
            sim_out["data_source"] = data_source
            st.session_state["thaw_consolidation_results"] = sim_out
            st.success(
                f"Solve complete — final settlement {sim_out['final_settlement_mm']:.2f} mm, "
                f"peak rate {sim_out['max_settlement_rate_mm_day']:.2f} mm/day."
            )

    # -------------------------------------------------------------------------------------
    # LEFT COLUMN (70%) -- LIVE SIMULATION
    # -------------------------------------------------------------------------------------
    with col_sim:
        st.markdown(render_thaw_consolidation_marquee(), unsafe_allow_html=True)

        results: dict[str, Any] | None = st.session_state.get("thaw_consolidation_results")

        if results is None:
            st.info(
                "⬅ Configure Sub-Modules 5A (Consolidation Parameters) and 5B (Solver & Loading) "
                "in the Control Panel, then click **RUN THAW CONSOLIDATION SOLVER** to populate "
                "the Live Simulation panel."
            )
            return

        z = results["z"]
        t_days = results["t"] / 86400.0
        st.caption(f"Temperature data source: **{results['data_source']}**")

        # ---- Dual visualization: settlement-vs-time line | void-ratio/pressure heatmap ----
        col_line, col_heat = st.columns(2)

        with col_line:
            fig_line, (ax_s, ax_front) = plt.subplots(2, 1, figsize=(6.0, 6.4))
            ax_s.plot(t_days, results["settlement_m"] * 1000.0, color="#c0630a", linewidth=1.8)
            ax_s.set_title("Surface Settlement vs. Time", fontsize=10, fontweight="bold")
            ax_s.set_xlabel("Time [days]", fontsize=9)
            ax_s.set_ylabel("Settlement, S(t) [mm]", fontsize=9)
            ax_s.grid(alpha=0.3)

            ax_front.plot(t_days, results["front_depth_history"], color="#0a4d8c", linewidth=1.8)
            ax_front.set_title("Thaw Front Depth vs. Time", fontsize=10, fontweight="bold")
            ax_front.set_xlabel("Time [days]", fontsize=9)
            ax_front.set_ylabel("Front depth, Xf [m]", fontsize=9)
            ax_front.invert_yaxis()
            ax_front.grid(alpha=0.3)

            fig_line.tight_layout()
            st.pyplot(fig_line, clear_figure=True)
            fig_line.clf()
            plt.close(fig_line)

        with col_heat:
            field_choice = st.radio(
                "Heatmap field", ["Excess Pore Pressure, u [kPa]", "Void Ratio, e [-]"],
                horizontal=True, key="mod5_heatmap_toggle",
            )
            fig_heat, ax_heat = plt.subplots(figsize=(6.2, 6.4))
            if field_choice.startswith("Excess"):
                data = results["u_history"].T / 1000.0
                sns.heatmap(data, cmap="YlOrRd", cbar_kws={"label": "u [kPa]"},
                            ax=ax_heat, xticklabels=False, yticklabels=False)
                ax_heat.set_title("Excess Pore Pressure Field, u(z,t)", fontsize=10, fontweight="bold")
            else:
                data = results["e_history"].T
                sns.heatmap(data, cmap="BrBG_r", cbar_kws={"label": "e [-]"},
                            ax=ax_heat, xticklabels=False, yticklabels=False)
                ax_heat.set_title("Void Ratio Field, e(z,t)", fontsize=10, fontweight="bold")
            ax_heat.set_xlabel("Time →", fontsize=9)
            ax_heat.set_ylabel("Depth (top=surface) →", fontsize=9)
            fig_heat.tight_layout()
            st.pyplot(fig_heat, clear_figure=True)
            fig_heat.clf()
            plt.close(fig_heat)

        # ---- Raw data grid ----
        st.markdown("#### 📐 Raw Nodal Grid — Void Ratio & Excess Pore Pressure")
        df = pd.DataFrame(
            np.round(results["u_history"] / 1000.0, 4),
            index=[f"t={td:.2f}d" for td in t_days],
            columns=np.round(z, 3),
        )
        df.index.name = "Time"
        st.dataframe(df, width="stretch", height=280)
        csv_bytes = df.to_csv().encode("utf-8")
        st.download_button(
            "Download excess pore pressure tensor (CSV)", data=csv_bytes,
            file_name="thaw_consolidation_u_matrix.csv", mime="text/csv",
        )

        # ---- Diagnostics ----
        st.markdown("#### Additional Diagnostics")
        d1, d2 = st.columns(2)
        with d1:
            st.metric("Final settlement", f"{results['final_settlement_mm']:.2f} mm")
            st.metric("Final thaw-front depth", f"{results['front_depth_history'][-1]:.3f} m")
        with d2:
            st.metric("Peak settlement rate", f"{results['max_settlement_rate_mm_day']:.2f} mm/day")
            st.metric("Min void ratio reached", f"{results['e_history'].min():.3f}")

        # ---- Deterministic engineering assessment ----
        st.markdown("#### 🚦 Deterministic Engineering Assessment")
        status, message = evaluate_module5_status(results["max_settlement_rate_mm_day"])
        if status == "red":
            st.error(message)
        elif status == "yellow":
            st.warning(message)
        else:
            st.success(message)
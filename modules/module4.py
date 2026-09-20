"""
modules/module4.py
=====================
Module 4C: Thermo-Elastic Restrained Stress & THMC Degradation -- Streamlit
render() entry point (the Advanced Control Panel UI).

Called by registry.py / app.py. All physics (4B.1/4B.2/4B.3) lives in
core_physics/restrained_stress.py; this file is presentation-only.

4C.1  Physics toggles     -- checkboxes wired straight to StressParams flags
4C.2  Pipeline diagnostics -- live 🟢/🔴 sensors for Modules 1/2/3's handoffs
4C.3  Execution trigger    -- gated: requires all sensors green, OR Mock Data Mode
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from core_physics.restrained_stress import (
    StressParams,
    run_restrained_stress_simulation,
    generate_mock_strain_field,
    generate_mock_temperature_field,
    evaluate_module4_status,
)
import state_pipeline


def render() -> None:
    if "restrained_stress_results" not in st.session_state:
        st.session_state["restrained_stress_results"] = None

    st.caption(
        "Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — "
        "Module 4: Thermo-Elastic Restrained Stress & THMC Degradation"
    )

    col_sim, col_input = st.columns([7, 3], gap="medium")

    # -------------------------------------------------------------------------------------
    # RIGHT COLUMN (30%) -- 4C. ADVANCED CONTROL PANEL
    # -------------------------------------------------------------------------------------
    with col_input:
        st.subheader("⚙️ Control Panel")

        with st.expander("4A · Thermo-Elastic & THMC Parameters", expanded=True):
            youngs_modulus_frozen = st.number_input(
                "Young's Modulus, E (frozen soil) [Pa]", min_value=1.0e5,
                value=45.0e6, step=1.0e6, format="%.3e",
            )
            poisson_ratio_frozen = st.number_input(
                "Poisson's Ratio, ν [-]", min_value=0.01, max_value=0.49,
                value=0.35, step=0.01, format="%.3f",
            )
            concrete_yield_strength_mpa = st.number_input(
                "Concrete Yield Strength [MPa]", min_value=0.01,
                value=25.0, step=0.5, format="%.2f",
            )
            creep_n = st.number_input(
                "Creep Power-Law Exponent, n [-]", min_value=1.0, max_value=10.0,
                value=3.0, step=0.5, format="%.2f",
            )
            seepage_vx = st.number_input(
                "Seepage Velocity, v_x [m/s]", min_value=0.0,
                value=1.0e-7, step=1.0e-8, format="%.2e",
            )
            initial_salinity_ppt = st.number_input(
                "Initial Salinity [ppt]", min_value=0.0,
                value=5.0, step=0.5, format="%.2f",
            )

        # ---- 4C.1 THMC Physics Toggles ---------------------------------------------
        with st.expander("4C.1 · THMC Physics Toggles", expanded=True):
            st.caption("Toggle individual physics sub-routines to compare pure elasticity vs. full THMC.")
            enable_creep = st.checkbox("Enable Visco-Plastic Creep", value=True)
            enable_advective_seepage = st.checkbox("Enable Advective Seepage (TH Coupling)", value=True)
            enable_solute_rejection = st.checkbox("Enable Solute Rejection / Salinity", value=True)
            enable_damage = st.checkbox("Enable Continuum Damage Mechanics", value=True)

        # ---- 4C.2 Pipeline Handoff Diagnostics --------------------------------------
        with st.expander("4C.2 · Pipeline Handoff Diagnostics", expanded=True):
            thermal_bundle = state_pipeline.get_thermal_field()
            suction_bundle = state_pipeline.get_suction_field()
            strain_bundle = state_pipeline.get_strain_field()

            thermal_ok = thermal_bundle is not None
            suction_ok = suction_bundle is not None
            strain_ok = strain_bundle is not None

            st.markdown(
                f"🌡️ Thermal Field (Mod 1): {'🟢 Fetched' if thermal_ok else '🔴 Missing'}  \n"
                f"💧 Cryosuction Field (Mod 2): {'🟢 Fetched' if suction_ok else '🔴 Missing'}  \n"
                f"🧊 Heave Tensor (Mod 3): {'🟢 Fetched' if strain_ok else '🔴 Missing'}"
            )

            mock_mode = st.checkbox(
                "🧪 Enable Mock Data Mode (override missing sensors)", value=not strain_ok,
                help="Lets you run the engine standalone with synthetic fields when upstream "
                     "modules haven't been executed yet in this session.",
            )

        st.markdown("---")
        st.markdown("### 4C.3 · Execution Trigger")

        all_sensors_green = thermal_ok and suction_ok and strain_ok
        can_run = all_sensors_green or mock_mode
        if not can_run:
            st.warning(
                "Run all of Modules 1, 2 and 3 first, or enable **Mock Data Mode** above, "
                "to activate the engine."
            )

        run_clicked = st.button(
            "Run Restrained Stress Engine", type="primary", width="stretch",
            disabled=not can_run,
        )

        if run_clicked:
            p = StressParams(
                youngs_modulus_frozen=youngs_modulus_frozen,
                poisson_ratio_frozen=poisson_ratio_frozen,
                concrete_yield_strength_mpa=concrete_yield_strength_mpa,
                creep_n=creep_n,
                seepage_vx=seepage_vx,
                initial_salinity_ppt=initial_salinity_ppt,
                enable_creep=enable_creep,
                enable_advective_seepage=enable_advective_seepage,
                enable_solute_rejection=enable_solute_rejection,
                enable_damage=enable_damage,
            )

            if strain_ok:
                eps_v_field = strain_bundle["strain_field"]
                x, y = strain_bundle["x"], strain_bundle["y"]
                strain_source = "Module 3 (live)"
            else:
                x, y, eps_v_field = generate_mock_strain_field()
                strain_source = "MOCK (Module 3 not yet run)"

            T_matrix = None
            thermal_source = "unavailable"
            if thermal_ok:
                T_curr = thermal_bundle["T_history"][-1]
                if T_curr.shape == eps_v_field.shape:
                    T_matrix = T_curr
                    thermal_source = "Module 1 (live)"
            if T_matrix is None and mock_mode:
                T_matrix = generate_mock_temperature_field(x, y)
                thermal_source = "MOCK (Module 1 not yet run, or grid mismatch)"

            suction_context = suction_bundle if suction_ok else None

            with st.spinner("Solving THMC-coupled restrained stress engine (4B.1 → 4B.2 → 4B.3)..."):
                sim_out: dict[str, Any] = run_restrained_stress_simulation(
                    eps_v_field, x, y, p, T_matrix=T_matrix, suction_context=suction_context,
                )
            sim_out["strain_source"] = strain_source
            sim_out["thermal_source"] = thermal_source
            sim_out["suction_source"] = "Module 2 (live)" if suction_ok else "unavailable"
            st.session_state["restrained_stress_results"] = sim_out
            st.success(
                f"THMC solve complete — peak lateral pressure "
                f"{sim_out['max_lateral_pressure_mpa']:.4f} MPa, peak damage {sim_out['max_damage']:.3f}."
            )

    # -------------------------------------------------------------------------------------
    # LEFT COLUMN (70%) -- LIVE SIMULATION
    # -------------------------------------------------------------------------------------
    with col_sim:
        from core_physics.restrained_stress import render_restrained_stress_marquee
        st.markdown(render_restrained_stress_marquee(), unsafe_allow_html=True)

        results: dict[str, Any] | None = st.session_state.get("restrained_stress_results")

        if results is None:
            st.info(
                "⬅ Configure Sub-Module 4A (THMC Parameters), toggle the desired physics in "
                "4C.1, check the pipeline sensors in 4C.2, then click **Run Restrained Stress "
                "Engine** to populate the Live Simulation panel."
            )
            return

        p: StressParams = results["params"]
        x, y = results["x"], results["y"]
        st.caption(
            f"Data sources — strain: **{results['strain_source']}** · "
            f"thermal: **{results['thermal_source']}** · suction: **{results['suction_source']}**"
        )

        # ---- Dual visualization: Matplotlib pressure-vs-depth | Seaborn togglable heatmap ----
        col_line, col_heat = st.columns(2)

        with col_line:
            fig_line, ax = plt.subplots(figsize=(6.0, 6.4))
            ax.plot(results["wall_face_pressure_mpa"], y, color="#0a4d8c", linewidth=1.8, marker="o", markersize=2.5)
            ax.axvline(p.concrete_yield_strength_mpa, color="#c0392b", linestyle="--", linewidth=1.3,
                       label=f"Allowable ({p.concrete_yield_strength_mpa:.2f} MPa)")
            ax.set_title("Lateral Earth Pressure vs. Depth — Wall Interface", fontsize=10, fontweight="bold")
            ax.set_xlabel("Lateral pressure, |σxx| [MPa]", fontsize=9)
            ax.set_ylabel("Depth, y [m]", fontsize=9)
            ax.invert_yaxis()
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            fig_line.tight_layout()
            st.pyplot(fig_line, clear_figure=True)
            fig_line.clf()
            plt.close(fig_line)

        with col_heat:
            field_choice = st.radio(
                "Heatmap field", ["Principal Stress σ₁ [MPa]", "Damage Scalar D [-]"],
                horizontal=True, key="mod4_heatmap_toggle",
            )
            fig_heat, ax_heat = plt.subplots(figsize=(6.2, 6.4))
            if field_choice.startswith("Principal"):
                sns.heatmap(
                    results["sigma_1_mpa"], cmap="Spectral_r", cbar_kws={"label": "σ₁ [MPa]"},
                    ax=ax_heat, xticklabels=False, yticklabels=False,
                )
                ax_heat.set_title("Principal Stress Field, σ₁", fontsize=10, fontweight="bold")
            else:
                sns.heatmap(
                    results["D_field"], cmap="Reds", vmin=0.0, vmax=1.0, cbar_kws={"label": "Damage, D [-]"},
                    ax=ax_heat, xticklabels=False, yticklabels=False,
                )
                ax_heat.set_title("Continuum Damage Scalar Field, D", fontsize=10, fontweight="bold")
            ax_heat.set_xlabel("x →", fontsize=9)
            ax_heat.set_ylabel("y (depth) →", fontsize=9)
            fig_heat.tight_layout()
            st.pyplot(fig_heat, clear_figure=True)
            fig_heat.clf()
            plt.close(fig_heat)

        # ---- Raw data grid: nodal stress tensor + damage ----
        st.markdown("#### 📐 Raw Nodal Grid — Stress Tensor & Damage")
        df = pd.DataFrame({
            "x [m]": np.meshgrid(x, y)[0].ravel(),
            "y [m]": np.meshgrid(x, y)[1].ravel(),
            "sigma_xx [MPa]": results["sigma_xx_mpa"].ravel(),
            "sigma_yy [MPa]": results["sigma_yy_mpa"].ravel(),
            "tau_xy [MPa]": results["tau_xy_mpa"].ravel(),
            "sigma_1 [MPa]": results["sigma_1_mpa"].ravel(),
            "D [-]": results["D_field"].ravel(),
            "eps_vp [-]": results["eps_vp_field"].ravel(),
        }).round(6)
        st.dataframe(df, width="stretch", height=280)
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download nodal stress/damage tensor (CSV)", data=csv_bytes,
            file_name="restrained_stress_tensor.csv", mime="text/csv",
        )

        # ---- Diagnostics ----
        st.markdown("#### Additional Diagnostics")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.metric("Peak lateral pressure", f"{results['max_lateral_pressure_mpa']:.4f} MPa")
            st.metric("Peak damage, D_max", f"{results['max_damage']:.3f}")
        with d2:
            st.metric("β_max (restraint ratio)", f"{results['beta_field'].max():.3f}")
            st.metric("Creep-active fraction", f"{results['creep_active_fraction']*100:.2f} %")
        with d3:
            st.metric("Seepage erosion index", f"{results['seepage_erosion_index']:.3f}")
            if results["advective"] is not None:
                st.metric("Depressed T_f (min)", f"{results['advective']['T_f_depressed_field_c'].min():.2f} °C")
            else:
                st.metric("Depressed T_f (min)", "n/a (no T field)")

        # ---- Deterministic engineering assessment ----
        st.markdown("#### 🚦 Deterministic Engineering Assessment")
        status, message = evaluate_module4_status(
            results["max_lateral_pressure_mpa"], p.concrete_yield_strength_mpa,
            results["max_damage"], results["creep_active_fraction"], results["seepage_erosion_index"],
        )
        if status == "red":
            st.error(message)
        elif status == "yellow":
            st.warning(message)
        else:
            st.success(message)

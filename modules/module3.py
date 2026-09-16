"""
modules/module3.py
=====================
Module 3: Volumetric Frost Heave Tensor -- Streamlit render() entry point.

Called by registry.py / app.py. All physics lives in core_physics/frost_heave.py;
this file is presentation-only. Consumes Module 1's temperature field from the
shared state pipeline (state_pipeline.py) -- if Module 1 has not been run yet in
this session, falls back to a clearly-labeled synthetic MOCK temperature field
so the module remains fully explorable standalone.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from core_physics.frost_heave import (
    HeaveParams,
    run_frost_heave_simulation,
    generate_mock_temperature_fields,
    evaluate_module3_status,
    render_frost_heave_marquee,
)
import state_pipeline


def render() -> None:
    if "frost_heave_results" not in st.session_state:
        st.session_state["frost_heave_results"] = None

    st.caption(
        "Artificial Ground Freezing & Thermo-Hydro-Mechanical Engine — "
        "Module 3: Volumetric Frost Heave Tensor"
    )

    col_sim, col_input = st.columns([7, 3], gap="medium")

    # -------------------------------------------------------------------------------------
    # RIGHT COLUMN (30%) -- CONTROL PANEL
    # -------------------------------------------------------------------------------------
    with col_input:
        st.subheader("⚙️ Control Panel")

        thermal = state_pipeline.get_thermal_field()
        has_real_field = thermal is not None and len(thermal["T_history"]) >= 2
        if has_real_field:
            st.success("✅ Live temperature field received from Module 1.")
        else:
            st.warning(
                "⚠️ No Module 1 output found in the shared pipeline yet. Using a "
                "**MOCK** radial-cooling temperature field so Module 3 can still be "
                "explored standalone. Run Module 1 first for physically coupled results."
            )

        with st.expander("3A · Frost Heave Parameters", expanded=True):
            porosity = st.number_input("Initial porosity, n [-]", min_value=0.05, max_value=0.95, value=0.42, step=0.01, format="%.3f")
            sp0 = st.number_input(
                "Base Segregation Potential, SP₀ [mm²/(°C·day)]", min_value=0.0, value=2.0, step=0.1, format="%.2f",
                help="Konrad classification guide: <0.3 negligible, 1-3 moderate, 3-6 high, >6 very high frost susceptibility.",
            )
            overburden_pressure = st.number_input("Overburden pressure, P [kPa]", min_value=0.0, value=50.0, step=5.0, format="%.2f")
            pressure_decay_a = st.number_input("Pressure decay coefficient, a [1/kPa]", min_value=0.0001, value=0.0300, step=0.0010, format="%.4f")

        with st.expander("3B · Freezing Fringe & Structural Tolerance", expanded=True):
            fringe_upper = st.number_input("Freezing fringe upper bound [°C]", value=0.0, step=0.1, format="%.2f")
            fringe_lower = st.number_input("Freezing fringe lower bound [°C]", value=-0.5, step=0.1, format="%.2f")
            structural_tolerance = st.number_input(
                "Allowable structural heave tolerance [mm/day]", min_value=0.01, value=2.0, step=0.1, format="%.2f",
                help="Triggers the RED alert if the peak local heave rate meets or exceeds this value.",
            )

        st.markdown("---")
        st.markdown("### 3C · Volumetric Heave Tensor Engine")
        run_clicked = st.button("🧊 COMPUTE FROST HEAVE TENSOR", type="primary", width="stretch")

        if run_clicked:
            p = HeaveParams(
                porosity=porosity, sp0_mm2_per_c_day=sp0,
                overburden_pressure_kpa=overburden_pressure, pressure_decay_coeff_per_kpa=pressure_decay_a,
                fringe_t_upper_c=fringe_upper, fringe_t_lower_c=fringe_lower,
                structural_tolerance_mm_per_day=structural_tolerance,
            )

            if has_real_field:
                T_curr = thermal["T_history"][-1]
                T_prev = thermal["T_history"][-2]
                dt_s = float(thermal["t"][-1] - thermal["t"][-2])
                x, y = thermal["x"], thermal["y"]
                data_source = "Module 1 (live)"
            else:
                x, y, T_prev, T_curr, dt_s = generate_mock_temperature_fields()
                data_source = "MOCK (Module 1 not yet run)"

            with st.spinner("Computing vectorized volumetric frost heave tensor..."):
                sim_out: dict[str, Any] = run_frost_heave_simulation(T_curr, T_prev, dt_s, x, y, p)
            sim_out["data_source"] = data_source
            st.session_state["frost_heave_results"] = sim_out
            st.success(f"Heave tensor computed — peak local rate {sim_out['max_heave_rate_mm_day']:.3f} mm/day.")

    # -------------------------------------------------------------------------------------
    # LEFT COLUMN (70%) -- LIVE SIMULATION / VISUALS
    # -------------------------------------------------------------------------------------
    with col_sim:
        st.markdown(render_frost_heave_marquee(), unsafe_allow_html=True)

        results: dict[str, Any] | None = st.session_state.get("frost_heave_results")

        if results is None:
            st.info(
                "⬅ Configure Sub-Modules 3A (Frost Heave Parameters) and 3B (Freezing Fringe & "
                "Structural Tolerance) in the Control Panel, then click **COMPUTE FROST HEAVE "
                "TENSOR** to populate the Live Simulation panel."
            )
        else:
            p = results["params"]
            x, y = results["x"], results["y"]
            st.caption(f"Temperature field source: **{results['data_source']}**")

            col_line, col_heat = st.columns(2)

            with col_line:
                fig_line, ax = plt.subplots(figsize=(6.2, 6.4))
                ax.plot(x, results["uplift_rate_profile_mm_day"], color="#a8501f", linewidth=1.8)
                ax.axhline(p.structural_tolerance_mm_per_day, color="#c0392b", linestyle="--", linewidth=1.2,
                           label=f"Structural tolerance ({p.structural_tolerance_mm_per_day:.2f} mm/day)")
                ax.set_title("1D Ground-Surface Uplift Rate Profile", fontsize=10, fontweight="bold")
                ax.set_xlabel("Horizontal position, x [m]", fontsize=9)
                ax.set_ylabel("Depth-integrated uplift rate [mm/day]", fontsize=9)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                fig_line.tight_layout()
                st.pyplot(fig_line, clear_figure=True)
                fig_line.clf()
                plt.close(fig_line)

            with col_heat:
                fig_heat, ax_heat = plt.subplots(figsize=(6.2, 6.4))
                sns.heatmap(
                    results["strain_field"], cmap="Spectral_r", center=0.0,
                    cbar_kws={"label": "Volumetric strain increment, Δε_v [-]"},
                    ax=ax_heat, xticklabels=False, yticklabels=False,
                )
                ax_heat.set_title("Volumetric Strain Tensor Field, Δε_v(x,y)", fontsize=10, fontweight="bold")
                ax_heat.invert_yaxis()
                fig_heat.tight_layout()
                st.pyplot(fig_heat, clear_figure=True)
                fig_heat.clf()
                plt.close(fig_heat)

            st.markdown("#### 📐 Raw Nodal Grid — Localized Heave Rate ḣ(x,y) [mm/day]")
            df_heave = pd.DataFrame(
                np.round(results["h_dot_field_mm_day"], 4),
                index=np.round(y, 3),
                columns=np.round(x, 3),
            )
            df_heave.index.name = "y [m]"
            df_heave.columns.name = "x [m]"
            st.dataframe(df_heave, width="stretch", height=280)
            csv_bytes = df_heave.to_csv().encode("utf-8")
            st.download_button(
                "Download full heave-rate tensor (CSV)", data=csv_bytes,
                file_name="frost_heave_rate_matrix.csv", mime="text/csv",
            )

            st.markdown("#### Additional Diagnostics")
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Peak local heave rate", f"{results['max_heave_rate_mm_day']:.3f} mm/day")
                st.metric("Mean heave rate within fringe", f"{results['mean_fringe_heave_rate_mm_day']:.3f} mm/day")
            with d2:
                st.metric("Active fringe cell count", f"{int(results['fringe_mask'].sum())}")
                st.metric("Effective segregation potential, SP", f"{results['SP_effective']:.3e} m²/(°C·s)")

            st.markdown("#### 🚦 Deterministic Engineering Assessment")
            fringe_cell_count = int(results["fringe_mask"].sum())
            if fringe_cell_count == 0:
                st.warning(
                    "⚠️ **UNRESOLVED FREEZING FRINGE**: zero grid nodes fell within the "
                    f"[{p.fringe_t_lower_c:.2f}°C, {p.fringe_t_upper_c:.2f}°C] fringe band on "
                    "this temperature field. With a coarse grid and/or a steep radial gradient "
                    "near the freeze pipe, the fringe can fall entirely between nodes -- the "
                    "0.000 mm/day result below is a discretization artifact, not a physical "
                    "finding. Refine Module 1's grid (more nx/ny), widen the fringe band in 3B, "
                    "or let the front advance further (longer duration) and re-run."
                )
            else:
                status, message = evaluate_module3_status(
                    results["max_heave_rate_mm_day"], p.structural_tolerance_mm_per_day
                )
                if status == "red":
                    st.error(message)
                elif status == "yellow":
                    st.warning(message)
                else:
                    st.success(message)

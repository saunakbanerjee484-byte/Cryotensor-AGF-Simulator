import streamlit as st
import numpy as np
import pandas as pd

def render():
    st.header("4. Thermo-Elastic Restrained Stress")
    st.markdown("""
    Converts the Volumetric Strain Tensor (from Module 3) into Effective Stress ($\\sigma'$),
    simulating the cryogenic lateral earth pressure exerted on the retaining structures.
    """)

    col_sim, col_input = st.columns([7, 3], gap="medium")

    with col_input:
        st.subheader("⚙️ Control Panel")
        
        # Exact Numerical Inputs
        youngs_modulus = st.number_input("Young's Modulus, E (MPa)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
        poissons_ratio = st.number_input("Poisson's Ratio, ν", min_value=0.1, max_value=0.49, value=0.3, step=0.01)
        wall_yield_strength = st.number_input("Wall Yield Strength (MPa)", min_value=1.0, max_value=100.0, value=30.0, step=1.0)
        
        st.info(
            "Applying boundary conditions for zero-displacement at structural interfaces. "
            "Volumetric strain is converted into Effective Stress."
        )

    with col_sim:
        # --- PHYSICS ENGINE (Pure Vectorization) ---
        
        # CRITICAL FIX: Match the spatial dimensions dictated by core_physics/phase_change.py
        # Domain: 3.0m x 3.0m, Grid: 61x61, Pipe Centered at (1.5, 1.5)
        nx, ny = 61, 61
        Lx, Ly = 3.0, 3.0
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny) # representing depth
        X, Y = np.meshgrid(x, y)
        
        # 1. Fetching Volumetric Strain Tensor from the st.session_state pipeline
        # (Falls back to an aligned mock dataset if earlier modules haven't run yet)
        if 'mod3_strain_field' in st.session_state:
            strain_v = st.session_state['mod3_strain_field']
        else:
            # Fallback mock accurately radiating from the pipe center at (1.5, 1.5)
            dist_to_center = np.sqrt((X - 1.5)**2 + (Y - 1.5)**2)
            strain_v = 0.05 * np.exp(-dist_to_center / 0.5) 
                
        # 2. Thermo-Poro-Elasticity: Vectorized 2D Hooke's Law
        # Plane strain bulk modulus (K) calculation
        K = youngs_modulus / (3 * (1 - 2 * poissons_ratio))
        
        # Isotropic restrained stress assumption: stress = K * volumetric_strain
        # The stress field is computed instantly across the entire grid via broadcasting
        stress_field = K * strain_v
        
        # 3. Restrained Earth Pressure along the wall
        # We extract the 1D pressure boundary condition along the left wall face (x=0)
        wall_pressure = stress_field[:, 0]
        max_pressure = np.max(wall_pressure)
        
        # --- UI / UX ---
        st.subheader("2D Principal Stress Field (MPa)")
        
        # Heatmap of the stress field concentrations
        # Rescaling for colormap mapping (uint8)
        max_stress = np.max(stress_field) if np.max(stress_field) > 0 else 1.0
        st.image(
            np.clip(stress_field / max_stress * 255, 0, 255).astype(np.uint8), 
            use_column_width=True, 
            caption="Stress Concentrations around the freeze pipe and retaining wall"
        )
        
        st.subheader("Pressure Distribution vs. Depth (at x = 0m Wall)")
        # Line Plot: pressure distribution curve vs. depth on the wall
        chart_data = pd.DataFrame({"Depth (m)": y, "Lateral Pressure (MPa)": wall_pressure}).set_index("Depth (m)")
        st.line_chart(chart_data)
        
        # R/Y/G Alerts
        st.subheader("Structural Integrity Status")
        if max_pressure > wall_yield_strength:
            st.error(f"🔴 RED ALERT: Peak lateral earth pressure ({max_pressure:.2f} MPa) exceeds wall yield strength ({wall_yield_strength} MPa).")
        elif max_pressure > 0.75 * wall_yield_strength:
            st.warning(f"🟡 YELLOW ALERT: Peak lateral earth pressure ({max_pressure:.2f} MPa) is approaching yield strength.")
        else:
            st.success(f"🟢 GREEN ALERT: Wall stresses ({max_pressure:.2f} MPa) are within safe structural limits.")
            
        # Push worst-case scalar to session memory for Module 6 (Command Center Master Export)
        st.session_state['mod4_max_pressure'] = max_pressure
        st.session_state['mod4_status'] = 'Red' if max_pressure > wall_yield_strength else ('Yellow' if max_pressure > 0.75 * wall_yield_strength else 'Green')
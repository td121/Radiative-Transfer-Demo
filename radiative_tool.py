import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

# --- Configuration & Styling ---
st.set_page_config(page_title="AHL Ch.2: Radiative Transfer Demo", layout="wide")
st.title("Chapter 2: Radiative Processes & Remote Sounding")
st.markdown("*Interactive demo for Andrews, Holton, & Leovy*")


# --- Tabs for different concepts ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["1. Beer-Lambert & Scattering", "2. The Schwarzchild Equation", "3. Weighting Functions (Sounding)","4. LBL Integration (Matrix Method)","5. Angular Integration (Diffusivity Factor)","6. Line Shapes (Lorentz/Doppler/Voigt)", "The Curve of Growth"])

# ==========================================
# TAB 1: EXTINCTION & SCATTERING
# ==========================================
with tab1:
    st.header("Extinction: Absorption vs. Scattering")
    st.markdown(r"Visualizing the depletion of a beam: $dI = -k \rho I ds$")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        I0 = st.slider("Incident Intensity ($I_0$)", 0.0, 100.0, 100.0)
        k_abs = st.slider("Absorption Coefficient ($k_a$)", 0.0, 1.0, 0.1)
        k_scat = st.slider("Scattering Coefficient ($k_s$)", 0.0, 1.0, 0.1)
        density = st.slider("Medium Density ($\rho$)", 0.1, 5.0, 1.0)
        
        st.info("**Key Concept:** Extinction is the sum of absorption and scattering.")

    with col2:
        z = np.linspace(0, 10, 100)
        k_ext = k_abs + k_scat # equation 2.2.6
        tau = k_ext * density * z  # Optical depth
        I_z = I0 * np.exp(-tau) #Remaining Intensity
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(z, I_z, label="Remaining Intensity", color='blue', linewidth=3)
        ax.fill_between(z, I_z, color='blue', alpha=0.1)
        
        # Plot the "lost" energy
        ax.plot(z, I0 - I_z, label="Extinguished (Absorbed+Scattered)", color='red', linestyle='--')
        
        ax.set_xlabel("Distance through medium (s)")
        ax.set_ylabel("Intensity (Radiance)")
        ax.set_title(f"Beer-Lambert Law (Optical Depth $\\tau$ max = {tau[-1]:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    st.divider()
    
    colx, coly = st.columns([1, 2])
    with colx:
        # --- Scattering Phase Function Demo ---
        st.subheader("Scattering Phase Functions")
        st.markdown("Where does the scattered photon go? (Rayleigh vs. Mie)")
        
        scat_type = st.radio("Select Scattering Regime:", ["Rayleigh (Molecules)", "Mie (Aerosols/Clouds)"])
        
        theta = np.linspace(0, 2*np.pi, 360)
        
        if scat_type == "Rayleigh (Molecules)":
            # Rayleigh phase function proportional to 1 + cos^2(theta)
            P = 0.75 * (1 + np.cos(theta)**2)
            title_text = "Rayleigh (Isotropic-ish)"
        else:
            # Henyey-Greenstein approximation for Mie
            g = st.slider("Asymmetry Factor (g)", 0.0, 0.99, 0.7)
            P = (1 - g**2) / ((1 + g**2 - 2*g*np.cos(theta))**(1.5))
            title_text = f"Mie (Forward Scattering, g={g})"
    with coly:
        fig2, ax2 = plt.subplots(figsize=(4, 4), subplot_kw={'projection': 'polar'})
        ax2.plot(theta, P, color='green')
        ax2.set_title(title_text)
        ax2.set_rticks([])  # Less clutter
        ax2.grid(True)
        
        st.pyplot(fig2)

# ==========================================
# TAB 2: RADIATIVE TRANSFER EQUATION
# ==========================================
# ==========================================
# TAB 2: RADIATIVE TRANSFER & BOLTZMANN (UPDATED)
# ==========================================
with tab2:
    st.header("The Schwarzchild Equation & Boltzmann Effect")
    st.markdown(r"$$dI = -k\rho I ds + k\rho B(T) ds$$")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Atmospheric State")
        T_surface = st.slider("Surface Temperature (K)", 200, 1000, 300)
        T_atmos = st.slider("Atmosphere Temperature (K)", 100, 1000, 250)
        
        st.divider()
        st.subheader("Microphysics (Boltzmann)")
        st.markdown("Does the gas absorb more when hot or cold?")
        
        # New Control: Energy State of the Transition
        energy_state = st.radio(
            "Transition Type (Lower State Energy E):",
            ["Ground State (Low E)", "High-J / Hot Band (High E)"],
            help="Ground State lines get weaker as T rises (population scatters). Hot Bands get stronger as T rises (population fills)."
        )
        
        gas_amount = st.slider("Gas Amount (u)", 0.1, 5.0, 1.0)

    with col2:
        # --- PHYSICS ENGINE ---
        
        # 1. Boltzmann Population Factor (Simplified)
        # Fraction of molecules in the lower state required for this transition
        # N_i ~ exp(-E/kT) / Partition_Function(T)
        
        k_b = 0.695 # wavenumber/K approx
        
        if energy_state == "Ground State (Low E)":
            E_lower = 0
            # Partition function approximation (rotational ~ T)
            # S ~ 1/T * exp(0) -> decays with T
            line_strength = (296 / T_atmos) 
        else:
            E_lower = 1000 # High energy state (cm-1)
            # S ~ 1/T * exp(-E/kT) -> grows then decays, but grows at earth temps
            line_strength = (296 / T_atmos) * np.exp(-E_lower * ((1/T_atmos) - (1/296))) * 5.0

        # 2. Calculate Optical Depth from Line Strength
        tau = line_strength * gas_amount
        
        # 3. Calculate Emissivity (Kirchhoff's Law: e = a = 1 - exp(-tau))
        emissivity = 1 - np.exp(-tau)
        
        # 4. Radiative Transfer
        B_surface = 5.67e-8 * T_surface**4
        B_atmos = 5.67e-8 * T_atmos**4
        I_top = B_surface * (1 - emissivity) + B_atmos * emissivity

        # --- PLOTTING ---
        fig3, ax3 = plt.subplots(1, 2, figsize=(10, 5))
        
        # Subplot 1: Line Strength vs Temperature Curve
        t_range = np.linspace(100, 1000, 100)
        if energy_state == "Ground State (Low E)":
             s_curve = (296 / t_range)
             title_s = "Line Strength (Decreases with T)"
        else:
             s_curve = (296 / t_range) * np.exp(-E_lower * ((1/t_range) - (1/296))) * 5.0
             title_s = "Line Strength (Increases with T)"
             
        ax3[0].plot(t_range, s_curve, color='green', linestyle='--')
        ax3[0].scatter([T_atmos], [line_strength], color='green', s=100, label='Current T')
        ax3[0].set_xlabel("Temperature (K)")
        ax3[0].set_ylabel("Line Strength S(T)")
        ax3[0].set_title(title_s)
        ax3[0].grid(True)
        ax3[0].legend()

        # Subplot 2: Radiative Output
        labels = ['Surface\n(Source)', 'Atmos\n(Emission)', 'Total\n(To Space)']
        # Scaled for visibility
        values = [B_surface, B_atmos * emissivity, I_top] 
        colors = ['red', 'blue', 'purple']
        
        ax3[1].bar(labels, values, color=colors)
        ax3[1].set_ylabel("Flux (W/m2)")
        ax3[1].set_title(f"Radiative Balance\n(Calculated Emissivity $\epsilon$ = {emissivity:.2f})")
        
        st.pyplot(fig3)

# ==========================================
# TAB 3: REMOTE SOUNDING (WEIGHTING FUNCTIONS)
# ==========================================
with tab3:
    st.header("Remote Sounding: The Weighting Function")
    st.markdown(r"$$I_{\nu} = \int_{0}^{\infty} B_{\nu}(T(z)) W(z) dz$$")
    st.markdown("Where in the atmosphere is the satellite actually looking?")

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Channel Selection")
        # Control the absorption coefficient (simulating picking different frequencies)
        abs_strength = st.slider("Absorption Strength (k)", 0.1, 10.0, 1.0, help="High k = Opaque (Peaking high). Low k = Transparent (Peaking low).")
        
        scale_height = st.number_input("Scale Height H (km)", value=7.0)
        
        st.info("""
        **Teaching Point:**
        Adjust 'Absorption Strength'. 
        * **Strong absorption:** The satellite only sees the top layers (Weighting function peaks high).
        * **Weak absorption:** The satellite sees deeper (Weighting function peaks low).
        """)

    with col2:
        # 1. Define vertical grid
        z = np.linspace(0, 80, 200) # 0 to 80 km
        
        # 2. Define Density Profile (Exponential)
        rho = np.exp(-z / scale_height)
        
        # 3. Calculate Optical Depth (integral from Space z=inf down to z)
        # For exponential atmosphere, tau(z) = H * k * rho(z) roughly
        tau = abs_strength * scale_height * rho
        
        # 4. Transmittance (from space to z)
        transmittance = np.exp(-tau)
        
        # 5. Weighting Function d(Trans)/dz
        # W(z) = k * rho * exp(-tau)
        W = abs_strength * rho * transmittance
        
        # Normalized W for plotting
        W_norm = W / np.max(W)
        
        # --- PLOTTING ---
        fig4, ax4 = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        
        # Plot 1: Optical Depth & Transmittance
        ax4[0].plot(transmittance, z, label="Transmittance to Space", color='green', linestyle='--')
        ax4[0].plot(tau, z, label="Optical Depth", color='gray')
        ax4[0].set_xlim(-0.1, 5)
        ax4[0].set_ylim(0, 80)
        ax4[0].set_ylabel("Altitude (km)")
        ax4[0].set_xlabel("Value")
        ax4[0].legend()
        ax4[0].grid(True)
        ax4[0].set_title("Optical Properties")
        
        # Plot 2: The Weighting Function
        ax4[1].plot(W_norm, z, color='purple', linewidth=3)
        ax4[1].fill_betweenx(z, W_norm, color='purple', alpha=0.3)
        ax4[1].set_xlabel("Weighting Function (Normalized)")
        ax4[1].set_title(f"Where the Satellite 'Sees' \n (Peak ~ {z[np.argmax(W_norm)]:.1f} km)")
        ax4[1].grid(True)
        
        # Add a line for the peak
        peak_z = z[np.argmax(W_norm)]
        ax4[1].axhline(peak_z, color='red', linestyle=':')
        
        st.pyplot(fig4)
        

# ==========================================
# TAB 4: LBL INTEGRATION (FELS & SCHWARZKOPF MATRIX)
# ==========================================
with tab4: # Add this to your tab list definition at the top too!
    st.header("The Fels & Schwarzkopf (1981) Approach")
    st.markdown("""
    **The Problem:** Integrating 100,000 spectral lines for $CO_2$ every few seconds is impossible for a climate model.
    
    **The Solution:** We **pre-calculate** the Transmission Function between every pair of levels $(z_i, z_j)$ and store it in a matrix.
    """)
    
    col1, col2 = st.columns([1, 2])

    # --- SIMULATION SETUP ---
    # Create a vertical grid (Mesosphere to Surface)
    nz = 50
    z_grid = np.linspace(0, 80, nz) # 0 to 80 km
    dz = z_grid[1] - z_grid[0]
    
    # Standard Atmosphere Temp Profile (Approximation)
    def get_temp(z):
        if z < 12: return 288 - 6.5*z
        elif z < 20: return 210
        elif z < 47: return 210 + 2.8*(z-20)
        elif z < 51: return 285.6
        else: return 285.6 - 2.8*(z-51)
    
    T_profile_ref = np.array([get_temp(zi) for zi in z_grid])

    with col1:
        st.subheader("Matrix Controls")
        co2_conc = st.slider("CO2 Concentration (Multiplier)", 0.1, 5.0, 1.0, help="1.0 = Present Day. 4.0 = 4xCO2")
        
        st.write("---")
        st.markdown("**Perturb Temperature**")
        temp_offset = st.slider("Stratospheric Warming/Cooling (K)", -20, 20, 0)
        
        # Apply perturbation to profile
        T_profile = T_profile_ref.copy()
        # Add offset only to middle atmosphere (20km+)
        T_profile[z_grid > 20] += temp_offset

    # --- PHYSICS ENGINE (The Matrix Calculation) ---
    # We simulate a 15um band transmission matrix.
    # Physics: Transmission depends on the Mass Path between level i and j.
    # T_ij = exp( - k * |u_i - u_j| )
    
    # 1. Calculate pressure/density proxy (exponential decay)
    H = 7.0 # Scale height
    p_grid = 1013 * np.exp(-z_grid/H)
    
    # 2. Build the Transmission Matrix (The Fels-Schwarzkopf Look-up Table)
    # In reality, this is loaded from a massive file. We calculate a proxy here.
    # Strong absorption regime (Square root limit approximation for CO2)
    
    T_matrix = np.zeros((nz, nz))
    k_co2 = 0.1 * co2_conc # Absorption strength scaled by user slider
    
    for i in range(nz):
        for j in range(nz):
            # Mass path difference roughly proportional to pressure difference
            delta_p = np.abs(p_grid[i] - p_grid[j])
            
            # Transmission (approximation for strong line limit)
            # T = exp(- sqrt(k * dp)) 
            tr = np.exp(-np.sqrt(k_co2 * delta_p))
            T_matrix[i, j] = tr

    # --- FLUX & COOLING RATE CALCULATION ---
    # F_up(z) = B_surface * T(0, z) + Integral( B(z') d T(z', z) )
    # Simplified discrete exchange:
    
    cooling_rate = np.zeros(nz)
    sigma = 5.67e-8
    
    # Calculate Flux Divergence (Net Cooling)
    # This is a simplified "Cool to Space" + "Exchange" approximation
    for i in range(1, nz-1):
        # 1. Cool to Space (CTS) Term
        # Energy emitted by layer i that escapes to top (index nz-1)
        B_i = sigma * T_profile[i]**4
        trans_to_space = T_matrix[i, -1]
        #CTS is proportional to emission * ability to escape
        
        # 2. Exchange Term (very simplified for demo)
        # Net flux divergence dF/dz
        # We simulate the cooling rate resulting from the matrix
        
        # Pure Matrix implementation of F_net
        F_net = 0
        for j in range(nz):
            # Exchange between i and j
            weight = (T_matrix[i, j] - T_matrix[i, j-1]) if j>0 else 0
            F_net += sigma * T_profile[j]**4 * weight
            
        # Approximation: Cooling rate is related to local emission vs incoming
        # Here we use the Newtonian Cooling approximation modified by the matrix opacity
        # If matrix is opaque (T near 0), cooling is local. If transparent, cooling is to space.
        
        # Use CTS approximation for visualization clarity (Fels 1981 dominant term in Meso)
        cooling_rate[i] = - (B_i * trans_to_space) * 0.1 # Scaling for demo units

    # --- VISUALIZATION ---
    with col2:
        # Plot 1: The Transmission Matrix
        fig_mat, ax_mat = plt.subplots(figsize=(6, 5))
        c = ax_mat.imshow(T_matrix, origin='lower', extent=[0, 80, 0, 80], cmap='Greens_r', vmin=0, vmax=1)
        ax_mat.set_title(f"Pre-Calculated Transmission Matrix\n(CO2 x {co2_conc})")
        ax_mat.set_xlabel("Emitting Level z' (km)")
        ax_mat.set_ylabel("Receiving Level z (km)")
        plt.colorbar(c, label="Transmission $\mathcal{T}(z, z')$")
        st.pyplot(fig_mat)
        
        st.caption("Dark Green = Opaque (No exchange). White = Transparent (High exchange). The diagonal is always 1.")

    # Plot 2: Cooling Rate & Temp
    st.write("---")
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig_t, ax_t = plt.subplots(figsize=(4, 6))
        ax_t.plot(T_profile, z_grid, color='red')
        ax_t.set_xlabel("Temperature (K)")
        ax_t.set_ylabel("Altitude (km)")
        ax_t.set_title("Temperature Profile")
        ax_t.grid(True)
        st.pyplot(fig_t)
        
    with col_b:
        fig_c, ax_c = plt.subplots(figsize=(4, 6))
        ax_c.plot(cooling_rate, z_grid, color='blue')
        ax_c.set_xlabel("Cooling Rate (K/day approx)")
        ax_c.set_title("Resulting Cooling Rate")
        ax_c.set_xlim(-5, 0)
        ax_c.grid(True)
        st.pyplot(fig_c)
        
# ==========================================
# TAB 5: ANGULAR INTEGRATION (Diffusivity Factor)
# ==========================================

with tab5:
    st.header("The Diffusivity Factor Approximation")
    st.markdown(r"""
    **The Question:** Can we replace the complex integration over all angles ($E_3$) with a simple exponential decay ($e^{-\beta \tau}$)?
    
    **The Approximation:**
    $$ 2 E_3(\tau) \approx e^{-1.66 \tau} $$ 
    
    **Calculation**
    $$ 2×E_3(1)≈2×0.1097=0.219$$ (Exact Flux Transmission)
    
    Ramanathan (1985) shows that the diffusivity factor actually changes with optical depth:

    At τ→0 (Top of atmosphere): Factor →2.0.

    At τ→∞ (Deep atmosphere): Factor →1.5.

    At τ=1: Factor ≈1.53.
    
    The textbook justifies β≈1.66 (5/3) based on Heating Rates, not just Transmission.
    
    """)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Tune the Factor")
        beta = st.slider("Diffusivity Factor (β)", 1.0, 2.5, 1.66, step=0.01)
        show_ramanathan = st.checkbox("Show Ramanathan's Variable Factor (Advanced)")
        st.info("""
        * **β = 1.0:** Assumes all light goes straight up (Vertical Beam).
        * **β = 2.0:** Assumes a very slanted average path ($60^\circ$).
        * **β = 1.66:** The standard approximation.
        """)

    with col2:
        # Generate Optical Depths
        tau = np.linspace(0, 3, 100)
        
        # 1. Exact Solution (Flux Transmission = 2 * E3(tau))
        # scipy.special.expn(n, x) calculates En(x)
        T_exact = 2 * sp.expn(3, tau)
        
        # 2. Approximation
        T_approx = np.exp(-beta * tau)
        
        # 3. Beam Transmission (Vertical) for comparison
        T_vertical = np.exp(-1.0 * tau)
        
        fig5, ax5 = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        if show_ramanathan:
            # Ramanathan et al. (1985) Eq 2.4.33a
            # beta becomes a function of tau, not a constant
            beta_var = 1.5 + (0.5 / (1 + 4*tau + 10*tau**2))
            T_ramanathan = np.exp(-beta_var * tau)
            
            ax5[0].plot(tau, T_ramanathan, label="Variable Diffusivity (Eq 2.4.33a)", color='blue', linestyle='-.')
            ax5[0].legend()
            
            # Update error plot
            error_ram = (T_ramanathan - T_exact)
            ax5[1].plot(tau, error_ram, color='blue', linestyle='-.', label="Variable Factor Error")
            ax5[1].legend()
        # Plot 1: Transmission curves
        ax5[0].plot(tau, T_exact, label="Exact Flux Transmission ($2E_3$)", color='black', linewidth=3)
        ax5[0].plot(tau, T_approx, label=f"Approximation ($e^{{-{beta}\\tau}}$)", color='red', linestyle='--')
        ax5[0].plot(tau, T_vertical, label="Vertical Beam ($e^{-\\tau}$)", color='green', alpha=0.3, linestyle=':')
        
        ax5[0].set_ylabel("Transmission")
        ax5[0].set_title("Exact vs. Approximation")
        ax5[0].legend()
        ax5[0].grid(True)
        
        # Plot 2: Relative Error
        # Avoid divide by zero at infinity (though T approaches 0)
        error = (T_approx - T_exact)
        
        ax5[1].plot(tau, error, color='red')
        ax5[1].axhline(0, color='black', linestyle='-')
        ax5[1].set_ylabel("Error (Approx - Exact)")
        ax5[1].set_xlabel("Optical Depth (τ)")
        ax5[1].set_title(f"Error Residuals (Best at β ≈ 1.66)")
        ax5[1].grid(True)
        
        st.pyplot(fig5)
        
# ==========================================
# TAB 6: LINE SHAPES (Lorentz vs Doppler vs Voigt)
# ==========================================


with tab6:
    st.header("The Battle of the Shapes")
    st.markdown(r"""
    Comparison of broadening mechanisms.
    * **Doppler:** $\alpha_D$ (Temperature dependent). Gaussian shape.
    * **Lorentz:** $\alpha_L$ (Pressure dependent). Cauchy shape.
    * **Voigt:** Convolution of both.
    """)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Atmospheric Conditions")
        P_hPa = st.slider("Pressure (hPa)", 0.01, 1000.0, 10.0, format="%.2f")
        T_K = st.slider("Temperature (K)", 100, 1000, 250)
        
        st.divider()
        st.subheader("Visualization Control")
        window_width = st.slider("Window Width (cm-1)", 0.001, 0.5, 0.05, format="%.4f")
        use_log = st.checkbox("Logarithmic Y-Axis (See the wings!)", value=True)
        
    with col2:
        # --- PHYSICS ENGINE ---
        nu = np.linspace(-window_width, window_width, 2000) # Higher res for sharp peaks
        
        # Constants
        nu0 = 667.0  # CO2 band center
        c = 3e8
        k_boltz = 1.38e-23
        mass_co2 = 44 * 1.66e-27
        
        # 1. Calculate Widths (Using AHL standard definitions)
        # Doppler e-folding width (AHL Eq 2.3.5)
        alpha_D_param = (nu0 / c) * np.sqrt(2 * k_boltz * T_K / mass_co2)
        # Convert to HWHM for consistency in text display
        hwhm_D = alpha_D_param * np.sqrt(np.log(2))
        
        # Lorentz Width (HWHM)
        alpha_L = 0.07 * (P_hPa / 1000.0) * np.sqrt(296/T_K)
        
        # 2. Calculate Profiles (Area Normalized = 1)
        
        # Lorentz (AHL Eq 2.3.1)
        L = (1 / (np.pi * alpha_L)) / (1 + (nu / alpha_L)**2)
        
        # Doppler (AHL Eq 2.3.4)
        # Note: 1/(alpha_D * sqrt(pi)) * exp(-(nu/alpha_D)^2)
        D = (1 / (alpha_D_param * np.sqrt(np.pi))) * np.exp(-(nu / alpha_D_param)**2)
        
        # Voigt (Scipy implementation)
        # Scipy wofz inputs:
        # sigma = Doppler standard deviation = alpha_D_param / sqrt(2)
        # gamma = Lorentz HWHM = alpha_L
        sigma = alpha_D_param / np.sqrt(2)
        z = (nu + 1j*alpha_L) / (sigma * np.sqrt(2))
        V = np.real(sp.wofz(z)) / (sigma * np.sqrt(2*np.pi))
        
        # --- PLOTTING ---
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        
        # Plot all on ONE axis to show relative strengths
        ax6.plot(nu, L, color='blue', linestyle='--', label=f'Lorentz (P={P_hPa} hPa)')
        ax6.plot(nu, D, color='green', linestyle='--', label=f'Doppler (T={T_K} K)')
        ax6.plot(nu, V, color='black', linewidth=2, alpha=0.8, label='Voigt (Actual)')
        
        if use_log:
            ax6.set_yscale('log')
            ax6.set_ylim(bottom=1e-4 * np.max(V)) # Prevent log(0) issues
            ax6.set_title("Line Shapes (Log Scale) - Notice the Wings!")
        else:
            ax6.set_title("Line Shapes (Linear Scale) - Notice the Peak!")

        ax6.set_xlabel("Frequency from Center ($cm^{-1}$)")
        ax6.set_ylabel("Normalized Line Shape $f(\\nu)$")
        ax6.legend()
        ax6.grid(True, which="both", alpha=0.3)
        
        st.pyplot(fig6)
        
        # Metrics Display
        st.info(f"""
        **Regime Check:**
        * Lorentz Width (HWHM): **{alpha_L:.5f}** $cm^{{-1}}$
        * Doppler Width (HWHM): **{hwhm_D:.5f}** $cm^{{-1}}$
        * Ratio ($\alpha_L / \alpha_D$): **{alpha_L/hwhm_D:.2f}**
        
        *(Ratio > 1 means Pressure Broadening dominates. Ratio < 1 means Doppler Broadening dominates.)*
        """)
        
        # ==========================================
# TAB 7: CURVE OF GROWTH
# ==========================================
with tab7:
    st.header("The Curve of Growth")
    st.markdown(r"""
    **How does absorption ($W$) grow as we add more gas ($u$)?**
    
    1.  **Linear Regime:** Optically thin. Absorption grows linearly with gas amount.
    2.  **Saturation (Flat) Regime:** The line center is opaque. Adding gas barely helps.
    3.  **Square Root Regime:** The wings take over (only for Pressure Broadening).
    """)

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Line Parameters")
        line_type = st.radio("Line Shape Model", ["Lorentz (Pressure Broadened)", "Doppler (Thermal Only)"])
        
        st.write("---")
        st.info("""
        **Look for the Slope:**
        * **Slope 1:** Linear ($W \propto u$)
        * **Slope 0:** Saturation (Doppler stays here longer)
        * **Slope 0.5:** Square Root ($W \propto \sqrt{u}$) - *Lorentz only!*
        """)
        
    with col2:
        # --- PHYSICS ENGINE ---
        
        # 1. Setup frequency grid (Needs to be wide enough for wings, sharp enough for core)
        # We work in dimensionless units where line width alpha = 1
        nu = np.linspace(-100, 100, 5000) 
        
        # 2. Define Shape Profiles (Normalized to peak = 1 for simplicity of tau definition)
        if line_type == "Lorentz (Pressure Broadened)":
            # Lorentzian: 1 / (1 + x^2)
            # We assume HWHM = 1
            phi = 1 / (1 + nu**2)
            label_text = "Lorentzian Curve of Growth"
            color_line = "blue"
        else:
            # Doppler: exp(-x^2)
            # We assume Doppler Width = 1
            phi = np.exp(-nu**2)
            label_text = "Doppler Curve of Growth"
            color_line = "green"
            
        # 3. Calculate Equivalent Width (W) for varying Optical Depths (u)
        # We range u (path length/amount) from very small (0.01) to massive (10,000)
        u_range = np.logspace(-2, 4, 50) # Logarithmic steps
        W_values = []
        
        for u in u_range:
            # Optical depth profile tau(nu) = u * phi(nu)
            tau_nu = u * phi
            
            # Absorptance A(nu) = 1 - exp(-tau)
            A_nu = 1 - np.exp(-tau_nu)
            
            # Equivalent Width W = Integral of A(nu) d_nu
            # Using trapezoidal rule
            W = np.trapz(A_nu, nu)
            W_values.append(W)
            
        # --- PLOTTING ---
        fig7, ax7 = plt.subplots(figsize=(8, 6))
        
        # Log-Log Plot is standard for Curve of Growth
        ax7.loglog(u_range, W_values, marker='o', markersize=4, color=color_line, linewidth=2)
        
        # Add Reference Slopes (The "Regimes")
        # 1. Linear Limit (Low u) - Reference line
        ax7.loglog(u_range[:10], u_range[:10] * (W_values[0]/u_range[0]), 
                   linestyle=':', color='gray', label="Linear Limit (Slope 1)")
        
        # 2. Square Root Limit (High u) - Only usually valid for Lorentz
        if line_type == "Lorentz (Pressure Broadened)":
            # Scale the sqrt line to match the end of the data
            scale_factor = W_values[-1] / np.sqrt(u_range[-1])
            ax7.loglog(u_range[-20:], np.sqrt(u_range[-20:]) * scale_factor, 
                       linestyle='--', color='red', label="Square Root Limit (Slope 1/2)")
        
        ax7.set_xlabel("Optical Depth at Line Center ($u \propto$ Amount of Gas)")
        ax7.set_ylabel("Equivalent Width $W$ (Total Absorption)")
        ax7.set_title(label_text)
        ax7.grid(True, which="both", alpha=0.3)
        ax7.legend()
        
        st.pyplot(fig7)
        
        # Educational check
        if line_type == "Doppler (Thermal Only)" and u_range[-1] > 100:
            st.warning("Notice how the Doppler line 'stalls' (saturates) and barely grows? Without pressure broadening (Lorentz wings), the atmosphere effectively becomes opaque and stops absorbing new energy quickly.")
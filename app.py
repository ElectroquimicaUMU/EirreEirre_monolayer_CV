import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import io

from main import CVsim, FRT

st.set_page_config(page_title="MH vs BV Simulation", layout="wide")
st.title("Marcus–Hush vs Butler–Volmer CV Simulation")

# Sidebar parameters
with st.sidebar:
    st.header("Simulation Parameters")
    lambda1 = st.slider("λ (Marcus–Hush)", 0.1, 2.0, 0.5, step=0.1)
    k01 = st.number_input("k₀₁ (s⁻¹)", value=0.1)
    k02 = st.number_input("k₀₂ (s⁻¹)", value=0.1)
    E02 = st.number_input("E₀₂ (V)", value=-0.25)
    alpha = st.slider("α (BV transfer coeff)", 0.1, 1.0, 0.5, step=0.1)
    Es = st.number_input("Step Size (Es) [V]", value=1e-3)
    Ein = st.number_input("Initial E (Ein)", value=0.25)
    Efin = st.number_input("Final E (Efin)", value=-0.7)
    rate = st.number_input("Scan Rate (V/s)", value=0.1)

# Run simulation
result = CVsim(lambda1 * FRT, k01, k02, E02, alpha, Es, Ein, Efin, rate)
E = result.Pot[1:]

# ------------------ Voltammogram Plot ------------------
st.subheader("Voltammogram (Ψ vs E)")
fig1, ax1 = plt.subplots()
ax1.plot(E, result.IntMH[1:], label="MH")
ax1.plot(E, result.IntBV[1:], "r--", label="BV")
ax1.set_xlabel("E - E₀₁ / V")
ax1.set_ylabel("Ψ (Bard)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# ------------------ Surface Excess Plot ------------------
st.subheader("Surface Excesses (MH only)")
fig2, ax2 = plt.subplots()
ax2.plot(E, result.fO[1:], label="fO")
ax2.plot(E, result.fR[1:], label="fR")
ax2.plot(E, result.fI[1:], label="fI")
ax2.set_xlabel("E / V")
ax2.set_ylabel("Surface excess")
ax2.set_title("fO, fR, fI vs potential")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# ------------------ File Downloads ------------------
st.subheader("Download Data as .txt")

def make_txt_download(filename, header, data):
    buffer = io.StringIO()
    np.savetxt(buffer, data, header=header)
    st.download_button(
        label=f"Download {filename}",
        data=buffer.getvalue(),
        file_name=filename,
        mime="text/plain"
    )

# Prepare data
mh_data = np.column_stack((E, result.IntMH[1:]))
bv_data = np.column_stack((E, result.IntBV[1:]))
surf_data = np.column_stack((E, result.fO[1:], result.fR[1:], result.fI[1:]))

# Download buttons
make_txt_download("MH_curve.txt", "E (V)\tIntMH", mh_data)
make_txt_download("BV_curve.txt", "E (V)\tIntBV", bv_data)
make_txt_download("surface_excesses.txt", "E (V)\tfO\tfR\tfI", surf_data)
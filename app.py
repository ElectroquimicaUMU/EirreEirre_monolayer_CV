import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io

from main import CVsim, FRT

st.set_page_config(page_title="MH vs BV CV Simulation", layout="wide")
st.title("Marcus–Hush vs Butler–Volmer CV Simulation")

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Simulation Parameters")

    lambda1 = st.slider("λ (eV)", 0.1, 2.0, 0.5, 0.1)
    surface_model = st.selectbox("Surface excess model", ["MH", "BV"])

    k01 = st.number_input("k₀₁ (s⁻¹)", value=0.1)
    k02 = st.number_input("k₀₂ (s⁻¹)", value=0.1)
    E02 = st.number_input("E₀₂ (V)", value=-0.25)
    alpha = st.slider("α", 0.1, 1.0, 0.5, 0.05)

    Es = st.number_input("Potential step Es (V)", value=1e-3, format="%.1e")
    Ein = st.number_input("Initial potential Ein (V)", value=0.25)
    Efin = st.number_input("Final potential Efin (V)", value=-0.7)
    rate = st.number_input("Scan rate (V/s)", value=0.1)

# ---------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------
result = CVsim(
    lambda1=lambda1 * FRT,
    k01=k01,
    k02=k02,
    E02=E02,
    alpha=alpha,
    Es=Es,
    Ein=Ein,
    Efin=Efin,
    rate=rate,
    surface_model=surface_model,
)

E = result.Pot[1:]

# ---------------------------------------------------------------------
# Voltammograms
# ---------------------------------------------------------------------
st.subheader("Voltammogram")

fig1, ax1 = plt.subplots()
ax1.plot(E, result.IntMH[1:], label="MH")
ax1.plot(E, result.IntBV[1:], "r--", label="BV")
ax1.set_xlabel("E / V")
ax1.set_ylabel("Ψ (Bard)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# ---------------------------------------------------------------------
# Surface excesses
# ---------------------------------------------------------------------
st.subheader(f"Surface excesses ({surface_model})")

fig2, ax2 = plt.subplots()
ax2.plot(E, result.fO[1:], label="fO")
ax2.plot(E, result.fR[1:], label="fR")
ax2.plot(E, result.fI[1:], label="fI")
ax2.set_xlabel("E / V")
ax2.set_ylabel("Surface excess")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# ---------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------
st.subheader("Download data")

def download_txt(label, filename, header, data):
    buffer = io.StringIO()
    np.savetxt(buffer, data, header=header)
    st.download_button(label, buffer.getvalue(), filename, "text/plain")

download_txt(
    "Download MH curve",
    "MH_curve.txt",
    "E (V)\tIntMH",
    np.column_stack((E, result.IntMH[1:])),
)

download_txt(
    "Download BV curve",
    "BV_curve.txt",
    "E (V)\tIntBV",
    np.column_stack((E, result.IntBV[1:])),
)

download_txt(
    f"Download surface excesses ({surface_model})",
    f"surface_excess_{surface_model}.txt",
    "E (V)\tfO\tfR\tfI",
    np.column_stack((E, result.fO[1:], result.fR[1:], result.fI[1:])),
)

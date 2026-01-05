import streamlit as st
import matplotlib.pyplot as plt
from main import CVsim, FRT

st.set_page_config(page_title="MH vs BV Simulation", layout="wide")
st.title("Marcus-Hush vs Butler-Volmer Simulation")

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

Pot, IntMH, IntBV, kMH1_s, kMH2_s, kBV1_s, kBV2_s = CVsim(
    lambda1 * FRT, k01, k02, E02, alpha, Es, Ein, Efin, rate
)

E = Pot[1:]

# --- Voltammograms ---
st.subheader("Voltammogram (Ψ vs E)")
fig1, ax1 = plt.subplots()
ax1.plot(E, IntMH[1:], label="MH")
ax1.plot(E, IntBV[1:], "r--", label="BV")
ax1.set_xlabel("E - E₀₁ / V")
ax1.set_ylabel("Ψ (Bard)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# --- Rate Constants ---
st.subheader("Rate Constants k_red(E)")
fig2, ax2 = plt.subplots()
E_half = E[:len(E)//2]
ax2.semilogy(E_half, kBV1_s[1:len(E)//2], "r--", label="BV k_red, step 1")
ax2.semilogy(E_half, kBV2_s[1:len(E)//2], "m--", label="BV k_red, step 2")
ax2.semilogy(E_half, kMH1_s[1:len(E)//2], "b", label="MH k_red, step 1")
ax2.semilogy(E_half, kMH2_s[1:len(E)//2], "c", label="MH k_red, step 2")
ax2.set_xlabel("E - E₀₁ / V")
ax2.set_ylabel("k_red (s⁻¹)")
ax2.set_title("Reduction Rate Constants")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from main import CVsim, FRT

st.set_page_config(layout="wide")
st.title("Marcus–Hush vs Butler–Volmer CV Simulator")

with st.sidebar:
    lambda1 = st.slider("λ (eV)", 0.1, 2.0, 0.5, 0.1)
    surface_model = st.selectbox("Surface excess model", ["MH", "BV"])
    k01 = st.number_input("k01 (s⁻¹)", 0.1)
    k02 = st.number_input("k02 (s⁻¹)", 0.1)
    E02 = st.number_input("E02 (V)", -0.25)
    alpha = st.slider("α", 0.1, 1.0, 0.5)
    Es = st.number_input("Es (V)", 1e-3, format="%.1e")
    Ein = st.number_input("Ein (V)", 0.25)
    Efin = st.number_input("Efin (V)", -0.7)
    rate = st.number_input("Scan rate (V/s)", 0.1)

res = CVsim(
    lambda1 * FRT,
    k01,
    k02,
    E02,
    alpha,
    Es,
    Ein,
    Efin,
    rate,
    surface_model,
)

E = res.Pot[1:]

# --- CV ---
st.subheader("Voltammograms")
fig, ax = plt.subplots()
ax.plot(E, res.IntMH[1:], label="MH")
ax.plot(E, res.IntBV[1:], "--", label="BV")
ax.set_xlabel("E / V")
ax.set_ylabel("Ψ")
ax.legend()
st.pyplot(fig)

# --- Surface excess ---
st.subheader(f"Surface excesses ({surface_model})")
fig, ax = plt.subplots()
ax.plot(E, res.fO[1:], label="fO")
ax.plot(E, res.fR[1:], label="fR")
ax.plot(E, res.fI[1:], label="fI")
ax.legend()
st.pyplot(fig)

# --- Rate constants ---
st.subheader("Rate constants (k_red)")
fig, ax = plt.subplots()
ax.semilogy(E, res.kMH1red_s[1:], label="MH k_red 1")
ax.semilogy(E, res.kMH2red_s[1:], label="MH k_red 2")
ax.semilogy(E, res.kBV1red_s[1:], "--", label="BV k_red 1")
ax.semilogy(E, res.kBV2red_s[1:], "--", label="BV k_red 2")
ax.legend()
st.pyplot(fig)

# --- Peaks ---
st.subheader("Voltammetric peak coordinates")
df = pd.DataFrame(res.peaks).T
df.columns = ["E_peak 1 (V)", "I_peak 1", "E_peak 2 (V)", "I_peak 2"]
st.dataframe(df)

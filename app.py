import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

from main import CVsim, FRT

st.set_page_config(layout="wide")
st.title("Marcus–Hush vs Butler–Volmer CV Simulator")

# ---------------------------------------------------------------------
with st.sidebar:
    lambda1 = st.number_input("λ (eV)", value=0.5, step=0.01)
    surface_model = st.selectbox("Surface excess model", ["MH", "BV"])
    k01 = st.number_input("k01 (s⁻¹)", value=0.1, step=0.01, format="%.3f")
    k02 = st.number_input("k02 (s⁻¹)", value=0.1, step=0.01, format="%.3f")
    E02 = st.number_input("E02 (V)", value=-0.25, step=0.001, format="%.3f")
    alpha = st.number_input("α", value=0.5, step=0.01)
    Es = st.number_input("Es (V)", value=1e-3, step=1e-4, format="%.4f")
    Ein = st.number_input("Ein (V)", value=0.25, step=0.001, format="%.3f")
    Efin = st.number_input("Efin (V)", value=-0.7, step=0.001, format="%.3f")
    rate = st.number_input("Scan rate (V/s)", value=0.1, step=0.01, format="%.3f")

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

# ---------------------------------------------------------------------
st.subheader("Voltammograms")
fig, ax = plt.subplots()
ax.plot(E, res.IntMH[1:], label="MH")
ax.plot(E, res.IntBV[1:], "--", label="BV")
ax.set_xlabel("E / V")
ax.set_ylabel("Ψ")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------------------
st.subheader(f"Surface excesses ({surface_model})")
fig, ax = plt.subplots()
ax.plot(E, res.fO[1:], label="fO")
ax.plot(E, res.fR[1:], label="fR")
ax.plot(E, res.fI[1:], label="fI")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------------------
st.subheader("Rate constants (k_red)")
fig, ax = plt.subplots()
ax.semilogy(E, res.kMH1red_s[1:], label="MH k_red 1")
ax.semilogy(E, res.kMH2red_s[1:], label="MH k_red 2")
ax.semilogy(E, res.kBV1red_s[1:], "--", label="BV k_red 1")
ax.semilogy(E, res.kBV2red_s[1:], "--", label="BV k_red 2")
ax.set_xlabel("E / V")
ax.set_ylabel("k_red (s⁻¹)")
ax.legend()
st.pyplot(fig)

# ---------------------------------------------------------------------
st.subheader("Detected peak coordinates")
rows = []
for model, peaks in res.peaks.items():
    for n in [1, 2]:
        if f"E_peak{n}" in peaks:
            rows.append(
                {
                    "Model": model,
                    "Peak": n,
                    "E (V)": peaks[f"E_peak{n}"],
                    "I": peaks[f"I_peak{n}"],
                }
            )
df = pd.DataFrame(rows)
st.dataframe(df)

# ---------------------------------------------------------------------
st.subheader("📥 Download data as .txt")

def download_txt(label, filename, header, data):
    buf = io.StringIO()
    np.savetxt(buf, data, header=header)
    st.download_button(label, buf.getvalue(), file_name=filename, mime="text/plain")

# --- Voltammograms ---
download_txt(
    "Download MH voltammogram",
    "MH_curve.txt",
    "E (V)\tIntMH",
    np.column_stack((E, res.IntMH[1:])),
)

download_txt(
    "Download BV voltammogram",
    "BV_curve.txt",
    "E (V)\tIntBV",
    np.column_stack((E, res.IntBV[1:])),
)

# --- Surface excesses ---
download_txt(
    f"Download surface excesses ({surface_model})",
    f"surface_excess_{surface_model}.txt",
    "E (V)\tfO\tfR\tfI",
    np.column_stack((E, res.fO[1:], res.fR[1:], res.fI[1:])),
)

# --- Rate constants ---
download_txt(
    "Download MH rate constants",
    "MH_rates.txt",
    "E (V)\tk_red1\tk_red2\tk_ox1\tk_ox2",
    np.column_stack((E, res.kMH1red_s[1:], res.kMH2red_s[1:], res.kMH1ox_s[1:], res.kMH2ox_s[1:])),
)

download_txt(
    "Download BV rate constants",
    "BV_rates.txt",
    "E (V)\tk_red1\tk_red2\tk_ox1\tk_ox2",
    np.column_stack((E, res.kBV1red_s[1:], res.kBV2red_s[1:], res.kBV1ox_s[1:], res.kBV2ox_s[1:])),
)
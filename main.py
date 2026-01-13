import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.signal import find_peaks
from collections import namedtuple

FRT = 38.923074

CVResult = namedtuple("CVResult", [
    "Pot", "IntMH", "IntBV", "fO", "fR", "fI",
    "kMH1red_s", "kMH2red_s", "kBV1red_s", "kBV2red_s",
    "kMH1ox_s", "kMH2ox_s", "kBV1ox_s", "kBV2ox_s",
    "peaks"
])

def CVsim(lambda1, k01, k02, E02, alpha, Es, Ein, Efin, rate, surface_model="MH"):
    lambda2 = lambda1
    tau = Es / rate
    nt = int(2 * abs(Efin - Ein) / Es)

    Pot = np.zeros(nt)
    fO = np.zeros(nt)
    fR = np.zeros(nt)
    fI = np.zeros(nt)

    fO[0] = 1.0

    kMH1red = np.zeros(nt)
    kMH2red = np.zeros(nt)
    kMH1ox = np.zeros(nt)
    kMH2ox = np.zeros(nt)
    kBV1red = np.zeros(nt)
    kBV2red = np.zeros(nt)
    kBV1ox = np.zeros(nt)
    kBV2ox = np.zeros(nt)

    IntMH = np.zeros(nt)
    IntBV = np.zeros(nt)

    S01 = quad(lambda x: np.exp(-lambda1/4*(1 + x/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
    S02 = quad(lambda x: np.exp(-lambda2/4*(1 + x/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]

    for i in range(1, nt):
        Pot[i] = Pot[i-1] - Es if i < nt/2 else Pot[i-1] + Es
        nu1 = FRT * Pot[i]
        nu2 = FRT * (Pot[i] - E02)

        MH1_red = quad(lambda x: np.exp(-lambda1/4*(1 + (nu1 + x)/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
        MH2_red = quad(lambda x: np.exp(-lambda2/4*(1 + (nu2 + x)/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]
        MH1_ox  = quad(lambda x: np.exp(-lambda1/4*(1 + (-nu1 + x)/lambda1)**2)/(1 + np.exp(-x)), -50, 50)[0]
        MH2_ox  = quad(lambda x: np.exp(-lambda2/4*(1 + (-nu2 + x)/lambda2)**2)/(1 + np.exp(-x)), -50, 50)[0]

        kMH1red[i] = k01 * tau * MH1_red / S01
        kMH2red[i] = k02 * tau * MH2_red / S02
        kMH1ox[i]  = k01 * tau * MH1_ox / S01
        kMH2ox[i]  = k02 * tau * MH2_ox / S02

        kBV1red[i] = k01 * tau * np.exp(-alpha * nu1)
        kBV2red[i] = k02 * tau * np.exp(-alpha * nu2)
        kBV1ox[i]  = k01 * tau * np.exp((1 - alpha) * nu1)
        kBV2ox[i]  = k02 * tau * np.exp((1 - alpha) * nu2)

        if surface_model == "MH":
            fO[i] = fO[i-1] * np.exp(-(kMH1red[i] + kMH1ox[i]))
            den = kMH1red[i] - kMH2red[i] if abs(kMH1red[i] - kMH2red[i]) > 1e-15 else 1e-15
            fR[i] = 1 + (fR[i-1] - 1) * np.exp(-(kMH2red[i] + kMH2ox[i])) +                 fO[i-1] * (np.exp(-(kMH1red[i] + kMH1ox[i])) - np.exp(-(kMH2red[i] + kMH2ox[i]))) * kMH2red[i] / den
        else:
            fO[i] = fO[i-1] * np.exp(-(kBV1red[i] + kBV1ox[i]))
            den = kBV1red[i] - kBV2red[i] if abs(kBV1red[i] - kBV2red[i]) > 1e-15 else 1e-15
            fR[i] = 1 + (fR[i-1] - 1) * np.exp(-(kBV2red[i] + kBV2ox[i])) +                 fO[i-1] * (np.exp(-(kBV1red[i] + kBV1ox[i])) - np.exp(-(kBV2red[i] + kBV2ox[i]))) * kBV2red[i] / den

        fI[i] = 1 - fO[i] - fR[i]

        IntMH[i] = (fO[i] * kMH1red[i] + fI[i] * kMH2red[i]) / Es / FRT
        IntBV[i] = (fO[i] * kBV1red[i] + fI[i] * kBV2red[i]) / Es / FRT

    Pot = np.asarray(Pot)
    kMH1red_s = np.asarray(kMH1red) / tau
    kMH2red_s = np.asarray(kMH2red) / tau
    kMH1ox_s = np.asarray(kMH1ox) / tau
    kMH2ox_s = np.asarray(kMH2ox) / tau
    kBV1red_s = np.asarray(kBV1red) / tau
    kBV2red_s = np.asarray(kBV2red) / tau
    kBV1ox_s = np.asarray(kBV1ox) / tau
    kBV2ox_s = np.asarray(kBV2ox) / tau

    def detect_peaks(current):
        peaks, _ = find_peaks(current)
        valleys, _ = find_peaks(-current)
        peak_dict = {}
        if len(peaks) > 0:
            peak_idx = peaks[np.argmax(current[peaks])]
            peak_dict["E_peak1"] = Pot[peak_idx]
            peak_dict["I_peak1"] = current[peak_idx]
        if len(valleys) > 0:
            valley_idx = valleys[np.argmin(current[valleys])]
            peak_dict["E_peak2"] = Pot[valley_idx]
            peak_dict["I_peak2"] = current[valley_idx]
        return peak_dict

    return CVResult(
        Pot, IntMH, IntBV, fO, fR, fI,
        kMH1red_s, kMH2red_s, kBV1red_s, kBV2red_s,
        kMH1ox_s, kMH2ox_s, kBV1ox_s, kBV2ox_s,
        {"MH": detect_peaks(IntMH), "BV": detect_peaks(IntBV)}
    )
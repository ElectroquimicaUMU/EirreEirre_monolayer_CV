import numpy as np
import scipy.integrate as integrate
from collections import namedtuple

# ---------------------------------------------------------------------
FRT = 38.923074

CVResult = namedtuple(
    "CVResult",
    [
        "Pot",
        "IntMH",
        "IntBV",
        "kMH1red_s",
        "kMH2red_s",
        "kMH1ox_s",
        "kMH2ox_s",
        "kBV1red_s",
        "kBV2red_s",
        "kBV1ox_s",
        "kBV2ox_s",
        "fO",
        "fR",
        "fI",
        "peaks",
    ],
)

# ---------------------------------------------------------------------
def _find_peaks(E, I):
    """
    Identify the two most intense peaks (by |I|),
    without assuming anodic/cathodic character.
    """
    E = np.asarray(E)
    I = np.asarray(I)

    idx = np.argsort(np.abs(I))[-2:][::-1]
    i1, i2 = idx[0], idx[1]

    return {
        "E_peak1": E[i1],
        "I_peak1": I[i1],
        "E_peak2": E[i2],
        "I_peak2": I[i2],
    }

# ---------------------------------------------------------------------
def CVsim(
    lambda1,
    k01,
    k02,
    E02,
    alpha,
    Es,
    Ein,
    Efin,
    rate,
    surface_model="MH",
):
    lambda2 = lambda1
    tau = Es / rate
    nt = int(2 * abs(Efin - Ein) / Es)

    Pot = [Ein] * nt

    kMH1red = [0.0] * nt
    kMH2red = [0.0] * nt
    kBV1red = [0.0] * nt
    kBV2red = [0.0] * nt

    IntMH = [0.0] * nt
    IntBV = [0.0] * nt

    fO_MH, fI_MH, fR_MH = [1.0] * nt, [0.0] * nt, [0.0] * nt
    fO_BV, fI_BV, fR_BV = [1.0] * nt, [0.0] * nt, [0.0] * nt

    S01 = integrate.quad(
        lambda x: np.exp(-lambda1 / 4 * (1 + x / lambda1) ** 2)
        / (1 + np.exp(-x)),
        -50,
        50,
    )[0]

    S02 = integrate.quad(
        lambda x: np.exp(-lambda2 / 4 * (1 + x / lambda2) ** 2)
        / (1 + np.exp(-x)),
        -50,
        50,
    )[0]

    for i in range(1, nt):
        Pot[i] = Pot[i - 1] - Es if i < nt / 2 else Pot[i - 1] + Es
        nu1 = FRT * Pot[i]
        nu2 = FRT * (Pot[i] - E02)

        MH1 = integrate.quad(
            lambda x: np.exp(-lambda1 / 4 * (1 + (nu1 + x) / lambda1) ** 2)
            / (1 + np.exp(-x)),
            -50,
            50,
        )[0]
        MH2 = integrate.quad(
            lambda x: np.exp(-lambda2 / 4 * (1 + (nu2 + x) / lambda2) ** 2)
            / (1 + np.exp(-x)),
            -50,
            50,
        )[0]

        kMH1red[i] = k01 * tau * MH1 / S01
        kMH2red[i] = k02 * tau * MH2 / S02

        fO_MH[i] = fO_MH[i - 1] * np.exp(-kMH1red[i])
        den = kMH1red[i] - kMH2red[i] if abs(kMH1red[i] - kMH2red[i]) > 1e-15 else 1e-15
        fR_MH[i] = (
            1
            + (fR_MH[i - 1] - 1) * np.exp(-kMH2red[i])
            + fO_MH[i - 1]
            * (np.exp(-kMH1red[i]) - np.exp(-kMH2red[i]))
            * kMH2red[i]
            / den
        )
        fI_MH[i] = 1 - fO_MH[i] - fR_MH[i]
        IntMH[i] = (fO_MH[i] * kMH1red[i] + fI_MH[i] * kMH2red[i]) / Es / FRT

        kBV1red[i] = k01 * tau * np.exp(-alpha * nu1)
        kBV2red[i] = k02 * tau * np.exp(-alpha * nu2)

        fO_BV[i] = fO_BV[i - 1] * np.exp(-kBV1red[i])
        denBV = kBV1red[i] - kBV2red[i] if abs(kBV1red[i] - kBV2red[i]) > 1e-15 else 1e-15
        fR_BV[i] = (
            1
            + (fR_BV[i - 1] - 1) * np.exp(-kBV2red[i])
            + fO_BV[i - 1]
            * (np.exp(-kBV1red[i]) - np.exp(-kBV2red[i]))
            * kBV2red[i]
            / denBV
        )
        fI_BV[i] = 1 - fO_BV[i] - fR_BV[i]
        IntBV[i] = (fO_BV[i] * kBV1red[i] + fI_BV[i] * kBV2red[i]) / Es / FRT

    Pot = np.asarray(Pot)
    E = Pot[1:]

    kMH1red_s = np.asarray(kMH1red) / tau
    kMH2red_s = np.asarray(kMH2red) / tau
    kBV1red_s = np.asarray(kBV1red) / tau
    kBV2red_s = np.asarray(kBV2red) / tau

    kMH1ox_s = kMH1red_s[::-1]
    kMH2ox_s = kMH2red_s[::-1]
    kBV1ox_s = kBV1red_s[::-1]
    kBV2ox_s = kBV2red_s[::-1]

    peaks = {
        "MH": _find_peaks(E, IntMH[1:]),
        "BV": _find_peaks(E, IntBV[1:]),
    }

    if surface_model.upper() == "MH":
        fO, fR, fI = fO_MH, fR_MH, fI_MH
    else:
        fO, fR, fI = fO_BV, fR_BV, fI_BV

    return CVResult(
        Pot,
        np.asarray(IntMH),
        np.asarray(IntBV),
        kMH1red_s,
        kMH2red_s,
        kMH1ox_s,
        kMH2ox_s,
        kBV1red_s,
        kBV2red_s,
        kBV1ox_s,
        kBV2ox_s,
        np.asarray(fO),
        np.asarray(fR),
        np.asarray(fI),
        peaks,
    )

import numpy as np
import scipy.integrate as integrate
from collections import namedtuple

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
FRT = 38.923074

# ---------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------
CVResult = namedtuple(
    "CVResult",
    [
        "Pot",
        "IntMH",
        "IntBV",
        "kMH1_s",
        "kMH2_s",
        "kBV1_s",
        "kBV2_s",
        "fO",
        "fR",
        "fI",
    ],
)

# ---------------------------------------------------------------------
# CV simulation
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

    # Surface excesses (both models computed)
    fO_MH, fI_MH, fR_MH = [1.0] * nt, [0.0] * nt, [0.0] * nt
    fO_BV, fI_BV, fR_BV = [1.0] * nt, [0.0] * nt, [0.0] * nt

    # MH normalization constants
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

    # -----------------------------------------------------------------
    # CV loop
    # -----------------------------------------------------------------
    for i in range(1, nt):
        Pot[i] = Pot[i - 1] - Es if i < nt / 2 else Pot[i - 1] + Es

        nu1 = FRT * Pot[i]
        nu2 = FRT * (Pot[i] - E02)

        # -------- Marcus–Hush --------
        MH_int1 = integrate.quad(
            lambda x: np.exp(-lambda1 / 4 * (1 + (nu1 + x) / lambda1) ** 2)
            / (1 + np.exp(-x)),
            -50,
            50,
        )[0]

        MH_int2 = integrate.quad(
            lambda x: np.exp(-lambda2 / 4 * (1 + (nu2 + x) / lambda2) ** 2)
            / (1 + np.exp(-x)),
            -50,
            50,
        )[0]

        kMH1red[i] = k01 * tau * MH_int1 / S01
        kMH2red[i] = k02 * tau * MH_int2 / S02

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

        # -------- Butler–Volmer --------
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

    # Select surface excess model
    if surface_model.upper() == "MH":
        fO, fR, fI = fO_MH, fR_MH, fI_MH
    elif surface_model.upper() == "BV":
        fO, fR, fI = fO_BV, fR_BV, fI_BV
    else:
        raise ValueError("surface_model must be 'MH' or 'BV'")

    return CVResult(
        Pot=np.asarray(Pot),
        IntMH=np.asarray(IntMH),
        IntBV=np.asarray(IntBV),
        kMH1_s=np.asarray(kMH1red) / tau,
        kMH2_s=np.asarray(kMH2red) / tau,
        kBV1_s=np.asarray(kBV1red) / tau,
        kBV2_s=np.asarray(kBV2red) / tau,
        fO=np.asarray(fO),
        fR=np.asarray(fR),
        fI=np.asarray(fI),
    )

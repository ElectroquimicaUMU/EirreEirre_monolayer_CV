import numpy as np
import scipy.integrate as integrate

FRT = 38.923074

def CVsim(lambda1, k01, k02, E02, alpha, Es, Ein, Efin, rate):
    lambda2 = lambda1
    tau = Es / rate
    nt = int(2 * abs(Efin - Ein) / Es)

    Pot = [Ein] * nt
    kMH1red = [0.0] * nt
    kMH2red = [0.0] * nt
    kBV1red = [0.0] * nt
    kBV2red = [0.0] * nt
    fO = [1.0] * nt
    fI = [0.0] * nt
    fR = [0.0] * nt
    fOBV = [1.0] * nt
    fIBV = [0.0] * nt
    fRBV = [0.0] * nt
    IntMH = [0.0] * nt
    IntBV = [0.0] * nt

    S01 = integrate.quad(lambda x: np.exp(-lambda1/4*(1+(0+x)/lambda1)**2)/(1+np.exp(-x)), -50, 50)[0]
    S02 = integrate.quad(lambda x: np.exp(-lambda2/4*(1+(0+x)/lambda2)**2)/(1+np.exp(-x)), -50, 50)[0]

    for i in range(1, nt):
        Pot[i] = Pot[i-1] - Es if i < nt/2 else Pot[i-1] + Es
        nu1 = FRT * Pot[i]
        nu2 = FRT * (Pot[i] - E02)

        MH_int1 = integrate.quad(lambda x: np.exp(-lambda1/4*(1+(nu1+x)/lambda1)**2)/(1+np.exp(-x)), -50, 50)[0]
        MH_int2 = integrate.quad(lambda x: np.exp(-lambda2/4*(1+(nu2+x)/lambda2)**2)/(1+np.exp(-x)), -50, 50)[0]
        kMH1red[i] = k01 * tau * MH_int1 / S01
        kMH2red[i] = k02 * tau * MH_int2 / S02

        fO[i] = fO[i-1] * np.exp(-kMH1red[i])
        den = (kMH1red[i] - kMH2red[i]) if abs(kMH1red[i] - kMH2red[i]) > 1e-15 else 1e-15
        fR[i] = 1 + (fR[i-1] - 1) * np.exp(-kMH2red[i]) + fO[i-1] * (np.exp(-kMH1red[i]) - np.exp(-kMH2red[i])) * kMH2red[i] / den
        fI[i] = 1 - fO[i] - fR[i]
        IntMH[i] = (fO[i] * kMH1red[i] + fI[i] * kMH2red[i]) / Es / FRT

        kBV1red[i] = k01 * tau * np.exp(-alpha * nu1)
        kBV2red[i] = k02 * tau * np.exp(-alpha * nu2)

        fOBV[i] = fOBV[i-1] * np.exp(-kBV1red[i])
        denBV = (kBV1red[i] - kBV2red[i]) if abs(kBV1red[i] - kBV2red[i]) > 1e-15 else 1e-15
        fRBV[i] = 1 + (fRBV[i-1] - 1) * np.exp(-kBV2red[i]) + fOBV[i-1] * (np.exp(-kBV1red[i]) - np.exp(-kBV2red[i])) * kBV2red[i] / denBV
        fIBV[i] = 1 - fOBV[i] - fRBV[i]
        IntBV[i] = (fOBV[i] * kBV1red[i] + fIBV[i] * kBV2red[i]) / Es / FRT

    Pot = np.asarray(Pot)
    kMH1red_s = np.asarray(kMH1red) / tau
    kMH2red_s = np.asarray(kMH2red) / tau
    kBV1red_s = np.asarray(kBV1red) / tau
    kBV2red_s = np.asarray(kBV2red) / tau

    return Pot, np.asarray(IntMH), np.asarray(IntBV), kMH1red_s, kMH2red_s, kBV1red_s, kBV2red_s

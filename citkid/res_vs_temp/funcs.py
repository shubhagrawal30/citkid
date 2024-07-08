import numpy as np
from scipy.special import digamma # complex digamma function
from scipy.special import iv as I_n # modified bessel function of the first kind
from scipy.special import kv as K_n # modified bessel function of the second kind

k_B = 1.380649e-23
h = 6.62607015e-34
hbar = h / (2*np.pi)
N0_Al = 1.0737e47
N0_Nb = 6.135e48

def fr_vs_temp(temperature, fr0, D, alpha, Tc, gamma = 1):
    """
    Calculates the resonance frequency at the given temperature, including
    the Mattis-Bardeen and TLS components of the temperature dependence

    Parameters:
    temperature (float or array-like): in K
    fr0 (float): frequency at 0 K
    D (float): fitting parameter. See TLS write-up for details
    alpha (float): kinetic inductance fraction
    Tc (float): superconducting transition temperature in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits.
        This parameter should be enforced if fitting

    Returns:
    fr (float or array-like): resonance frequency(ies) at the given
        temperature(s)
    """
    kT = k_B * temperature
    Delta0 = 1.762 * k_B * Tc
    zeta = Delta0 / kT
    xi = h * fr0 / (2 * kT)

    g_tls = (np.real(digamma(1/2 + 1j * xi / np.pi)) - np.log(2 * xi)) / np.pi
    # There may be a factor of 1 / 2pi in the log. Sources are inconsistent
    g_mb = np.sqrt(2 * np.pi / zeta) + 2 * np.exp(-xi) * I_n(0, xi)
    g_mb = - g_mb * np.exp(-zeta) / 2
    fr = fr0 * (1 + D * g_tls + alpha * gamma * g_mb)
    return fr

def Q_vs_temp(temperature, fr0, D, alpha, Tc, A, B, m, n, delta_z,
                 gamma = 1, N0 = N0_Al):
    """
    Calculates the resonator quality factor at a given temperature, including
    the Mattis-Bardeen and TLS components of the temperature dependence

    Parameters:
    temperature (float or array-like): in K
    fr0 (float): frequency at 0 K
    D (float): fitting parameter. See TLS write-up for details
    alpha (float): kinetic inductance fraction
    Tc (float): superconducting transition temperature in K
    A (float): model parameter. Compared to Basu Thakur 2017, A -> P / A
    B (float): model parameter
    m (float): model power parameter
    n (float): model power parameter
    delta_z (float): 1 / Qr at T = 0
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits.
        This parameter should be enforced if fitting
    N0 (float): single-spin density of states at the Fermi Level.
        This parameter should be enforced if fitting

    Returns:
    Q (float or array-like): quality factor(s) at the given temperature(s)
    """
    kT = k_B * temperature
    Delta0 = 1.762 * k_B * Tc
    zeta = Delta0 / kT
    xi = h * fr0 / (2 * kT)

    num = D * np.tanh(xi)
    den0 = A * np.tanh(xi) ** 2 / (1 + B * np.tanh(xi) * temperature ** m)
    den = np.sqrt(1 + den0 ** n)
    delta_tls = num / den

    omega = 2 * np.pi * fr0
    # T << Tc approximation for the thermal quasiparticle density
    nth = 2 * N0 * np.sqrt(2 * np.pi * kT * Delta0) * np.exp(-Delta0 / kT)
    # T << Tc approximation of S1
    S1 = 2 / np.pi * np.sqrt(2 * Delta0 / (np.pi * kT)) * np.sinh(xi) * K_n(0, xi)
    delta_qp = alpha * gamma * S1 * nth / (2 * N0 * Delta0)

    return 1 / (delta_tls + delta_qp + delta_z)

################################################################################
######################### Funcs without TLS component ##########################
################################################################################
def fr_vs_temp_notls(temperature, fr0, alpha, Tc, gamma = 1):
    """
    Calculates the resonance frequency at the given temperature, including
    only the Mattis-Bardeen component of the temperature dependence. This model
    does not accurately extract the low-temperature dependence of the data if
    TLS behavior is present.

    Parameters:
    temperature (float or array-like): in K
    fr0 (float): frequency at 0 K
    alpha (float): kinetic inductance fraction
    Tc (float): superconducting transition temperature in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits.
        This parameter should be enforced if fitting

    Returns:
    fr (float or array-like): resonance frequency(ies) at the given
        temperature(s)
    """
    kT = k_B * temperature
    Delta0 = 1.762 * k_B * Tc
    zeta = Delta0 / kT
    xi = h * fr0 / (2 * kT)

    g_mb = np.sqrt(2 * np.pi / zeta) + 2 * np.exp(-xi) * I_n(0, xi)
    g_mb = - g_mb * np.exp(-zeta) / 2
    fr = fr0 * (1 + alpha * gamma * g_mb)
    return fr

def Q_vs_temp_notls(temperature, fr0, alpha, Tc, delta_z,
                 gamma = 1, N0 = N0_Al):
    """
    Calculates the quality factor at the given temperature, including
    only the Mattis-Bardeen component of the temperature dependence. This model
    does not accurately extract the low-temperature dependence of the data if
    TLS behavior is present.

    Parameters:
    temperature (float or array-like): in K
    fr0 (float): frequency at 0 K
    alpha (float): kinetic inductance fraction
    Tc (float): superconducting transition temperature in K
    delta_z (float): 1 / Qr at T = 0
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits.
        This parameter should be enforced if fitting
    N0 (float): single-spin density of states at the Fermi Level.
        This parameter should be enforced if fitting

    Returns:
    Q (float or array-like): quality factor(s) at the given temperature(s)
    """
    kT = k_B * temperature
    Delta0 = 1.762 * k_B * Tc
    zeta = Delta0 / kT
    xi = h * fr0 / (2 * kT)

    omega = 2 * np.pi * fr0
    # T << Tc approximation for the thermal quasiparticle density
    nth = 2 * N0 * np.sqrt(2 * np.pi * kT * Delta0) * np.exp(-Delta0 / kT)
    # T << Tc approximation of S1
    S1 = 2 / np.pi * np.sqrt(2 * Delta0 / (np.pi * kT)) * np.sinh(xi) * K_n(0, xi)
    delta_qp = alpha * gamma * S1 * nth / (2 * N0 * Delta0)
    return 1 / (delta_qp + delta_z)

################################################################################
######################## Funcs with only TLS component #########################
################################################################################
def fr_vs_temp_tls(temperature, fr0, D):
    """
    Calculates the resonance frequency at the given temperature, including
    only the TLS component of the temperature dependence. This model works at
    low temperatures where the Mattis-Bardeen component is negligible.

    Parameters:
    temperature (float or array-like): in K
    fr0 (float): frequency at 0 K
    D (float): fitting parameter. See TLS write-up for details

    Returns:
    fr (float or array-like): resonance frequency(ies) at the given
        temperature(s)
    """
    kT = k_B * temperature
    xi = h * fr0 / (2 * kT)

    g_tls = (np.real(digamma(1/2 + 1j * xi / np.pi)) - np.log(2 * xi)) / np.pi
    # There may be a factor of 1 / 2pi in the log. Sources are inconsistent
    fr = fr0 * (1 + D * g_tls)
    return fr

def Q_vs_temp_tls(temperature, fr0, D, A, B, m, n, delta_z):
    """
    Calculates the quality factor at the given temperature, including
    only the TLS component of the temperature dependence. This model works at
    low temperatures where the Mattis-Bardeen component is negligible.

    Parameters:
    temperature (float or array-like): in K
    fr0 (float): frequency at 0 K
    D (float): fitting parameter. See TLS write-up for details
    A (float): model parameter. Compared to Basu Thakur 2017, A -> P / A
    B (float): model parameter
    m (float): model power parameter
    n (float): model power parameter
    delta_z (float): 1 / Q at T = 0

    Returns:
    Q (float or array-like): quality factor(s) at the given temperature(s)
    """
    kT = k_B * temperature
    xi = h * fr0 / (2 * kT)

    num = D * np.tanh(xi)
    den0 = A * np.tanh(xi) ** 2 / (1 + B * np.tanh(xi) * temperature ** m)
    den = np.sqrt(1 + den0 ** n)
    delta_tls = num / den
    return 1 / (delta_tls + delta_z)

import numpy as np
from numba import jit

def responsivity(P, R_0, P_0):
    """
    Calculates the responsivity dx / dP of a KID

    Parameters:
    P (float or array-like): power (W)
    R_0 (float): responsivity at P = 0 (1 / W)
    P_0 (float): power at which the responsivity is rolled off by sqrt(2) from
        its value at P = 0

    Returns:
    (float or array-like): responsivity corresponding to the value(s) in P
    """
    if P_0 == 0:
        return np.nan
    return - R_0 * (1 + P / P_0) ** (-0.5)

@jit(nopython = True)
def responsivity_int(P, R_0, P_0, c):
    """
    Calculates the integrated form of the responsivity. This function describes
    the behavior of x versus P

    Parameters:
    P (float or array-like): power (W)
    R_0 (float): responsivity at P = 0 (1 / W)
    P_0 (float): power at which the responsivity is rolled off by sqrt(2) from
        its value at P = 0
    c (float): integration constant. If x is calibrated perfectly,
        c = - 2* R_0 * P_0. In practice, there may be a small offset

    Returns:
    (float or array-like): integrated responsivity corresponding to the value(s)
        in P
    """
    return 2 * R_0 * P_0 * (1 + P / P_0) ** (0.5) + c

@jit(nopython = True)
def responsivity_int_for_fitter(P, R_0, P_0, c):
    """
    Performs the same fucntion as responsivity_int, but with parameters rescaled
    for fitting.

    Parameter scaling factors:
    log(P) -> P
    1e-9 * R_0 -> R_0
    1e16 * P_0 -> P_0
    1e8 * c -> c

    Return scaling factors:
    x -> 1e6 * x
    """
    return responsivity_int(np.exp(P), R_0 * 1e9, P_0 * 1e-16, c * 1e-8) * 1e6

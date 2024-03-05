import numpy as np
from numba import jit, vectorize
from scipy import stats

def calc_qc_qi(qr, amp, phi):
    """
    Calculates Qc and Qi from Qr and amp, where amp = Qr / Qc and
    1 / Qr = 1 / Qc + 1 / Qi

    Parameters:
    qr (float): total quality factor
    amp (float): qr / ac
    phi (float): impedance mismatch angle

    Returns:
    qc (float): coupling quality factor
    qi (float): internal quality factor
    """
    qc = qr / amp
    qi = 1.0 / ((1.0 / qr) - (1.0 / qc))
    return qc, qi

def bounds_check(p0, bounds):
    """
    Checks that p0 is strictly within the bounds. If not, modifies bounds to be
    10% lower or higher than p0. I need to check that this is OK for parameters
    like phi that are cyclical.

    Parameters:
    p0(np.array): initial guesses for all parameters
    bounds (tuple): 2d tuple of low values bounds[0] the high values bounds[1]
        to bound the fitting problem

    Returns:
    new_bounds (tuple): modified bounds
    """
    lower_bounds = []
    upper_bounds = []
    for p, lb, ub in zip(p0, bounds[0], bounds[1]):
        if p <= lb:
            lower_bounds.append(p * 0.9)
        else:
            lower_bounds.append(lb)
        if p >= ub:
            upper_bounds.append(p * 1.1)
        else:
            upper_bounds.append(ub)
    return lower_bounds, upper_bounds

def calculate_residuals(z, z_fit):
    """
    Given IQ data and fitted IQ data, return the chi squared value and p value

    Parameters:
    z (np.array, complex): array of measured S21 data
    z_fit (np.array, complex): array of fitted S21 data

    Returns:
    chi_sq (float): chi squared value
    """
    z = np.hstack((np.real(z), np.imag(z)))
    z_fit = np.hstack((np.real(z_fit), np.imag(z_fit)))
    res = np.sqrt(sum((z - z_fit)**2)) / len(z)
    return res

@vectorize(nopython=True)
def cardan(a, b, c, d):
    """
    Analytical root finding.
    Based on code from https://github.com/Wheeler1711/submm_python_routines

    Parameters:
    a, b, c, d (float): polynomial coefficients

    Returns:
    root (float): largest real root
    """
    J = np.exp(2j * np.pi / 3)
    Jc = 1 / J
    u = np.empty(2, np.complex128)
    z0 = b / 3 / a
    a2, b2 = a * a, b * b
    p = -b2 / 3 / a2 + c / a
    q = (b / 27 * (2 * b2 / a2 - 9 * c / a) + d) / a
    D = -4 * p * p * p - 27 * q * q
    r = np.sqrt(-D / 27 + 0j)
    one_third = 1 / 3.0
    u = ((-q - r) / 2) ** one_third
    v = ((-q + r) / 2) ** one_third
    w = u * v
    w0 = np.abs(w + p / 3)
    w1 = np.abs(w * J + p / 3)
    w2 = np.abs(w * Jc + p / 3)
    if w0 < w1:
        if w2 < w0:
            v *= Jc
    elif w2 < w1:
        v *= Jc
    else:
        v *= J
    roots = np.asarray((u + v - z0, u * J + v * Jc - z0, u * Jc + v * J - z0))
    # print(roots)
    where_real = np.where(np.abs(np.imag(roots)) < 1e-15)
    # if len(where_real)>1: print(len(where_real))
    # print(D)
    if D > 0:
        return np.max(np.real(roots))  # three real roots
    else:
        # one real root get the value that has the smallest imaginary component
        return np.real(roots[np.argsort(np.abs(np.imag(roots)))][0])

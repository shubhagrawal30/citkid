import numpy as np
from numba import ji

@jit(nopython = True)
def nonlinear_iq_phase(f, i, q, ioff, qoff, fr, Qr, beta, theta0):
    """
    Returns the phase of a nonlinear_IQ resonator at the given frequency

    Parameters:
    f (float or np.array): frequency in Hz
    i (float or np.array): real part of complex S21 transmission
        corresponding to f
    q (float or np.array): imaginary part of complex S21 transmission
        corresponding to f
    ioff (float): real part of off-resonance complex S21 transmission
    qoff (float): imaginary part of off-resonance complex S21 transmission
    fr (float): resonance frequency in Hz
    Qr (float): total quality factor
    beta (float): dimensionless scaling factor
    theta0 (float): theta offset parameters

    Returns:
    theta (float or np.array): theta values
    """
    zabs = (i - ioff) ** 2 + (q - Qoff) ** 2
    xi = -2 * Qr * (f - fr + beta * zabs / fr)
    theta = theta0 + 2 * np.arctan(xi)
    return theta

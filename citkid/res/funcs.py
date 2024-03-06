import numpy as np
from numba import jit
from .util import cardan

@jit(nopython=True)
def nonlinear_iq(f, fr, Qr, amp, phi, a, i0, q0, tau):
    """
    Describes the transmission through a nonlinear resonator

                    (-j*2*pi*f*tau)    /                           (j phi)   \
        (i0+j*q0)*e^                * |1 -        Qr             e^           |
                                      |     --------------  X  ------------   |
                                       \     Qc * cos(phi)       (1+ 2jy)    /

        where the nonlinearity of y is described by
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr

    Parameters:
    f (np.array): array of frequencies in Hz
    fr (float): resonance frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl
    i0 (float): I gain factor
    q0 (float): Q gain factor
        i0 + j * q0 describes the overall constant gain and phase offset
    tau(float): cable delay in seconds

    Returns:
    z (np.array): array of complex IQ data corresponding to f
    """
    deltaf = f - fr
    yg = Qr * deltaf / fr
    y = get_y(yg, a)
    s21_readout = (i0 + 1.j * q0) * np.exp(-2.j * np.pi * deltaf * tau)
    s21_res = (1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y))
    z = s21_readout * s21_res
    return z

@jit(nopython=True)
def circle_objective(params, x, y):
    """
    Objective for circle fitting

    Parameters:
    params (A: float, B: float, R: float): circle fit parameters. (A, B) is the
        origin and R is the radius
    x (np.array): x data
    y (np.array): y data

    Returns:
    error (float): error for minimization
    """
    A, B, R = params
    error = sum(((x - A) ** 2+(y - B) ** 2 - R ** 2) ** 2)
    return error

################################################################################
######################### Utility functions ####################################
################################################################################

@jit(nopython=True)
def get_y(yg, a):
    """
    Calculates the largest real root of
        yg = y + a / (1 + y^2)

    Parameters:
    yg (float or np.array): unmodified resonance shift
        yg = Qr * (f - fr) / fr
    a (float): nonlinearity parameter

    Returns:
    y (float or np.array): largest real root of the above equation
    """
    y = cardan(4.0, -4.0 * yg, 1.0, -(yg + a))
    return y

################################################################################
######################## Equations for fitter ##################################
################################################################################

@jit(nopython=True)
def nonlinear_iq_for_fitter(f, fr, Qr, amp, phi, a, i0, q0, tau):
    """
    Same as nonlinear_iq, but returns stacked real and imaginary components
    for the fitter.
    """
    z = nonlinear_iq(f, fr, Qr, amp, phi, a, i0, q0, tau)
    return np.hstack((np.real(z), np.imag(z)))

import numpy as np
import citkid.res.fitter as fitter
from .util_old import get_peak_fwhm
from scipy.ndimage import gaussian_filter

def guess_p0_nonlinear_iq(f, z):
    """
    Get initial guess parameters for nonlinear_iq.
    Assumes gain and cable delay have been removed

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data

    Returns:
    p0 (list): initial guess fit parameters
        [fr, Qr, amp, phi, a, i0, q0, tau]
    """
    # guess i0 and q0
    z0_guess = np.mean(np.concatenate([z[:2], z[-2:]]))
    i0_guess = np.real(z0_guess)
    q0_guess = np.imag(z0_guess)
    # guess tau
    tau_guess = 0
    # guess phi and amp
    phi_guess, amp_guess = guess_phi_amp(z, z0_guess)
    # guess Qr
    Qr_guess = guess_Qr(f, z, z0_guess, phi_guess)
    # guess a
    a_guess = guess_a(f, z, Qr_guess)
    # guess fr
    fr_guess = guess_fr(f, z, z0_guess, phi_guess, a_guess, Qr_guess)
    # Package and return p0
    p0 = [fr_guess, Qr_guess, amp_guess, phi_guess, a_guess,
          i0_guess, q0_guess, tau_guess]
    return p0

def guess_Qr(f, z, z0, phi):
    """
    Guesses Qr given complex S21 data.

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance
    phi (float): impedance mismatch angle

    Returns:
    Qr_guess (float): Qr guess
    """
    # rotate data, then filter
    z_rot = abs((1 - z / z0) * np.exp(-1j * phi))
    z_rot = gaussian_filter(z_rot, sigma = 10)
    peak, fwhm = get_peak_fwhm(f, z_rot)
    poly = [-0.38741422, 15.38414659]
    Qr_guess = np.exp(np.polyval(poly, np.log(fwhm)))
    return Qr_guess

def guess_a(f, z, Qr):
    """
    Guesses the nonlinearity parameter

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    Qr (float): guess for the total quality factor

    Returns:
    a_guess (float): nonlinearity parameter guess
    """
    fdiff = np.mean([f[1:], f[:-1]], axis = 0)
    zdiff = abs(z[1:] - z[:-1])
    peak, width_spac = get_peak_fwhm(fdiff, zdiff)
    fr = fdiff[peak]

    x = np.log(width_spac / fr * Qr)
    if x < -3.43:
        a_guess = 1
    elif x > 0.17:
        a_guess = 1e-2
    else:
        poly = [-0.05591812, -0.27491759, -0.5516617, 0.10680838]
        a_guess = np.polyval(poly, x)
    return a_guess

def guess_fr(f, z, z0, phi, a, Qr):
    """
    Guess the resonance frequency

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance
    phi (float): impedance mismatch angle
    a (float): nonlinearity parameter guess
    Qr (float): guess for the total quality factor

    Returns:
    fr_guess (float): guess for the resonance frequency
    """
    z_rot = (1 - z / z0) * np.exp(-1j * phi)
    fdiff = np.mean([f[1:], f[:-1]], axis = 0)
    zdiff = abs(z_rot[1:] - z_rot[:-1])
    ix = np.argmax(zdiff)
    fr_guess = fdiff[ix]
    # Modification from nonlinearity parameter and Q
    poly = [-1.04607392,  0.9999992]
    fr_guess /= np.polyval(poly, a / Qr)
    return fr_guess

def guess_phi_amp(z, z0):
    """
    Guess the impedance mismatch rotation and amplitude.

    Parameters:
    z (np.array): array of complex S21 data
    z0 (complex): value of z off resonance

    Returns:
    phi_guess (float): phi guess
    amp_guess (float): amp guess
    """
    popt, _ = fitter.fit_iq_circle(z, plotq = False)
    xc, yc, R = popt
    # angle between center of circle and off resonance point
    x1, y1, = -np.real(z0), -np.imag(z0)
    x2, y2 = xc + x1, yc + y1
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    phi_guess = np.arctan2(det, dot)
    amp_guess = 2 * R * np.abs(np.cos(phi_guess)) / np.abs(z0)
    return phi_guess, amp_guess

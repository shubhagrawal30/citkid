import numpy as np
import citkid.res.fitter as fitter
from scipy import signal
from .util import cardan


def guess_p0_nonlinear_iq(f, z):
    """
    Get initial guess parameters for nonlinear_iq.
    Assumes gain and cable delay have been removed
    Modified from https://github.com/Wheeler1711/submm_python_routines

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    Returns:
    p0 (list):
    """
    f, z = f.copy(), z.copy()
    # i0 and q0 guess
    z0_guess = np.mean(np.concatenate([z[:2], z[-2:]]))
    i0_guess = np.real(z0_guess)
    q0_guess = np.imag(z0_guess)

    # Guess Qr
    Qr_guess, dist_index = guess_Qr(f, z, z0_guess)

    # Guess a
    a_guess = guess_a_nl(f, z, Qr_guess)

    # Guess phi and amp
    phi_guess, amp_guess = guess_phi_amp(z, z0_guess)

    # Guess fr
    fr_guess = np.mean(f)
    # Package and return p0
    p0 = [fr_guess, Qr_guess, amp_guess, phi_guess, a_guess, i0_guess, q0_guess]
    return p0

    # Next, with a good guess for fr, we can determine phi and amp really easily


    # Left off here. Qr and a guesses are really good
    #########################################################################################
    # Code from submm

    ##############################################################################################
    # guess f0 from largest spacing
    # largest spacing works better than distance from offres or min(S21)
    fdiff = np.mean([f[1:], f[:-1]], axis = 0)
    zdiff = abs(np.diff(z))
    peak, _ = signal.find_peaks(zdiff, height = (max(zdiff) + min(zdiff)) / 8)
    if not len(peak):
        peak = len(zdiff) // 2
        width  = len(zdiff) / 4 # I haven't checked this
    else:
        peak = peak[len(peak) // 2]
        width = signal.peak_widths(zdiff, [peak], rel_height = 0.5)[0][0]
    fr_guess = fdiff[peak]
    width_spac = np.median(fdiff[1:] - fdiff[:-1]) * width
    fr_index = peak

    # guess impedance rotation phi
    y_at_fr_dist = cardan(4.0, 0, 1.0, - a_guess)
    z_adj = (1 - z[dist_index] / z0_guess) * (1 + 2j * y_at_fr_dist)
    phi_guess = np.angle(z_adj)
    # phi_guess = 0


    # guess amp using polynomial fit
    dB = 20 * np.log10(np.abs(z / z0_guess))
    depth = np.max(dB) -  np.min(dB)
    if depth > 50:
        amp_guess = 0.9
    elif depth > 30:
        amp_guess = 0.5
    else:
        poly = [4.91802646e-05, -3.47894107e-03, 8.56340493e-02, 3.73634664e-03]
        amp_guess = np.polyval(poly, depth)



def guess_Qr(f, z, z0):
    """
    Guesses Qr given complex S21 data. Accurate to within 30%

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    z0 (complex): z value off resonance

    Returns:
    Qr_guess (float): guess for Qr
    dist_index (int): index at the point of furthest distance in the IQ plane
        from the off-resonance data
    """
    zdist = abs(z - z0)
    peak, _ = signal.find_peaks(zdist, height = (max(zdist) + min(zdist)) / 4)
    if not len(peak):
        peak = len(zdist) // 2
        width  = len(zdist) / 8 # Need to modify this later
    else:
        peak = peak[len(peak) // 2]
        width = signal.peak_widths(zdist, [peak], rel_height = 0.5)[0][0]
    dist_index = peak
    width_dist = np.median(f[1:] - f[:-1]) * width
    p = [ 0.99993354, -0.55088388]
    Qr_guess = 3.032441051857037 * np.exp(np.polyval(p, np.log(f[peak] / width_dist)))
    return Qr_guess, dist_index

def guess_a_nl(f, z, Qr_guess):
    """
    Guesses the nonlinearity parameter
    Accurate to within a factor of 2 for a = 0.1 to a = 2, can be up to a factor
    of 10 off at low a

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    Qr_guess (float): guess for the total quality factor

    Returns:
    a_guess (float): nonlinearity parameter guess
    """

    fdiff = np.mean([f[1:], f[:-1]], axis = 0)
    zdiff = abs(np.diff(z))
    peak, _ = signal.find_peaks(zdiff, height = (max(zdiff) + min(zdiff)) / 8)
    if not len(peak):
        peak = len(zdiff) // 2
        width  = len(zdiff) / 8 # Need to modify this later
    else:
        peak = peak[len(peak) // 2]
        width = signal.peak_widths(zdiff, [peak], rel_height = 0.5)[0][0]
    fr_guess = f[peak]
    width_spac = np.median(fdiff[1:] - fdiff[:-1]) * width

    x = np.log(width_spac / fr_guess * Qr_guess)
    if x < -3.43:
        a_guess = 1
    elif x > 0.3:
        a_guess = 1e-2
    else:
        poly = [-0.05591812, -0.27491759, -0.5516617, 0.10680838]
        a_guess = np.polyval(poly, x)
    return a_guess

def guess_phi_amp(z, z0):
    """
    Guess the impedance mismatch rotation and amplitude.

    Parameters:
    z (np.array): array of complex S21 data
    z0 (complex): value of z off resonance

    Returns:
    phi_guess (float): phi guess value
    amp_guess (float): amplitude guess value
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

def get_y_resonance(a):
    """
    Given a nonlinearity parameter a, returns the value of y at f = fr

    Parameters:
    a (float): nonlinearity parameter

    Returns:
    y (float): value of y at f = fr
    """
    y = cardan(4.0, 0, 1.0, a)
    return y

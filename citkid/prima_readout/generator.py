### Generate random resonance data
import numpy as np
from numba import jit, vectorize

@jit(nopython=True)
def generate_data(npoints_fine = 600, npoints_gain = 50):
    """
    Generates random resonance data. Assumes resonance frequencies were found
    from the point of max spacing. I can add other options if desired.

    Returns:
    ffine (np.array): fine-sweep frequency data in Hz
    zfine (np.array): fine-sweep complex S21 data
    fgain (np.array): gain-sweep frequency data in Hz
    zgain (np.array): gain-sweep complex S21 data
    """
    fr, Qr, amp, phi, a, p_amp, p_phase, noise_std, fine_bw =\
        generate_resonance_parameters()
    # Rough sweep
    f = np.linspace(fr - fine_bw / 2, fr + fine_bw / 2, 400)
    z = nonlinear_iq_simple(f, fr, Qr, amp, phi, a, p_amp, p_phase)
    # z += get_noise(f, noise_std * np.median(np.abs(z)))
    f0 = update_fr_spacing(f, z)
    # Fine sweep
    ffine = np.linspace(f0 - fine_bw / 2, f0 + fine_bw / 2, 600)
    zfine = nonlinear_iq_simple(ffine, fr, Qr, amp, phi, a, p_amp, p_phase)
    # zfine += get_noise(ffine, noise_std * np.median(np.abs(zfine)))
    # Gain sweep
    fgain = np.linspace(f0 - 10 * fine_bw / 2, f0 + 10 * fine_bw / 2, 50)
    zgain = nonlinear_iq_simple(fgain, fr, Qr, amp, phi, a, p_amp, p_phase)
    # zgain += get_noise(fgain, noise_std * np.median(np.abs(zfine)))
    return ffine, zfine, fgain, zgain

@jit(nopython=True)
def generate_resonance_parameters():
    """
    Generates random resonance model parameters

    Returns:
    f (array-like): array of frequencies in Hz
    fr (float): resonance frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl
    p_amp (array-like): gain polynomial coefficients
    p_phase (array-like): phase polynomial coefficients
    noise_std (float): z noise standard deviation
    fine_bw (float): fine-scan bandwidth in Hz
    """
    fr  = np.random.uniform(0.4e9, 2.4e9)
    Qr  = np.random.uniform(1e3, 500e3)
    amp = np.random.uniform(1e-5, 1 - 1e-5)
    phi = np.random.uniform(-np.pi / 2, np.pi / 2)
    a   = np.random.uniform(1e-5, 0.8)
    p_amp  = [np.random.uniform(-100, 100), np.random.uniform(-100, 100),
               np.random.uniform(-100, 100), np.random.uniform(-100, 100)]
    p_amp = [1]
    p_phase = [np.random.uniform(-1e-9, -1e-6), np.random.uniform(-1e3, 1e3)]
    noise_std = np.random.uniform(1e-3, 2)
    noise_nstd = 0
    fwhm = fr / Qr
    fine_bw = np.random.uniform(fwhm * 2, fwhm * 5)
    return fr, Qr, amp, phi, a, p_amp, p_phase, noise_std, fine_bw

@jit(nopython=True)
def get_noise(f, std_dev):
    """
    Generates z noise. It won't be Gaussian in real life but this is good
    enough for testing

    Parameters:
    f (np.array): frequency array
    std_dev (float): standard deviation of the noise

    Returns:
    z (np.array): random z noise
    """
    i = np.random.normal(0, std_dev, len(f))
    q = np.random.normal(0, std_dev, len(f))
    z = (i + 1j * q)
    return z
################################################################################
########################### Resonance model ####################################
################################################################################
@jit(nopython = True)
def nonlinear_iq_simple(f, fr, Qr, amp, phi, a, p_amp, p_phase):
    """
    Describes the transmission through a nonlinear resonator

                                    (j phi)
        1 -        Qr             e^
             --------------  X  ------------
              Qc * cos(phi)       (1+ 2jy)

        where the nonlinearity of y is described by
            yg = y+ a/(1+y^2)  where yg = Qr*xg and xg = (f-fr)/fr
    This equation is further multiplied by a real polynomial determined by the
    coefficients in p_amp and a phase shift polynomial determined by the
    coefficients in p_phase

    Parameters:
    f (array-like): array of frequencies in Hz
    fr (float): resonance frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter. Bifurcation occurs at
        a = 4 * sqrt(3) / 9 ~ 0.77.  Sometimes referred to as a_nl
    p_amp (array-like): gain polynomial coefficients
    p_phase (array-like): phase polynomial coefficients

    Returns:
    z (np.array): array of complex IQ data corresponding to f
    """
    deltaf = f - fr
    yg = Qr * deltaf / fr
    y = get_y(yg, a)
    s21_res = (1. - (amp / np.cos(phi)) * np.exp(1.j * phi) / (1. + 2.j * y))
    s21_readout = 10 ** (polyval(p_amp, f) / 20) + 0j
    s21_readout *= np.exp(1j * polyval(p_phase, f))
    z = s21_readout * s21_res
    return z

@jit(nopython = True)
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

@vectorize(nopython = True)
def cardan(a, b, c, d):
    """
    Analyticaly calculates the largest real root of a 3rd-order polynomial
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
    where_real = np.where(np.abs(np.imag(roots)) < 1e-15)
    if D > 0:
        # three real roots: return the max
        return np.max(np.real(roots))
    else:
        # one real root: return value with smalles imaginary component
        return np.real(roots[np.argsort(np.abs(np.imag(roots)))][0])

@jit(nopython=True)
def update_fr_spacing(f, z):
    """
    Give a single resonator rough sweep dataset, return the updated resonance
    frequency by finding the max spacing between adjacent IQ points

    Parameters:
    f (np.array): frequency data in Hz
    z (np.array): complex S21 data

    Returns:
    f0_new (float): updated frequency in Hz
    """
    spacing = np.abs(z[1:] - z[:-1])
    ix = np.argmax(spacing)
    if ix == 0:
        f0_new = f[0]
    elif ix == len(f):
        f0_new = f[-1]
    else:
        f0_new = (f[ix] + f[ix + 1]) / 2
    return f0_new

@jit(nopython=True)
def polyval(p, x):
    """
    Performs the same function as np.polyval, but with numba compatability
    """
    y = np.zeros_like(x)
    for i in range(len(p)):
        y = x * y + p[i]
    return y

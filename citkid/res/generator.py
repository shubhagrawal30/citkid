from .funcs import nonlinear_iq
from numba import jit
import numpy as np
'''Code to generate random resonances for simulations'''

# @jit(nopython=True)
def get_random_dataset(normalize = False):
    """
    Gets a dataset with random system parameters and random resonators

    Parameters:
    normalize (bool): If True, normalizes z by its mean

    Returns:
    f (np.array): frequency array
    z (np.array): complex S21 array
    resonances (list): values are resonator parameter lists
        [fr, Qr, amp, phi, a]
    """
    # Speed this up by using frequency bin for each resonator instead of f
    npoints, frange, resonances, cable_delay, a0, f0, sin_parameters, noise_std \
            = get_random_system()
    f = np.linspace(frange[0], frange[1], npoints)
    z = get_system_z(f, cable_delay, a0, f0, sin_parameters, noise_std)
    for p in resonances:
        span = (p[0] / p[1]) * 20
        ix = np.where(np.abs(f - p[0]) < span)[0]
        i0, i1 = min(ix), max(ix)
        zmid = z[i0:i1] * get_resonance_s21(f[i0:i1], *p)
        z = np.concatenate((z[:i0], zmid, z[i1:]))
    fres = np.array([p[0] for p in resonances])
    if normalize:
        z /= np.mean(np.abs(z))
    return f, z, resonances

@jit(nopython=True)
def get_resonance_s21(f, fr, Qr, amp, phi, a):
    """
    Gets S21 of a resonance, with no added gain or phase terms.

    Parameters:
    f (np.array): array of frequencies in Hz
    fr (float): resonance frequency in Hz
    Qr (float): total quality factor
    amp (float): Qr / Qc, where Qc is the coupling quality factor
    phi (float): rotation parameter for impedance mismatch between KID and
        readout circuit
    a (float): nonlinearity parameter.

    Returns:
    z (np.array): complex S21 data
    """
    return nonlinear_iq(f, fr, Qr, amp, phi, a, 1, 0, 0)

@jit(nopython=True)
def decay(f, a0, f0):
    """
    Generate an exponential decay

    Parameters:
    f (np.array): array of frequencies
    a0 (np.array): exponential amplitude
    f0 (np.array): exponent

    Returns:
    z (np.array): exponential decay value corresponding to f
    """
    return a0 * np.exp(-f / f0) + 0j

@jit(nopython=True)
def get_noise(f, std_dev):
    """
    Generates z noise

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

@jit(nopython=True)
def get_system_z(f, cable_delay, a0, f0, sin_parameters, noise_std):
    """
    Generates a system S21 with an exponential decay, cable delay, and
    sinusoidal ripples.

    Parameters:
    f (np.array): frequency array
    cable_delay (float): in seconds
    a0 (float): exponential decay amplitude
    f0 (float): exponential decay exponent
    sin_parameters (list): list of sine parameters.
    noise_std (float): noise standard deviation

    Returns:
    z (np.array): system S21
    """
    z = decay(f, a0, f0)
    z *= np.exp(-2.0j * np.pi * (f - np.mean(f)) * cable_delay)
    z += get_noise(f, noise_std)
    for s in sin_parameters:
        z *= s[0] * 10 ** (1 + np.sin(2 * np.pi * f / s[1]))
    return z

@jit(nopython=True)
def create_random_resonance(fr, Qrrange, amprange, phirange, arange):
    """
    Generates a random resonance, given a range for each parameter

    Parameters:
    fr (float): resonance frequency
    Qrrange (tuple): lower and upper limit on total quality factor
    amprange (tuple): lower and upper limit on amp = Qr / Qc
    phirange (tuple): lower and upper limit on phi
    arange (tuple): lower and upper limit on a

    Returns:
    p (np.array): resonance parameters
    """
    p = np.array([fr])
    p = np.append(p, np.random.uniform(Qrrange[0], Qrrange[1]))
    p = np.append(p, np.random.uniform(amprange[0], amprange[1]))
    p = np.append(p, np.random.uniform(phirange[0], phirange[1]))
    p = np.append(p, np.random.uniform(arange[0], arange[1]))
    return p

@jit(nopython=True)
def create_random_resonances(nres, frrange, Qrrange, amprange, phirange, arange):
    """
    Generates a list of random resonances, given a range for each parameter

    Parameters:
    nres (int): number of resonators
    frrange (tuple): lower and upper limits on resonance frequency. Resonance
        frequencies with be roughly logarithmically spaced, but with noise
    Qrrange (tuple): lower and upper limit on total quality factor
    amprange (tuple): lower and upper limit on amp = Qr / Qc
    phirange (tuple): lower and upper limit on phi
    arange (tuple): lower and upper limit on a

    Returns:
    ps (list): values are output of create_random_resonance
    """
    frs = np.logspace(np.log10(frrange[0]), np.log10(frrange[1]), nres)
    dist = (frrange[1] - frrange[0]) / nres * 1.1
    frs = frs + np.random.uniform(-dist, dist, len(frs))
    ps = [create_random_resonance(fr, Qrrange, amprange, phirange, arange) for fr in frs]
    return ps

@jit(nopython=True)
def get_ranges(frange):
    """
    Gets resonance parameter ranges that are reasonable for typical KIDs

    Parameters:
    frange (tuple): lower and upper limit on resonance frequencies

    Returns:
    frrange (tuple): lower and upper limits on resonance frequency deviation
        from logarithmic spacing
    Qrrange (tuple): lower and upper limit on total quality factor
    amprange (tuple): lower and upper limit on amp = Qr / Qc
    phirange (tuple): lower and upper limit on phi
    arange (tuple): lower and upper limit on a
    """
    frrange = np.array((frange[0] * 1.05, frange[1] * 0.95)) # fr
    Qrrange = np.array((1e3, 200e3)) # Qr
    amprange = np.array((1e-3, 1 - 1e-5)) # amp
    phirange = np.array((-np.pi / 2, np.pi / 2)) # phi
    arange = np.array((0.01, 0.8)) #a
    return frrange, Qrrange, amprange, phirange, arange

@jit(nopython=True)
def get_random_system_parameters():
    """
    Gets random gain parameters for a typical system

    Returns:
    a0 (float): exponential decay amplitude
    noise_std (float): noise standard deviation
    f0 (float): exponential decay exponent
    cable_delay (float): in seconds
    sin_parameters (list): list of sine parameters
    """
    a0_range = np.array([10 ** (-100 / 20), 10 ** (100 / 20)])
    noise_factor_range = np.array([2.2, 3])
    noise_factor_range = np.array([1.5, 2.5])
    f0_range = np.array([50e6, 1e9])
    cable_delay_range = np.array([1e-9, 500e-9])
    sin_range = np.array([30e6, 1e9])

    a0 = np.random.uniform(a0_range[0], a0_range[1])
    noise_factor = np.random.uniform(noise_factor_range[0], noise_factor_range[1])
    noise_std = a0 / np.power(10, noise_factor)
    f0 = np.random.uniform(f0_range[0], f0_range[1])
    cable_delay = np.random.uniform(cable_delay_range[0], cable_delay_range[1])

    n_sin_params = np.random.uniform(0, 1)
    if n_sin_params < 0.95:
        sin_parameters = np.array([[1, np.random.uniform(sin_range[0], sin_range[1])]])
    elif n_sin_params < 0.99:
        sin_parameters = np.array([[1, np.random.uniform(sin_range[0], sin_range[1])], [1, np.random.uniform(sin_range[0], sin_range[1])]])
    else:
        sin_parameters = np.array([[1, np.random.uniform(sin_range[0], sin_range[1])], [1, np.random.uniform(sin_range[0], sin_range[1])],
                                   [1, np.random.uniform(sin_range[0], sin_range[1])]])
    return a0, noise_std, f0, cable_delay, sin_parameters

@jit(nopython=True)
def get_random_frange():
    """
    Gets a random number of resonators and frequency range

    Returns:
    nres (int): number of resonators
    frange (tuple): lower and upper frequency range
    """
    nres = int(np.random.uniform(1, 1200))
    frange_start = np.random.uniform(1e7, 1e9)
    frange_end = np.random.uniform(frange_start + nres * 0.5e6, 3e9)
    frange = np.array([frange_start, frange_end])
    frange = np.array([400e6, 2400e6])
    return nres, frange

@jit(nopython=True)
def get_random_resonances():
    """
    Gets random resonances, with typical range parameters

    Returns:
    resonances (list): values are resonator parameter lists
    frange (tuple): lower and upper frequency range
    """
    nres, frange = get_random_frange()
    frrange, Qrrange, amprange, phirange, arange = get_ranges(frange)
    resonances = create_random_resonances(nres, frrange, Qrrange, amprange, phirange, arange)
    return resonances, frange

@jit(nopython=True)
def get_random_system():
    """
    Gets random system parameters

    Returns:
    npoints (int): number of points in the data set
    frange (tuple): lower and upper frequency range
    resonances (list): values are resonator parameter lists
    cable_delay (float): in seconds
    a0 (float): exponential decay amplitude
    f0 (float): exponential decay exponent
    sin_parameters (list): list of sine parameters
    noise_std (float): noise standard deviation
    """
    a0, noise_std, f0, cable_delay, sin_parameters = get_random_system_parameters()
    resonances, frange = get_random_resonances()
    npoints = int(np.random.uniform(1000, 100000))
    return npoints, frange, resonances, cable_delay, a0, f0, sin_parameters, noise_std

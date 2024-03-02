import numpy as np

def guess_p0_nonlinear_iq(f, z):
    """
    Get initial guess parameters for nonlinear_iq.
    Modified from https://github.com/Wheeler1711/submm_python_routines
    More work can be done here

    Parameters:
    f (np.array): array of frequency data in Hz
    z (np.array): array of complex S21 data
    Returns:
    p0 (list):
    """
    # guess f0, furthest distance from offres in IQ space
    offres_z = np.mean(np.roll(z, 10)[:20])
    distance = abs(z - offres_z)
    fr_guess_index = np.argmax(distance)
    fr_guess = f[fr_guess_index]

    # guess Q by finding the FWHM around fr_guess
    # Subtract offres dB, then take absolute value to get width
    # This still doesn't work well for low-Q high phi resonators. There is
    # room to improve.
    # Try fitting to spacing difference.
    dB = 20 * np.log10(abs(z))
    dB -= np.mean(np.roll(dB, 10)[:20])
    dB = abs(dB)
    i_cent = np.argmax(dB)
    dBmax = dB[i_cent]
    diff = abs(dB - dBmax + 3)
    ix0 = np.argmin(diff[:i_cent])
    ix1 = np.argmin(diff[i_cent:]) + i_cent
    Q_guess = fr_guess / (f[ix1] - f[ix0])

    # guess amp
    dB = 20 * np.log10(np.abs(z))
    depth = np.max(dB) - np.min(dB)
    if depth > 50:
        amp_guess = 0.9
    elif depth > 30:
        amp_guess = 0.8
    else:
        # polynomial fit to amp versus depth.
        poly = [0.03130349, 0.]
        amp_guess = np.polyval(poly, depth)

    # guess impedance rotation phi
    # You can't do much better than this, because a lot of the observable
    # parameters are cyclical in pi / 2 intervals of phi, and depend heavily
    # on other parameters
    phi_guess = 0

    # guess non-linearity parameter
    diff = abs(z[1:] - z[:-1])
    spacing = np.log(np.max(diff) / np.min(diff))
    if spacing < 4:
        a_guess = 0.01
    elif spacing > 10:
        a_guess = 1
    else:
        a_guess = np.polyval([ 0.21535156, -0.51811171], spacing)

    # i0 and q0 guess
    i0_guess = (np.real(z[0]) + np.real(z[-1])) / 2.
    q0_guess = (np.imag(z[0]) + np.imag(z[-1])) / 2.

    # cable delay guess tau
    tau_guess = 0

    p0 = [fr_guess, Q_guess, amp_guess, phi_guess, a_guess, i0_guess, q0_guess,
          tau_guess]
    return p0

import numpy as np
# import citkid.res.fitter
from scipy import signal

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
    f, z = f.copy(), z.copy()
    # i0 and q0 guess
    z0_guess = 1. + 0.j
    i0_guess = np.real(z0_guess)
    q0_guess = np.imag(z0_guess)

    # cable delay guess tau
    tau_guess = 0

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

    # guess Q by finding the FWHM around fr_guess
    # Use the width of the distance of z from the off resonance data
    # This works better than S21 or spacing
    zdist = abs(z - np.mean(np.concatenate([z[:5], z[-5:]])))
    peak, _ = signal.find_peaks(zdist, height = (max(zdist) + min(zdist)) / 4)
    if not len(peak):
        peak = len(zdist) // 2
        width  = len(zdist) / 8 # I haven't checked this
    else:
        peak = peak[len(peak) // 2]
        width = signal.peak_widths(zdist, [peak], rel_height = 0.5)[0][0]
    width_dist = np.median(f[1:] - f[:-1]) * width
    p = [0.95155596, 1.03605832]
    Qr_guess = np.exp(np.polyval(p, np.log(fr_guess / width_dist)))

    # guess impedance rotation phi
    # This guess is only useful if the curve is rotated before fitting, because
    # around phi = pi or phi = -pi, the fit will jump back and forth between the
    # bounds
    diff = abs(f - fr_guess)
    ix = np.argsort(diff)[:2]
    phi_guess = np.angle(1 - np.mean(z[ix]))
    phi_guess -= 2 * np.pi * (phi_guess // (2 * np.pi))

    # guess amp
    # For amp, remove phase first
    amp_guess = np.max(np.real(z)) - np.min(np.real(z))
    d = np.max(20 * np.log10(np.abs(z))) - np.min(20 * np.log10(np.abs(z)))
    if d>30:
        amp_guess = 0.99
    else:
        amp_guess = 0.0037848547850284574 + 0.11096782437821565 * d - 0.0055208783469291173 * d ** 2 + 0.00013900471000261687 * d ** 3 + -1.3994861426891861e-06 * d ** 4  # polynomial fit to amp verus depth

    # guess non-linearity parameter
    diff = abs(z[1:] - z[:-1])
    spacing = np.log(np.max(diff) / np.min(diff))
    if spacing < 4:
        a_guess = 0.01
    elif spacing > 10:
        a_guess = 1
    else:
        a_guess = np.polyval([ 0.21535156, -0.51811171], spacing)
    # a_guess = 0.1



    p0 = [fr_guess, Qr_guess, amp_guess, phi_guess, a_guess, i0_guess, q0_guess,
          tau_guess]
    return p0

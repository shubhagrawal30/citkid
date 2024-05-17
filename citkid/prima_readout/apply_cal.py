import numpy as np
from ..res.gain import remove_gain

def calculate_theta(f, z, p_amp, p_phase, origin, x):
    """
    Calculates theta from complex z data and calibration parameters

    Parameters:
    f (float or array-like): frequency data corresponding to z, or tone
        frequency in Hz
    z (array-like): complex S21 data
    origin (complex): origin of the resonance circle
    x (complex): vector from the origin of the resonance circle to
    the on-resonance point

    Returns:
    theta (np.array): theta array corresponding to the values in z
    """
    z_rmvd = remove_gain(f, z, p_amp, p_phase)
    z_shifted = z_rmvd - origin
    theta = np.array([np.angle(np.vdot(x, zi)) for zi in z_shifted])
    return theta

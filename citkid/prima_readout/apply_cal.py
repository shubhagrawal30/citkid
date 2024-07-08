import numpy as np
from ..res.gain import remove_gain

def convert_iq_to_x(f, z, p_amp, p_phase, origin, v, p_x):
    """
    Calculates theta from complex z data and calibration parameters

    Parameters:
    f (float or array-like): frequency data corresponding to z, or tone
        frequency in Hz
    z (array-like): complex S21 data
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the resonance circle
    v (complex): vector from the origin of the resonance circle to
    the on-resonance point
    p_x (array-like): polynomial fit parameters to x versus theta

    Returns:
    x (np.array): x = df / f array corresponding to the values in z
    """
    theta = calculate_theta(f, z, p_amp, p_phase, origin, v)
    x = 1 - np.polyval(p_x, theta) / f
    return x

def calculate_theta(f, z, p_amp, p_phase, origin, v):
    """
    Calculates theta from complex z data and calibration parameters

    Parameters:
    f (float or array-like): frequency data corresponding to z, or tone
        frequency in Hz
    z (array-like): complex S21 data
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the resonance circle
    v (complex): vector from the origin of the resonance circle to
    the on-resonance point

    Returns:
    theta (np.array): theta array corresponding to the values in z
    """
    z_rmvd = remove_gain(f, z, p_amp, p_phase)
    z_shifted = z_rmvd - origin
    theta = np.array([np.angle(np.vdot(v, zi)) for zi in z_shifted])
    return theta

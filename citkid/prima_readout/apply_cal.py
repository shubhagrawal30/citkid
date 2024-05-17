import numpy as np

def calculate_theta(z, origin, x):
    """
    Calculates theta from complex z data and calibration parameters

    Parameters:
    z (array-like):
    origin (complex): origin of the resonance circle
    x (complex): vector from the origin of the resonance circle to
    the on-resonance point

    Returns:
    theta (np.array): theta array corresponding to the values in z
    """
    z = np.asarray(z)
    z_shifted = z - origin
    theta = np.array([np.angle(np.vdot(x, zi)) for zi in z_shifted])
    return theta

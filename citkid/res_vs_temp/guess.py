import numpy as np

def guess_p0_fr_vs_temp(temperature, fr, Tc_guess = 1.3):
    """
    Calculates an initial guess for fr_vs_temp. Tc_guess must be provided

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    Tc_guess (float): critical temperature guess in K

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, D_guess, alpha_guess, Tc_guess]
    """
    i = np.argmax(fr)
    ix_low = i - 2
    if ix_low < 4:
        ix_low = 4
    tlow, flow = temperature[0:ix_low], fr[0:ix_low]
    plow = np.polyfit(tlow, flow, 1)

    # D guess
    poly = [5.61657789e-11, 4.93808879e-4]
    D_guess = np.polyval(poly, plow[0])
    # f0 guess
    fr0_guess = max(fr)
    # Alpha guess
    alpha_guess = 0.7
    # p0
    p0 = [fr0_guess, D_guess, alpha_guess, Tc_guess]
    # bounds
    bounds = get_bounds_fr_vs_temp(p0)
    return p0, bounds 

def get_bounds_fr_vs_temp(p0):
    """
    Gets bounds for the fitter for fr_vs_temp

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, D_guess, alpha_guess, Tc_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 10, p0[2] / 3, p0[3] / 2],
              [p0[0] * (1 + 1e-5), p0[1] * 10, p0[2] * 3, p0[3] * 2]]
    return bounds

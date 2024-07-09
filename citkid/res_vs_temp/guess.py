import numpy as np

def guess_p0_fr_vs_temp(temperature, fr, Tc_guess = 1.3, gamma = 1):
    """
    Calculates an initial guess for fr_vs_temp. Tc_guess must be provided

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    Tc_guess (float): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, D_guess, alpha_guess, Tc_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    i = np.argmax(fr)
    ix_low = i - 2
    if ix_low < 4:
        ix_low = 4
    tlow, flow = temperature[0:ix_low], fr[0:ix_low]
    plow = np.polyfit(tlow, flow, 1)

    # D guess
    poly = [5.03204303e-11, 4.71528691e-04]
    D_guess = np.polyval(poly, plow[0])
    # f0 guess
    fr0_guess = max(fr)
    # Alpha guess
    alpha_guess = 0.7 / gamma
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

################################################################################
######################### Funcs without TLS component ##########################
################################################################################
def guess_p0_fr_vs_temp_notls(temperature, fr, Tc_guess = 1.3, gamma = 1):
    """
    Calculates an initial guess for fr_vs_temp_notls. Tc_guess must be provided

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    Tc_guess (float): critical temperature guess in K
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    # f0 guess
    fr0_guess = max(fr)
    # Alpha guess
    alpha_guess = 0.7 / gamma
    # p0
    p0 = [fr0_guess, alpha_guess, Tc_guess]
    # bounds
    bounds = get_bounds_fr_vs_temp_notls(p0)
    return p0, bounds

def get_bounds_fr_vs_temp_notls(p0):
    """
    Gets bounds for the fitter for fr_vs_temp_notls

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3, p0[2] / 2],
              [p0[0] * (1 + 1e-5), p0[1] * 3, p0[2] * 2]]
    return bounds

def guess_p0_Q_vs_temp_notls(temperature, Q, fr0_guess, gamma, N0,
                             alpha_guess = 0.7, Tc_guess = None):
    """
    Calculates an initial guess for Q_vs_temp_notls
    I haven't tested this for N0 different from N0_Al

    Parameters:
    temperature (array-like): temperature data in K
    Q (array-like): quality factor data
    fr0_guess (float): guess for fr0 in Hz
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    N0 (float): single-spin density of states at the Fermi Level
    alpha_guess (float): kinetic inductance fraction guess
    Tc_guess (float or None): critical temperature guess in K, or None to guess
        using the data

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess, delta_z_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    # delta_z guess
    delta_z_guess = 1 / max(Q)
    # alpha guess
    alpha_guess = 0.7 / gamma
    # Tc guess
    if Tc_guess is None:
        ix = np.argmax(Q)
        Q1 = Q[ix:]
        Tc0 = np.where(Q1 < Q[ix] / 2)[0]
        if len(Tc0):
            Tc0 = temperature[Tc0[0]]
        else:
            Tc0 = temperature[-1]
        poly = [ 0.03088547, -2.58625446,  9.9405062 , -0.46576281]
        Tc_guess = np.polyval(poly, Tc0)
    # Create p0
    p0 = [fr0_guess, alpha_guess, Tc_guess, delta_z_guess]
    bounds = get_bounds_Q_vs_temp_notls(p0)
    return p0, bounds

def get_bounds_Q_vs_temp_notls(p0):
    """
    Gets bounds for the fitter for Q_vs_temp_notls

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, alpha_guess, Tc_guess, delta_z_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 3, p0[2] / 2, p0[3] / 10],
              [p0[0] * (1 + 1e-5), p0[1] * 3, p0[2] * 2, p0[3] * 10]]
    return bounds


################################################################################
######################## Funcs with only TLS component #########################
################################################################################
def guess_p0_fr_vs_temp_tls(temperature, fr):
    """
    Calculates an initial guess for fr_vs_temp_tls

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz

    Returns:
    p0 (list): initial guess parameters
        [fr0_guess, D_guess]
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    i = np.argmax(fr)
    ix_low = i - 2
    if ix_low < 4:
        ix_low = 4
    tlow, flow = temperature[0:ix_low], fr[0:ix_low]
    plow = np.polyfit(tlow, flow, 1)

    # D guess
    poly = [5.03204303e-11, 4.71528691e-04]
    D_guess = np.polyval(poly, plow[0])
    # f0 guess
    fr0_guess = max(fr)

    p0 = [fr0_guess, D_guess]
    # bounds
    bounds = get_bounds_fr_vs_temp_tls(p0)
    return p0, bounds

def get_bounds_fr_vs_temp_tls(p0):
    """
    Gets bounds for the fitter for fr_vs_temp_tls

    Parameters:
    p0 (list): initial guess parameters
        [fr0_guess, D_guess]

    Returns:
    bounds (list): [lower_bounds, upper_bounds] for fitter corresponding to p0
    """
    bounds = [[p0[0] * (1 - 1e-5), p0[1] / 10],
              [p0[0] * (1 + 1e-5), p0[1] * 10]]
    return bounds

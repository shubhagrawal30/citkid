import numpy as np
from scipy.optimize import curve_fit
from .funcs import responsivity_int_for_fitter
from .guess import guess_p0_responsivity_int, get_bounds_responsivity_int
from .plot import plot_responsivity_int
from .data_io import make_fit_row

def fit_responsivity_int(power, x, f1, guess = None, guess_nfit = 3,
                         return_dataframe = False, plotq = False):
    """
    Fits x versus power data to the integrated responsivity equation.
    The fitter works best if there are at least three data points at P >> P_0.
    Otherwise, an alternative initial guess may be required.

    Parameters:
    power (array-like): array of blackbody powers in W
    x (array-like): array of fractional frequency shifts in Hz / Hz. This must
        be scaled close to x(P = 0) = 0 for the initial guess to work well
    f1 (float): frequency at P = 0 that was used to calculate x
    guess (list or None): If not None, guess is used as the initial guess
        [R0, P0, c]
    guess_nfit (int): number of high-power (P >> P_0) points in the data for the
        initial guess
    return_dataframe (bool): If True, returns a pandas series of the output data
        instead of individual parameters
    plotq (bool): If True, plots the fit and initial guess

    Returns:
    p0 (list): initial guess parameters [R0_guess, P0_guess, c_guess]
    popt (list): fit parameters [R0, P0, c]
    perr (list): fit parameter uncertainties [R0_err, P0_err, c_err]
    f0 (float): frequency at P = 0, determined by the fit
    f0err (float): uncertainty in f0
    (fig, ax): pyplot figure and axis, or (None, None) if not plotq
    """
    ix = np.argsort(power)
    power, x = power[ix], x[ix]
    # Initial guess
    if guess is not None:
        p0 = guess
        bounds = get_bounds_responsivity_int(p0)
    else:
        p0, bounds = guess_p0_responsivity_int(power, x, guess_nfit = guess_nfit)
    # Fit
    try:
        popt, pcov = curve_fit(responsivity_int_for_fitter, np.log(power),
                               x * 1e6, p0 = p0, bounds = bounds)
        perr = np.sqrt(np.diag(pcov))
        p0[0], popt[0], perr[0] = p0[0] * 1e9  , popt[0] * 1e9  , perr[0] * 1e9
        p0[1], popt[1], perr[1] = p0[1] * 1e-16, popt[1] * 1e-16, perr[1] * 1e-16
        # p0[2], popt[2], perr[2] = p0[2] * 1e-8 , popt[2] * 1e-8 , perr[2] * 1e-8
    except:
        popt = [np.nan, np.nan, np.nan]
        perr = [np.nan, np.nan, np.nan]
    # Plot
    if plotq:
        fig, ax = plot_responsivity_int(power, x, popt, p0)
    else:
        fig, ax = None, None
    # Determine f0
    f0 = f1 * popt[2]
    f0err = f1 * perr[2]
    if return_dataframe:
        row = make_fit_row(p0, popt, perr, f1, f0, f0err)
        return row, (fig, ax)
    return p0, popt, perr, f0, f0err, (fig, ax)

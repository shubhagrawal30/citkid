import numpy as np
from scipy import optimize
from .funcs import nonlinear_iq_for_fitter, nonlinear_iq, circle_objective
from .util import bounds_check, calculate_residuals
from .gain import fit_and_remove_gain_phase
from .plot import plot_nonlinear_iq, plot_gain_fit, plot_circle
from .plot import  combine_figures_vertically
import citkid.res.guess as guess

def fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine, frs, Qrs,
                               plotq = False, **kwargs):
    """
    Fits IQ data with gain amplitudes and phase correction from a gain scan.
    Cuts resonance frequencies from the gain scan in spans of fr / Qr around fr,
    where fr is an item in frs and Qr is a corresponding quality factor in Qrs

    The optimal fine scan width is 6 * fr / Qr
    The optimal gain scan width is 100 * fr / Qr

    Parameters:
    fgain (np.array): gain sweep frequency data
    zgain (np.array): gain sweep complex S21 data
    ffine (np.array): fine sweep frequency data
    zfine (np.array): fine sweep complex S21 data
    frs (list of float): resonance frequencies to cut from the gain scan
    Qrs (list of float): Qrs to cut from the gain scan.
    plotq (bool): If True, plots the fits.
    **kwargs: other arguments for fit_nonlinear_iq

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB
    p_phase (np.array): 1st-order polynomial fit parameters to phase
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter
    popt_err (np.array): standard errors on fit parameters
    res (float): fit residuals
    fig (pyplot.figure or None): figure with gain fit and nonlinear IQ fit if
        plotq, or None
    """

    p_amp, p_phase, zfine_rmvd, (fig_gain, axs_gain) = \
        fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, frs, Qrs,
                                  plotq = plotq)
    p0, popt, popt_err, res, (fig_fit, axs_fit) = fit_nonlinear_iq(ffine,
                                            zfine_rmvd, plotq = plotq, **kwargs)
    if plotq:
        fig = combine_figures_vertically(fig_gain, fig_fit)
    else:
        fig = None
    return p_amp, p_phase, p0, popt, popt_err, res, fig

def fit_nonlinear_iq(f, z, bounds = None, p0 = None, fr_guess = None,
                     fit_tau = True, tau_guess = None, plotq = False):
    """
    Fit a nonlinear IQ with from an S21 sweep. Uses scipy.optimize.curve_fit.
    It is assumed that the system gain and phase are removed from the data
    before fitting. i0, q0, and tau are fitted only for fine-tuning.

    The optimal span of the data is 6 * fr / Qr
    The optimal length of the data is 500, but down to 200 still works ok

    Parameters:
    f : numpy.array
        frequencies Hz
    z : numpy.array
        complex s21
    bounds (tuple or None): 2d tuple of low values bounds[0] the high values
        bounds[1] to bound the fitting problem. If None, sets default bounds
    p0 (list or None): initial guesses for all parameters
        fr_guess  = p0[0]
        Qr_guess  = p0[1]
        amp_guess = p0[2]
        phi_guess = p0[3]
        a_guess   = p0[4]
        i0_guess  = p0[5]
        q0_guess  = p0[6]
        tau_guess = p0[7]
        If None, calls citkid.fit.guess.guess_nonlinear_iq to find p0.
    fr_guess (float or None): if float, overrides p0[0].
    fit_tau (bool): if False, tau is enforced from p0[7] to speed up fitting.
        If True, tau is fit.
    tau_guess (float or None): If float, overides p0[7]
    plotq (bool): if True, plots the data with the fit

    Returns:
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter
    popt_err (np.array): standard errors on fit parameters
    res (float): fit residuals
    fig, ax (pyplot figure and axes, or None): plot of data with fit if plotq,
        or None, None
    """
    # Sort f and z
    f, z = np.array(f), np.array(z)
    ix = np.argsort(f)
    f, z = f[ix], z[ix]
    if p0 is None: # default initial guess
        p0 = guess.guess_p0_nonlinear_iq(f, z)
    if bounds is None:
        # default bounds. Phi range is increased to avoid jumping at bounds
        #                 fr,  Qr, amp,    phi,    a,   i0,   q0,     tau
        bounds = ([np.min(f), 1e3, .01, -np.pi * 1.5, 0, -1e2, -1e2, -1.0e-6],
                  [np.max(f), 1e7,   1,  np.pi * 1.5, 2,  1e2,  1e2,  1.0e-6])
    for index in [1, 5, 6]: # For Qr and z0, the initial guess should be good
        # These will be flipped in bounds_check if needed
        bounds[0][index] = p0[index] / 10
        bounds[1][index] = p0[index] * 10
    if fr_guess is not None:
        p0[0] = fr_guess
    if tau_guess is not None:
        p0[7] = tau_gues
    # Stack z data
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # Check bounds
    bounds = bounds_check(p0, bounds)
    # fit
    if not fit_tau:
        # Fit with tau enforced from p0[7]
        tau = p0[7]
        del bounds[0][7]
        del bounds[1][7]
        del p0[7]
        def fit_func(x_lamb, a, b, c, d, e, f, g):
            return nonlinear_iq_for_fitter(x_lamb, a, b, c, d, e, f, g, tau)
        popt, pcov = optimize.curve_fit(fit_func, f, z_stacked, p0,
                                        bounds = bounds)
        popt = np.insert(popt, 7, tau)
        popt_err = np.sqrt(np.diag(pcov))
        popt_err = np.insert(popt_err, 7, 0)
    else:
        # Fit without enforcing tau
        popt, pcov = optimize.curve_fit(nonlinear_iq_for_fitter, f, z_stacked,
                              p0, bounds = bounds)

        popt_err = np.sqrt(np.diag(pcov))
    z_fit = nonlinear_iq(f, *popt)
    res = calculate_residuals(z, z_fit)

    # plot
    if plotq:
        figax = plot_nonlinear_iq(f, z, popt, p0)
    else:
        figax = None, None
    return p0, popt, popt_err, res, figax

def fit_iq_circle(z, plotq = False):
    """
    Fits an IQ loop to a circle. The function describing the circle is

       [Re(S21)-A]^2 + [Im(S21)-B]^2 = R^2

       where the origin is (A, B) and the radius is R.

    Parameters:
    z (np.array): complex S21 data
    plotq (bool): if True, plots the fit and data

    Returns:
    popt (list): fit parameters (A, B, R).
    fig, ax (pyplot figure and axis): fit figure and axis, or None if not plotq
    """

    I, Q = np.real(z), np.imag(z)
    x0 = [(max(I) + min(I))/2, (max(Q) + min(Q))/2]
    x0.append((max(I) - min(I) + max(Q) - min(Q)) / 4)
    args = (I, Q)
    popt = optimize.fmin(circle_objective, x0, args, disp=0)

    if plotq:
        fig, ax = plot_circle(z, *popt)
    else:
        fig, ax = None, None
    return popt, fig

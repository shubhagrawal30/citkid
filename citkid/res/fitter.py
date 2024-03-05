import numpy as np
from scipy import optimize
from .funcs import nonlinear_iq_for_fitter, nonlinear_iq
from .util import bounds_check, calculate_residuals
from .plot import *
from .guess import guess_p0_nonlinear_iq

def fit_nonlinear_iq(f, z, bounds = None, p0 = None, fr_guess = None,
                     fit_tau = True, tau_guess = None):
    """Fit a nonlinear IQ with from an S21 sweep. Uses scipy.optimize.curve_fit.
       It is assumed that the system gain and phase are removed from the data
       before fitting. i0, q0, and tau are fitted only for fine-tuning.

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

    Returns:
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter
    popt_err (np.array): standard errors on fit parameters
    red_chi_sq (float): reduced chi squared
    p_value (float): p value
    """
    # Sort f and z
    f, z = np.array(f), np.array(z)
    ix = np.argsort(f)
    f, z = f[ix], z[ix]
    if bounds is None:
        # default bounds
        bounds = ([np.min(f), 1e3, .01, -1, -1, 0, -1e2, -1e2, -1.0e-6],
                  [np.max(f), 1e6,   1,  1, 1, 5,  1e2,  1e2,  1.0e-6])
    if p0 is None:
        # default initial guess
        p0 = guess_p0_nonlinear_iq(f, z)
    if fr_guess is not None:
        p0[0] = fr_guess
    if tau_guess is not None:
        p0[7] = tau_gues
    # Rotate z data by phi
    z0 = p0[5] + 1j * p0[6]
    z = z0 * (1 - ((1 - z / z0) * np.exp(-1j * p0[3])))
    phi_guess = p0[3]
    p0[3] = 0
    # Stack z data
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    bounds = bounds_check(p0, bounds)


    # fit
    if not fit_tau:
        # Fit with tau enforced from p0[7]
        del bounds[0][7]
        del bounds[1][7]
        del p0[7]
        def fit_func(x_lamb, a, b, c, d, e, f, g, h):
            return nonlinear_iq_for_fitter(x_lamb, a, b, c, d, e, f, g, h, tau)
        popt, pcov = optimize.curve_fit(fit_func, f, z_stacked, p0,
                                  bounds = bounds)#, maxfev = 1000 * (len(f) + 1))
        popt = np.insert(popt, 7, tau)
        # fill covariance matrix
        cov = np.ones((pcov.shape[0] + 1, pcov.shape[1] + 1)) * -1
        cov[0:7, 0:7] = pcov[0:7, 0:7]
        cov[8, 8] = pcov[7, 7]
        cov[8, 0:7] = pcov[7, 0:7]
        cov[0:7, 8] = pcov[0:7, 7]
        pcov = cov
    else:
        # Fit without enforcing tau
        popt, pcov = optimize.curve_fit(nonlinear_iq_for_fitter, f, z_stacked,
                              p0, bounds = bounds)#, maxfev = 1000 * (len(f) + 1))

    popt_err = np.sqrt(np.diag(pcov))
    z_fit = nonlinear_iq(f, *popt)
    res = calculate_residuals(z, z_fit)
    # Convert p0 and popt back to
    p0[3] += phi_guess
    popt[3] += phi_guess
    return p0, popt, popt_err, res, bounds

def fit_iq_circle(f, z, plotq = False):
    '''Fits an IQ loop to a circle. The function describing the circle is

       [Re(S21)-A]^2 + [Im(S21)-B]^2 = R^2

       where the origin is (A, B) and the radius is R.

    Parameters:
    f (np.array): frequency data
    z (np.array): complex S21 data
    plotq (bool): if True, plots the fit and data

    Returns:
    popt (list): fit parameters (A, B, R).
    popt_err (list): standard error on fit parameters -> not implemented yet
    fig, ax (pyplot figure and axis): fit figure and axis, or None if not plotq
    '''
    def objective(params, x, y):
        A, B, R = params
        error = sum(((x-A)**2+(y-B)**2-R**2)**2)
        return error
    I, Q = np.real(z), np.imag(z)
    x0 = ((max(I)+min(I))/2, (max(Q)+min(Q))/2, (max(I)-min(I)+max(Q)-min(Q))/4)
    args = (I, Q)
    popt = optimize.fmin(objective, x0, args, disp=0)

    if plotq:
        fig, ax = plot_circle(z, *popt)
    else:
        fig, ax = None, None
    return popt, fig

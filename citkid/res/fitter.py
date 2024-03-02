import numpy as np
from scipy.optimize import curve_fit
from .funcs import nonlinear_iq_for_fitter, nonlinear_iq
from .util import bounds_check, calculate_chi_squared
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
        bounds = ([np.min(f), 50, .01, -np.pi, 0, -np.inf, -np.inf, -1.0e-6],
                  [np.max(f), 200000, 1, np.pi, 5, np.inf, np.inf, 1.0e-6])
    if p0 is None:
        # default initial guess
        p0 = guess_p0_nonlinear_iq(f, z)
    if fr_guess is not None:
        p0[0] = fr_guess
    if tau_guess is not None:
        p0[7] = tau_guess
    z_stacked = np.hstack((np.real(z), np.imag(z)))
    # check bounds
    bounds = bounds_check(p0, bounds)
    # fit
    if not fit_tau:
        # Fit with tau enforced from p0[7]
        del bounds[0][7]
        del bounds[1][7]
        del p0[7]
        def fit_func(x_lamb, a, b, c, d, e, f, g):
            return nonlinear_iq_for_fitter(x_lamb, a, b, c, d, e, f, g, tau)
        popt, pcov = curve_fit(fit_func, f, z_stacked, p0, bounds = bounds)
        popt = np.insert(popt, 7, tau)
        # fill covariance matrix
        cov = np.ones((pcov.shape[0] + 1, pcov.shape[1] + 1)) * -1
        cov[0:7, 0:7] = pcov[0:7, 0:7]
        pcov = cov
    else:
        try:
            # Fit without enforcing tau
            popt, pcov = curve_fit(nonlinear_iq_for_fitter, f, z_stacked,
                                   p0, bounds = bounds)
        except Exception as e:
            print(p0, bounds)
            raise e
    # Create popt standard error and calculate chi squared
    z_fit = nonlinear_iq(f, *popt)
    popt_err = np.sqrt(np.diag(pcov))
    chi_sq, p_value = calculate_chi_squared(z = z, z_fit = z_fit)
    red_chi_sq = chi_sq / len(z)
    return p0, popt, popt_err, red_chi_sq, p_value

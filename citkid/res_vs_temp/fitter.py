import numpy as np
from scipy.optimize import curve_fit
from .guess import *
from .funcs import *
from .plot import *
from .data_io import *

def fit_fr_vs_temp(temperature, fr, gamma = 1, Tc_guess = 1.3, fr_err = None,
                   guess = None, enforced_alpha = None,
                   return_dataframe = False, plotq = False):
   """
   Fits resonance frequency versus temperature data to fr_vs_temp

   Parameters:
   temperature (array-like): temperature data in K
   fr (array-like): resonance frequency data in Hz
   gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
   Tc_guess (float): guess for the critical temperature
   fr_err (array-like or None): If not None, fr_err is the error on fr used in
        the fitting. If None, points are weighted equally
   guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess. [fr0_guess, D_guess, alpha_guess, Tc_guess]
    enforced_alpha (float or None): if float, enforces alpha to be this value
        instead of fitting. If None, fits for alpha
   return_dataframe (bool): if True, returns a pandas series of the output data
       instead of individual parameters
   plotq (bool): If True, plots the fit and initial guess

   Returns:
   p0 (list): initial guess parameters
        [fr0_guess, D_guess, alpha_guess, Tc_guess]
   popt (list): fit parameters [fr0, D, alpha, Tc]
   perr (list): fit parameter uncertainties [fr0_err, D_err, alpha_err, Tc_err]
   (fig, ax): pyplot figure and axis, or (None, None) if not plotq
   """
   temperature, fr = np.array(temperature), np.array(fr)
   if enforced_alpha is not None:
       fit_func = lambda a, b, c, e: fr_vs_temp(a, b, c, enforced_alpha, e,
                                                gamma = gamma)
   else:
       fit_func = lambda a, b, c, d, e: fr_vs_temp(a, b, c, d, e, gamma = gamma)

   ix = np.argsort(temperature)
   temperature, fr = temperature[ix], fr[ix]
   if fr_err is not None:
       fr_err = np.array(fr_err)[ix]
   # Initial guess
   if guess is not None:
       p0 = guess
       bounds = get_bounds_fr_vs_temp(p0)
   else:
       p0, bounds = guess_p0_fr_vs_temp(temperature, fr, Tc_guess, gamma)
   if enforced_alpha is not None:
       p0 = np.append(p0[:2], p0[3])
       bounds[0] = np.append(bounds[0][:2], bounds[0][3])
       bounds[1] = np.append(bounds[1][:2], bounds[1][3])
   # Fit
   if fr_err is None:
       sigma = None
       p00 = p0
   else:
       sigma = fr_err
       try:
           p00, _ = curve_fit(fit_func, temperature, fr, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except:
           p00 = p0
   try:
       popt, pcov = curve_fit(fit_func, temperature, fr, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
   except Exception as e:
       popt = [np.nan, np.nan, np.nan, np.nan]
       perr = [np.nan, np.nan, np.nan, np.nan]
   if enforced_alpha is not None:
       p0 = np.append(np.append(p0[:2], enforced_alpha), p0[2])
       popt = np.append(np.append(popt[:2], enforced_alpha), popt[2])
       perr = np.append(np.append(perr[:2],enforced_alpha), perr[2])
   # Plot
   if plotq:
       fig, ax = plot_fr_vs_temp(temperature, fr, fr_err, popt, p0, gamma)
   else:
       fig, ax = None, None

   if return_dataframe:
       row = make_fit_row_fr_vs_temp(p0, popt, perr, gamma)
       return row, (fig, ax)
   return p0, popt, perr, (fig, ax)

################################################################################
######################### Funcs without TLS component ##########################
################################################################################
def fit_fr_vs_temp_notls(temperature, fr, gamma = 1, Tc_guess = 1.3,
                         fr_err = None, guess = None, return_dataframe = False,
                         plotq = False):
   """
   Fits resonance frequency versus temperature data to fr_vs_temp_notls

   Parameters:
   temperature (array-like): temperature data in K
   fr (array-like): resonance frequency data in Hz
   gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
   Tc_guess (float): guess for the critical temperature
   fr_err (array-like or None): If not None, fr_err is the error on fr used in
        the fitting. If None, points are weighted equally
   guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess. [fr0_guess, alpha_guess, Tc_guess]
   return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
   plotq (bool): If True, plots the fit and initial guess

   Returns:
   p0 (list): initial guess parameters [fr0_guess, alpha_guess, Tc_guess]
   popt (list): fit parameters [fr0, alpha, Tc]
   perr (list): fit parameter uncertainties [fr0_err, alpha_err, Tc_err]
   (fig, ax): pyplot figure and axis, or (None, None) if not plotq
   """
   temperature, fr = np.array(temperature), np.array(fr)
   fit_func = lambda a, b, c, d: fr_vs_temp_notls(a, b, c, d, gamma = gamma)

   ix = np.argsort(temperature)
   temperature, fr = temperature[ix], fr[ix]
   if fr_err is not None:
       fr_err = np.array(fr_err)[ix]
   # Initial guess
   if guess is not None:
       p0 = guess
       bounds = get_bounds_fr_vs_temp_notls(p0)
   else:
       p0, bounds = guess_p0_fr_vs_temp_notls(temperature, fr, Tc_guess, gamma)
   # Fit
   if fr_err is None:
       sigma = None
       p00 = p0
   else:
       sigma = fr_err
       try:
           p00, _ = curve_fit(fit_func, temperature, fr, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except:
           p00 = p0
   try:
       popt, pcov = curve_fit(fit_func, temperature, fr, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
   except Exception as e:
       popt = [np.nan, np.nan, np.nan]
       perr = [np.nan, np.nan, np.nan]
   # Plot
   if plotq:
       fig, ax = plot_fr_vs_temp_notls(temperature, fr, fr_err, popt, p0, gamma)
   else:
       fig, ax = None, None

   if return_dataframe:
       row = make_fit_row_fr_vs_temp_notls(p0, popt, perr, gamma)
       return row, (fig, ax)
   return p0, popt, perr, (fig, ax)

def fit_Q_vs_temp_notls(temperature, Q, fr0_guess, Q_err = None, gamma = 1,
                        N0 = N0_Al, Tc_guess = None, guess = None,
                        return_dataframe = False, plotq = False):
   """
   Fits quality factor versus temperature to Q_vs_temp_notls

   Parameters:
   temperature (array-like): temperature data in K
   Q (array-like): quality factor data
   fr0_guess (float): guess for the frequency at 0 K
   Q_err (array-like or None): If not None, Q_err is the error on Q used in
        the fitting. If None, points are weighted equally
   gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
   N0 (float): single-spin density of states at the Fermi Level
   guess (list or None): If not None, overwrites the initial guess. Also
        overwrites fr0_guess. [fr0_guess, D_guess]
   return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
   plotq (bool): If True, plots the fit and initial guess

   Returns:
   p0 (list): initial guess parameters [fr0_guess, D_guess]
   popt (list): fit parameters [fr0, D]
   perr (list): fit parameter uncertainties [fr0_err, D_err]
   (fig, ax): pyplot figure and axis, or (None, None) if not plotq
   """
   temperature, Q = np.array(temperature), np.array(Q)
   fit_func = lambda a, b, c, d, e: Q_vs_temp_notls(a, b, c, d, e,
                                                    gamma = gamma, N0 = N0)

   ix = np.argsort(temperature)
   temperature, Q = temperature[ix], Q[ix]
   if Q_err is not None:
       Q_err = np.array(Q_err)[ix]
   # Initial guess
   if guess is not None:
       p0 = guess
       bounds = get_bounds_Q_vs_temp_notls(p0)
   else:
       alpha_guess = 0.7
       p0, bounds = guess_p0_Q_vs_temp_notls(temperature, Q, fr0_guess,
                                             alpha_guess = alpha_guess,
                                             Tc_guess = Tc_guess,
                                             gamma = gamma, N0 = N0)
   # Fit
   if Q_err is None:
       sigma = None
       p00 = p0
   else:
       sigma = Q_err
       try:
           p00, _ = curve_fit(fit_func, temperature, Q, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except:
           p00 = p0
   try:
       popt, pcov = curve_fit(fit_func, temperature, Q, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
   except Exception as e:
       raise e
       popt = [np.nan, np.nan, np.nan]
       perr = [np.nan, np.nan, np.nan]
   # Plot
   if plotq:
       fig, ax = plot_Q_vs_temp_notls(temperature, Q, Q_err, popt, p0, gamma, N0)
   else:
       fig, ax = None, None

   if return_dataframe:
       row = make_fit_row_Q_vs_temp_notls(p0, popt, perr, gamma, N0)
       return row, (fig, ax)
   return p0, popt, perr, gamma, N0, (fig, ax)

################################################################################
######################## Funcs with only TLS component #########################
################################################################################
def fit_fr_vs_temp_tls(temperature, fr, fr_err = None, guess = None,
                       return_dataframe = False, plotq = False):
   """
   Fits resonance frequency versus temperature data to fr_vs_temp_tls

   Parameters:
   temperature (array-like): temperature data in K
   fr (array-like): resonance frequency data in Hz
   fr_err (array-like or None): If not None, fr_err is the error on fr used in
        the fitting. If None, points are weighted equally
   guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess. [fr0_guess, D_guess]
   return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
   plotq (bool): If True, plots the fit and initial guess

   Returns:
   p0 (list): initial guess parameters [fr0_guess, D_guess]
   popt (list): fit parameters [fr0, D]
   perr (list): fit parameter uncertainties [fr0_err, D_err]
   (fig, ax): pyplot figure and axis, or (None, None) if not plotq
   """
   temperature, fr = np.array(temperature), np.array(fr)
   fit_func = lambda a, b, c: fr_vs_temp_tls(a, b, c)

   ix = np.argsort(temperature)
   temperature, fr = temperature[ix], fr[ix]
   if fr_err is not None:
       fr_err = np.array(fr_err)[ix]
   # Initial guess
   if guess is not None:
       p0 = guess
       bounds = get_bounds_fr_vs_temp_tls(p0)
   else:
       p0, bounds = guess_p0_fr_vs_temp_tls(temperature, fr)
   # Fit
   if fr_err is None:
       sigma = None
       p00 = p0
   else:
       sigma = fr_err
       try:
           p00, _ = curve_fit(fit_func, temperature, fr, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except:
           p00 = p0
   try:
       popt, pcov = curve_fit(fit_func, temperature, fr, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
   except Exception as e:
       popt = [np.nan, np.nan]
       perr = [np.nan, np.nan]
   # Plot
   if plotq:
       fig, ax = plot_fr_vs_temp_tls(temperature, fr, fr_err, popt, p0)
   else:
       fig, ax = None, None

   if return_dataframe:
       row = make_fit_row_fr_vs_temp_tls(p0, popt, perr)
       return row, (fig, ax)
   return p0, popt, perr, (fig, ax)

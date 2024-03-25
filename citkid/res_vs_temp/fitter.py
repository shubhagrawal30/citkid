import numpy as np
from scipy.optimize import curve_fit
from .guess import guess_p0_fr_vs_temp, get_bounds_fr_vs_temp
from .funcs import fr_vs_temp
from .plot import plot_fr_vs_temp
from .data_io import make_fit_row

def fit_fr_vs_temp(temperature, fr, Tc_guess = 1.3, fr_err = None, guess = None,
                   return_dataframe = False, plotq = False):
   """
   Fits resonance frequency versus temperature data to .funcs.fr_vs_temp.

   Parameters:
   temperature (array-like): temperature data in K
   fr (array-like): resonance frequency data in Hz
   Tc_guess (float): guess for the critical temperature
   fr_err (array-like or None): If not None, fr_err is the error on fr used in
        the fitting. If None, points are weighted equally
   guess (list or None): If not None, overwrites the initial guess. Also
        overwrites Tc_guess. [fr0_guess, D_guess, alpha_guess, Tc_guess]
   return_dataframe (bool): If True, returns a pandas series of the output data
       instead of individual parameters
   plotq (bool): If True, plots the fit and initial guess

   Returns:
   p0 (list): initial guess parameters
        [fr0_guess, D_guess, alpha_guess, Tc_guess]
   popt (list): fit parameters [fr0, D, alpha, Tc]
   perr (list): fit parameter uncertainties [fr0_err, D_err, alpha_err, Tc_err]
   (fig, ax): pyplot figure and axis, or (None, None) if not plotq
   """
   # raise Exception('Need to add plotting and data_io')
   temperature, fr = np.array(temperature), np.array(fr)

   ix = np.argsort(temperature)
   temperature, fr = temperature[ix], fr[ix]
   if fr_err is not None:
       fr_err = np.array(fr_err)[ix]
   # Initial guess
   if guess is not None:
       p0 = guess
       bounds = get_bounds_fr_vs_temp(p0)
   else:
       p0, bounds = guess_p0_fr_vs_temp(temperature, fr, Tc_guess)
   # Fit
   if fr_err is None:
       sigma = None
       p00 = p0
   else:
       sigma = fr_err * 1e6
       try:
           p00, _ = curve_fit(fr_vs_temp, temperature, fr, sigma = None,
                              p0 = p0, bounds = bounds)
          # To fit with sigma, the initial guess must be really good, so
          # update the initial guess with curve_fit without sigma
       except:
           p00 = p0
   try:
       popt, pcov = curve_fit(fr_vs_temp, temperature, fr, sigma = sigma,
                              p0 = p00, bounds = bounds, absolute_sigma = True)
       perr = np.sqrt(np.diag(pcov))
   except Exception as e:
       raise e
       popt = [np.nan, np.nan, np.nan]
       perr = [np.nan, np.nan, np.nan]
   # Plot
   if plotq:
       fig, ax = plot_fr_vs_temp(temperature, fr, fr_err, popt, p0)
   else:
       fig, ax = None, None

   if return_dataframe:
       row = make_fit_row(p0, popt, perr)
       return row, (fig, ax)
   return p0, popt, perr, (fig, ax)

import matplotlib.pyplot as plt
import numpy as np
from .funcs import fr_vs_temp

def plot_fr_vs_temp(temperature, fr, fr_err, popt, p0):
    """
    Plots the fit and initial guess to fr_vs_temp

    Parameters:
    temperatures (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    fr_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    p0 (list): initial guess parameters
        [fr0_guess, D_guess, alpha_guess, Tc_guess]
    popt (list): fit parameters [fr0, D, alpha, Tc]

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    fig, ax = plt.subplots(figsize = [3, 2.8], dpi = 200, layout = 'tight')
    ax.set_ylabel(r'$f_r$ (MHz)')
    ax.set_xlabel(r'Temperature (K)')
    ax.set_xscale('log')
    if fr_err is None:
        ax.plot(temperature, fr * 1e-6, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(temperature, fr * 1e-6, yerr = fr_err * 1e3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')

    xsamp = np.geomspace(min(temperature), max(temperature), 200)
    ysamp = fr_vs_temp(xsamp, *popt)
    ax.plot(xsamp, ysamp * 1e-6, '--r', label = 'Fit')
    ysamp = fr_vs_temp(xsamp, *p0)
    ax.plot(xsamp, ysamp * 1e-6, ':k', label = 'Guess')
    ax.legend()
    return fig, ax

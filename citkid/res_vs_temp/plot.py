import matplotlib.pyplot as plt
import numpy as np
from .funcs import *

def plot_fr_vs_temp(temperature, fr, fr_err, popt, p0, gamma):
    """
    Plots the fit and initial guess to fr_vs_temp

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    fr_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

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
        ax.errorbar(temperature, fr * 1e-6, yerr = fr_err * 1e-6, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')

    xsamp = np.geomspace(min(temperature), max(temperature), 200)
    ysamp = fr_vs_temp(xsamp, *popt, gamma = gamma)
    ax.plot(xsamp, ysamp * 1e-6, '--r', label = 'Fit')
    ysamp = fr_vs_temp(xsamp, *p0, gamma = gamma)
    ax.plot(xsamp, ysamp * 1e-6, ':k', label = 'Guess')
    ax.legend()
    return fig, ax

################################################################################
######################### Funcs without TLS component ##########################
################################################################################
def plot_fr_vs_temp_notls(temperature, fr, fr_err, popt, p0, gamma):
    """
    Plots the fit and initial guess to fr_vs_temp_notls

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    fr_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits

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
        ax.errorbar(temperature, fr * 1e-6, yerr = fr_err * 1e-6, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')

    xsamp = np.geomspace(min(temperature), max(temperature), 200)
    ysamp = fr_vs_temp_notls(xsamp, *popt, gamma = gamma)
    ax.plot(xsamp, ysamp * 1e-6, '--r', label = 'Fit')
    ysamp = fr_vs_temp_notls(xsamp, *p0, gamma = gamma)
    ax.plot(xsamp, ysamp * 1e-6, ':k', label = 'Guess')
    ax.legend()
    return fig, ax

def plot_Q_vs_temp_notls(temperature, Q, Q_err, popt, p0, gamma, N0):
    """
    Plots the fit and initial guess to Q_vs_temp_notls

    Parameters:
    temperature (array-like): temperature data in K
    Q (array-like): quality factor data
    Q_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    N0 (float): single-spin density of states at the Fermi Level

    Returns:
    fig, ax: pyplot figure and axis, or (None, None) if not plotq
    """
    fig, ax = plt.subplots(figsize = [3, 2.8], dpi = 200, layout = 'tight')
    ax.set_ylabel(r'$Q$ (kHz / GHz)')
    ax.set_xlabel(r'Temperature (K)')
    ax.set_xscale('log')
    if Q_err is None:
        ax.plot(temperature, Q * 1e-6, marker = '.', color = plt.cm.viridis(0),
                linestyle = '', label = 'Data')
    else:
        ax.errorbar(temperature, Q * 1e-6, yerr = Q * 1e-3, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')

    xsamp = np.geomspace(min(temperature), max(temperature), 200)
    ysamp = Q_vs_temp_notls(xsamp, *popt, gamma = gamma, N0 = N0)
    ax.plot(xsamp, ysamp * 1e-6, '--r', label = 'Fit')
    ysamp = Q_vs_temp_notls(xsamp, *p0, gamma = gamma, N0 = N0)
    ax.plot(xsamp, ysamp * 1e-6, ':k', label = 'Guess')
    ax.legend()
    return fig, ax

################################################################################
######################## Funcs with only TLS component #########################
################################################################################
def plot_fr_vs_temp_tls(temperature, fr, fr_err, popt, p0):
    """
    Plots the fit and initial guess to fr_vs_temp_tls

    Parameters:
    temperature (array-like): temperature data in K
    fr (array-like): resonance frequency data in Hz
    fr_err (None or array-like): error bars for the plot, or None to plot
        without error bars
    popt (list): fit parameters
    p0 (list): initial guess parameters

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
        ax.errorbar(temperature, fr * 1e-6, yerr = fr_err * 1e-6, marker = '.',
                    color = plt.cm.viridis(0), linestyle = '', label = 'Data')

    xsamp = np.geomspace(min(temperature), max(temperature), 200)
    ysamp = fr_vs_temp_tls(xsamp, *popt)
    ax.plot(xsamp, ysamp * 1e-6, '--r', label = 'Fit')
    ysamp = fr_vs_temp_tls(xsamp, *p0)
    ax.plot(xsamp, ysamp * 1e-6, ':k', label = 'Guess')
    ax.legend()
    return fig, ax

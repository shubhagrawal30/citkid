import matplotlib.pyplot as plt
import numpy as np
from .funcs import nonlinear_iq
import warnings
from io import BytesIO

def plot_nonlinear_iq(f, z, popt, p0, plot_guess = False):
    """
    Plots the fit to the nonlinear iq model

    Parameters:
    f (np.array): frequency data in Hz
    z (np.array): complex iq data
    popt (list): fit parameters
    p0 (list): initial guess parameters
    plot_guess (bool): If True, also plots the guess curve

    Returns:
    fig, ax (pyplot figure and axes): plot of data with fit, both in IQ space
        and as mag(S21)
    """
    fig, axs = plt.subplots(1, 2, figsize = [6, 2.8], dpi = 200)
    axs[0].set_ylabel('Q')
    axs[0].set_xlabel('I')
    axs[1].set_ylabel(r'$S_{21}$ (dB)')
    f0 = np.mean(f)
    axs[1].set_xlabel(f'(f - {round(f0 / 1e9, 4)} MHz) (kHz)')
    fig.tight_layout()

    axs[0].plot(np.real(z), np.imag(z), '.', color = plt.cm.viridis(0),
                markersize = 5, label = 'data')
    axs[1].plot((f - f0) / 1e3, 20 * np.log10(abs(z)), '.',
                color = plt.cm.viridis(0), markersize = 5)

    fsamp = np.linspace(min(f), max(f), 1000)
    zsamp = nonlinear_iq(fsamp, *popt)
    axs[0].plot(np.real(zsamp), np.imag(zsamp), '--k', label = 'fit')
    axs[1].plot((fsamp - f0) / 1e3, 20 * np.log10(abs(zsamp)), '--k')

    if plot_guess:
        zsamp = nonlinear_iq(fsamp, *p0)
        axs[0].plot(np.real(zsamp), np.imag(zsamp), '--r', label = 'guess')
        axs[1].plot((fsamp - f0) / 1e3, 20 * np.log10(abs(zsamp)), '--r')
        axs[0].legend(framealpha = 1)
    return fig, axs

def plot_circle(z, A, B, R):
    """
    Plots IQ data with a circular fit

    Parameters:
    z (np.array): complex IQ data
    A, B (float, float): circle origin
    R (float): circle radius

    Returns:
    fig, ax (pyplot figure and axis): data and fit plot
    """
    fig, ax = plt.subplots(figsize = (4, 4), dpi = 300)
    ax.plot(np.real(z), np.imag(z), 'r.')
    ax.set_aspect('equal', adjustable='datalim')
    cir = plt.Circle((A, B), R, color='k', fill=False, label='IQ loop fit')
    ax.add_patch(cir)
    ax.set(xlabel='I', ylabel='Q')
    return fig, ax

def plot_gain_fit(f0, dB0, f, dB, phase, p_amp, p_phase):
    """
    Plots the fit to gain amplitude and phase data

    Parameters:
    f0 (np.array): raw frequency data
    dB0 (np.array): raw amplitude data
    f (np.array): cut frequency data
    dB (np.array): cut amplitude data
    phase (np.array): cut phase data
    p_amp (list): amplitude fit parameters
    p_phase (list): phase fit parameters

    Returns:
    fig, axs (pyplot figure and axis): data and fit plot
    """
    fmean = np.mean(f0)
    fig, axs = plt.subplots(1, 2, figsize=[6, 2.8], dpi = 200)
    axs[1].set_ylabel('Phase')
    axs[1].set_xlabel(f'(f - {round(fmean / 1e9, 4)} MHz) (kHz)')
    axs[0].set_ylabel('|S21| (dB)')
    axs[0].set_xlabel(f'(f - {round(fmean / 1e9, 4)} MHz) (kHz)')

    color = plt.cm.viridis(0)
    color0 = plt.cm.viridis(0.99)
    axs[0].plot((f0 - fmean) * 1e-3, dB0, '.', color = color0, label='Raw data')
    axs[0].plot((f - fmean) * 1e-3, dB, '.', color = color, label='Fitted data')
    fsamp = np.linspace(np.min(f0),np.max(f0), 100)
    if ~np.any(np.isnan(p_amp)):
        axs[0].plot((fsamp - fmean) * 1e-3, np.polyval(p_amp, fsamp), '--k', label='Fit')

    axs[1].plot([], [], '.', color = color0, label = 'Raw data')
    axs[1].plot((f - fmean) * 1e-3, phase, '.', color = color, label='Fitted data')
    if ~np.any(np.isnan(p_phase)):
        axs[1].plot((fsamp - fmean) * 1e-3, np.polyval(p_phase, fsamp), '--k', label='Fit')

    axs[1].legend(framealpha=1)
    fig.tight_layout()
    return fig, axs

################################################################################
################################ Utilities #####################################
################################################################################
def save_figure_to_memory(fig):
    """
    Saves a matplotlib figure to memory. Use this to easily stitch together
    multiple figures without saving extra files

    Parameters:
    fig (pyplot.figure): figure to save

    Returns:
    buf (BytesIO): memory buffer of saved figure
    """
    buf = BytesIO()
    fig.set_facecolor('white')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0.05)
    buf.seek(0)
    return buf

def combine_figures_vertically(fig1, fig2):
    """
    Combine two matplotlib figures vertically for saving as a single file

    Parameters:
    fig1, fig2 (pyplot.figure): figures to combine

    Returns:
    fig (pyplot.figure): combined figure
    """
    buf1 = save_figure_to_memory(fig1)
    buf2 = save_figure_to_memory(fig2)
    plt.close(fig1)
    plt.close(fig2)
    fig, axs = plt.subplots(2, 1, dpi = 200)
    for ax in axs:
        ax.set_axis_off()
    axs[0].imshow(plt.imread(buf1))
    axs[1].imshow(plt.imread(buf2))
    fig.tight_layout()
    return fig

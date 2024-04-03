import matplotlib.pyplot as plt
import numpy as np
from ..res.plot import plot_circle

def plot_cal(ffine, zfine, popt_circle, fnoise, znoise, znoise_offres,
             theta_range, theta_fine, theta, poly):
    """
    Plot theta and x calibration. Left plot is the IQ loop with noise, and right
    plot is the theta to x calibration, if on-resonance noise is provided.

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data, with gain removed
    popt_circle (np.array): circle fit parameters
    fnoise (float): on-resonance noise tone frequency in Hz
    znoise (array-like or None): on-resonance complex noise data
    fnoise_offres (float): off-resonance noise tone frequency in Hz
    znoise_offres (array-like or None): off-resonance complex noise data
    theta (np.array): theta noise timestream
    theta_range (tuple): [lower, upper] range of theta over which the fit was
        performed
    theta_fine (np.array): fine sweep theta data
    poly (np.array): x vs theta polynomial fit parameters

    Returns:
    fig (plt.figure): calibration plot figure
    """
    fig, ax = plot_circle(zfine, *popt_circle)
    origin = popt_circle[0] + 1j * popt_circle[1]
    ax.plot([], [], '.r', label = 'data')
    ax.plot([], [], '-k', label = 'fit')

    if znoise is not None:
        fig.set_size_inches((8, 4))
        ax2 = fig.add_axes([0.5, 0, 0.4, 0.8])
        ax.set_position([0, 0, 0.4, 0.8])
        ax2.set_ylabel('x (Hz / MHz)')
        ax2.set_xlabel('Phase')
        nevery = int(len(znoise) / 10000)
        if nevery == 0:
            nevery = 1
        ax.plot(np.real(znoise[::nevery]), np.imag(znoise[::nevery]), '.b',
                markersize = 1, label = 'on-res noise')
        zn_med = np.mean(znoise - origin)
        r1 = zn_med * np.exp(1j * max(theta_range)) + origin
        r2 = zn_med * np.exp(1j * min(theta_range)) + origin
        ax.plot([np.real(origin), np.real(r1)],
                [np.imag(origin), np.imag(r1)], '--b')
        ax.plot([np.real(origin), np.real(r2)],
                [np.imag(origin), np.imag(r2)], '--b')
        # theta -> x calibration
        ax2.plot(theta[::nevery],
                 (1 - np.polyval(poly, theta[::nevery]) / fnoise) * 1e6, '.b',
                 markersize = 5)
        ix = (theta_fine < max(theta_range) + np.pi / 8)
        ix = ix & (theta_fine > min(theta_range) - np.pi / 8)
        theta_samp = np.linspace(min(theta_fine[ix]), max(theta_fine[ix]), 200)
        ax2.plot(theta_samp, (1 - np.polyval(poly, theta_samp) / fnoise) * 1e6,
                 '-k')
        ax2.plot(theta_fine[ix], (1 - ffine[ix] / fnoise) * 1e6, '.r')
        ax2.axvline(theta_range[0], linestyle = '--', color = 'b')
        ax2.axvline(theta_range[1], linestyle = '--', color = 'b')

    if znoise_offres is not None:
        nevery = int(len(znoise_offres) / 10000)
        if nevery == 0:
            nevery = 1
        ax.plot(np.real(znoise_offres[::nevery]),
                np.imag(znoise_offres[::nevery]), '.g', markersize = 1,
                label = 'off-res noise')

    ax.plot([], [], '--k', label = 'cal range')
    ax.legend(loc = 'center')
    return fig

def plot_timestream(dt, theta, dt_offres, theta_offres, poly, x, fnoise):
    """
    Plots noise timestreams. If theta is None, plots only the off-resonance
    theta timestream. If theta_offres is None, plots only the on-resonance
    theta timestream and the on-resonance x timestream, with and without
    deglitching and cosmic ray removal. If neither are None, plots on- and off-
    resonance timestreams.

    Parameters:
    dt (float or None): on-resonance timestream sample time in s
    theta (np.array or None): on-resonance theta timestream data
    dt_offres (float or None): off-resonance timestream sample time in s
    theta_offres (np.array or None): off-resonance theta timestream data
    poly (np.array): polynomial fit to x versus theta
    x (np.array): on-resonance x timestream data
    fnoise (float): on-resonance noise tone frequency in Hz

    Returns:
    fig (plt.figure): timestream plot
    """
    fig = None
    nplots = 0
    if theta is not None:
        nplots += 2
    if theta_offres is not None:
        nplots += 1
    if nplots > 1:
        fig, axs = plt.subplots(nplots, 1, figsize = [8, nplots * 2],
                                layout = 'tight', dpi = 200)
    if nplots == 1:
        axs = [axs]
    if theta is not None:
        axs[0].set_ylabel('x (Hz / kHz)\non-res')
        axs[1].set_ylabel('Phase\non-res')

        time = np.arange(0, len(theta) * dt, dt)
        axs[1].plot(time, theta, 'k')
        x_raw = 1 - np.polyval(poly, theta) / fnoise
        axs[0].plot(time, x_raw * 1e3, 'k', label = 'raw data')
        axs[0].plot(time, x * 1e3, 'r', label = 'deglitched data')
        axs[0].legend(loc = 'lower left')
    if theta_offres is not None:
        i = nplots - 1
        axs[i].set_ylabel('Phase\noff-res')
        axs[i].set_xlabel('Time (s)')
        time = np.arange(0, len(theta_offres) * dt_offres, dt_offres)
        axs[i].plot(time, theta_offres, 'k')
    return fig

def plot_psd(f_psd, spar, sper, sxx, f_psd_offres, spar_offres, sper_offres):
    if sxx is None:
        fig, ax0 = plt.subplots(figsize = [6, 2.5], dpi = 200)
    else:
        fig, [ax1, ax0] = plt.subplots(2, 1, figsize = [6, 5], sharex = True,
                                       dpi = 200)
        ax1.set_ylabel(r'$S_{xx}$ (1 / Hz)')
        ax1.set_yscale('log')
    for ax in (ax0, ax1):
        ax.set_xscale('log')
    ax0.set_xlabel('Frequency (Hz)')
    ax0.set_ylabel(r'PSD (dBc)')
    # Need to bin before plotting
    if spar_offres is not None:
        ax0.plot(f_psd_offres[1:], spar_offres[1:], '--r', label = 'par off')
        ax0.plot(f_psd_offres[1:], sper_offres[1:], '--b', label = 'per off')
    if spar is not None:
        ax0.plot(f_psd[1:], spar[1:], '-r', label = 'par on')
        ax0.plot(f_psd[1:], sper[1:], '-b', label = 'per on')
    ax0.legend()
    return fig

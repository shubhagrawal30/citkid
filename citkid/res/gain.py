import numpy as np
from .plot import plot_gain_fit

def fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, frs = [], Qrs = [],
                              plotq=False):
    """
    Removes the gain-scan fit parameters from the fine scan data

    Qrs should be no higher than 10 X Qr of the resonances

    Parameters:
    fgain (np.array): gain sweep frequency data
    zgain (np.array): gain sweep complex S21 data
    ffine (np.array): fine sweep frequency data
    zfine (np.array): fine sweep complex S21 data
    frs (list of float): resonance frequencies to cut from the gain scan
    Qrs (list of float): Qrs to cut from the gain scan.
    plotq (bool): If True, also plots fits to the gain scan
        and corrections to the fine-scan.

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB
    p_phase (np.array): 1st-order polynomial fit parameters to phase
    z_rmvd (np.array): zfine with gain amplitude and phase removed
    fig, axs (pyplot figure and axes, or None): if plotq, returns a plot of the
        gain amplitude and phase fits. Otherwise, returns (None, None)
    """
    fgain, zgain = np.array(fgain), np.array(zgain)
    ffine, zfine = np.array(ffine), np.array(zfine)
    fr_spans = []
    for fr, Qr in zip(frs, Qrs):
        fr_spans.append((fr, fr / Qr))
    p_amp, p_phase, (fig_gain, axs_gain) = fit_gain(fgain, zgain, fr_spans, plotq)
    z_rmvd = remove_gain(ffine, zfine, p_amp, p_phase)
    return p_amp, p_phase, z_rmvd, (fig_gain, axs_gain)

def remove_gain(f, z, p_amp, p_phase):
    """
    Removes the gain amplitude and phase from data

    Parameters:
    f (np.array): frequency data in Hz
    z (np.array): complex S21 data
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase

    Returns:
    z_rmvd (np.array): complex S21 data with gain amplitude and phase removed
    """
    z_rmvd = z / 10 ** (np.polyval(p_amp, f) / 20)
    z_rmvd = z_rmvd / np.exp(1j * np.polyval(p_phase, f))
    return z_rmvd

def fit_gain(f, z, fr_spans, plotq = False):
    """
    Fits the amplitude and phase of gain data. Amplitude is fit to a 2nd order
    polynomial and phase is fit to a 1st order polynomial

    Parameters:
    f <np.array>: gain frequency array
    z <np.array>: gain complex S21 array
    fr_spans (list): values are tuples (<float>,<float>) where the first value
        is the resonance frequency and the second is the span. These frequencies
        are removed from the gain data
    plotq (bool): If True, plots the fits and returns the figure

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to gain amplitude
    p_phase (np.array): 1st-order polynomial fit parameters to gain phase
    fig, axs (pyplot figure and axes, or None): if plotq, returns a plot of the
        gain amplitude and phase fits. Otherwise, returns (None, None)
    """
    for r in fr_spans:
        if not(len(r) == 2):
            raise ValueError('Incorrect fr_spans format')
        if r[1] < 0:
            raise ValueError('Span must be positive')
    dB = 20 * np.log10(abs(z))
    if plotq:
        f0, z0 = f.copy(), z.copy()
        dB0 = dB.copy()
    # Remove resonances from span data
    fcuts = [] # frequencies at which data is cut
    for fr, span in fr_spans:
        ix0 = f < fr - span
        ix1 = f > fr + span
        ix = ix0|ix1
        fix = f[~ix]
        if len(fix):
            fcuts.append(np.mean(fix))
        f, z, dB = f[ix], z[ix], dB[ix]
    phase = np.angle(z)
    phase = np.unwrap(2 * phase) / 2
    phase0 = phase.copy()
    if plotq:
        f1 = f.copy()

    # reject outliers and fit
    ix = abs(dB - np.mean(dB)) < 5 * np.std(dB)
    f, z, dB, phase = f[ix], z[ix], dB[ix], phase[ix]
    try:
        p_amp = np.polyfit(f, dB, 2)
        # Fit to each cut portion of phase separately, to avoid unwrapping problems
        fcuts = [0] + fcuts + [np.inf]
        fcuts = [[fcuts[i], fcuts[i + 1]] for i in range(len(fcuts) - 1)]
        pps = []
        for fcut in fcuts:
            ix = (f >= fcut[0]) & (f <= fcut[1])
            if len(f[ix]) >= 4:
                pps.append(np.polyfit(f[ix], phase[ix], 1))
        p_phase = np.mean(pps, axis = 0)
    except Exception as e:
        p_amp = np.array([np.nan,np.nan,np.nan])
        p_phase = np.array([np.nan,np.nan])

    if plotq:
        fig, axs = plot_gain_fit(f0, dB0, f, dB, phase, p_amp, p_phase)
    else:
        fig, axs = None, None
    return p_amp, p_phase, (fig, axs)

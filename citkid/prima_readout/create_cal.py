import numpy as np
from .apply_cal import calculate_theta
from ..res.gain import fit_gain, remove_gain
from ..res.fitter import fit_iq_circle

def get_theta_cal(ffine, zfine, fgain, zgain, fs, Qs, plotq = False):
    """
    Given fine sweep data, produce the calibration data to convert complex S21
    data to theta

    Parameters:
    ffine (array-like): fine sweep frequency data in Hz
    zfine (array-like): fine sweep complex S21 data
    fgain (array-like): gain sweep frequency data in Hz
    zgain (array-like): gain sweep complex S21 data
    fs (array-like): resonance frequencies to remove from gain sweeps when
        fitting gain data
    Qs (array-like): scaled resonance quality factors to remove from gain sweeps
        when fitting gain data. frequency ranges of width fs / Qs centered on fs
        are removed from the gain sweep data
    plotq (bool): if True, creates plots of the calibration

    Returns:
    p_amp (np.array): polynomial fit parameters to gain amplitude
    p_phase (np.array): polynomial fit parameters to gain phase
    origin (complex): origin of the resonance circle
    x (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    theta_fine (np.array): fine sweep theta data corresponding to the values in
        zfine
    fig_gain, fig_fine (pyplot.figure or None): gain and fine scan calibration
        plots if plotq, else (None, None)
    """
    # Fit gain
    fr_spans = [[fi, fi / Qi] for fi, Qi in zip(fs, Qs)]
    p_amp, p_phase, (fig_gain, ax_gain) = fit_gain(fgain, zgain, fr_spans,
                                                  plotq = plotq)
    zfine_rmvd = remove_gain(ffine, zfine, p_amp, p_phase)
    # Theta vector
    origin, x, fig_fine = get_theta_vec(zfine_rmvd, plotq = plotq)
    # fine-scan theta
    theta_fine = calculate_theta(zfine_rmvd, origin, x)
    return p_amp, p_phase, origin, x, theta_fine, (fig_gain, fig_fine)

def get_theta_vec(zfine_rmvd, plotq = False):
    """
    Given an IQ loop with gain removed, return the origin of the resonance
    circle and the unit vector that points from the origin to theta = 0

    Parameters:
    zfine_rmvd (np.array): fine sweep complex S21 data with gain removed

    Returns:
    origin (complex): origin of the resonance circle
    x (complex): unit vector pointing from the origin to the furthest point in
        zfine_rmvd from the origin
    """
    popt_circle, fig = fit_iq_circle(zfine_rmvd, plotq = plotq)
    origin = popt_circle[0] + 1j * popt_circle[1]
    R = popt_circle[2]
    z_offres = np.mean(np.concatenate([zfine_rmvd[:2], zfine_rmvd[-2:]]))
    ix = np.argmax(np.abs(zfine_rmvd - z_offres))
    z_onres = zfine_rmvd[ix]
    # Get x, the direction between the origin of the circle and the on-res point
    x = (z_onres - origin)
    x /= np.abs(x)
    if plotq:
        ax = fig.axes[0]
        ax.plot([], [], 'or', label = 'data')
        ax.plot([], [], '-k', label = 'circle fit')
        ax.plot([np.real(origin), np.real(z_onres)],
                [np.imag(origin), np.imag(z_onres)],
                '--k', label = r'$\theta = 0$')
        ax.plot(np.real(origin), np.imag(origin), 'xk', label = 'origin')
        ax.legend(framealpha = 1, bbox_to_anchor = (0.5, 0.7), loc = 'center',
                  ncols = 2, fontsize = 8)
    return origin, x, fig

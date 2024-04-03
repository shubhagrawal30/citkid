import numpy as np
from ..res.fitter import fit_iq_circle
from .plot import plot_cal, plot_timestream, plot_psd
import pyfftw

def compute_psd(ffine, zfine, fnoise, znoise, dt, fnoise_offres = None, znoise_offres = None,
                dt_offres = None, flag_crs =True, deglitch = 5, plot_calq = True,
                plot_psdq = True, plot_timestreamq = True):
    """
    Computes parallel and perpendicular noise PSDs, as well as Sxx

    Parameters:
    ffine (array-like): fine scan frequency data in Hz
    zfine (array-like): fine scan complex S21 data, with gain removed
    fnoise (float): on-resonance noise tone frequency in Hz
    znoise (array-like or None): on-resonance complex noise data, with gain
        removed. For off-resonance noise only, set this to None
    dt (float): sample time of the on-resonance noise timestream in s
    fnoise_offres (float): off-resonance noise tone frequency in Hz
    znoise_offres (array-like or None): off-resonance complex noise data, with
        gain removed. For on-resonance noise only, set this to None
    dt_offres (float): sample time of the off-resonance noise timestream in s
    flag_crs (bool): If True, flags cosmic rays and returns a list of indices
        where they were found. If False, does not flag cosmic rays
    deglitch (float or None): threshold for removing glitched data points from
        the timestream, or None to bypass deglitching. Points more than
        deglitch times the standard deviations of the theta timestream are
        removed from the data.
    plot_calq (bool): If True, plots the calibration figure. Else, returns None
        for the figure
    plot_psdq (bool): If True, plots the PSD figure. Else, returns None for the
        figure
    plot_timestreamq (bool): If True, plots the timestream figure. Else, returns
        None for the figure
    verbose (bool): If True, displays a progress bar while analyzing noise

    Returns:
    s_par (np.array): PSD of noise parallel to IQ loop
    s_per (np.array): PSD of noise perpendicular to IQ loop
    s_xx (np.array): PSD of fractional frequency noise
    f (np.array): frequency array for all PSDs
    theta (np.array): theta timestream with no cosmic ray removal or deglitching
    theta_clean (np.array): theta timestream with cosmic rays removed
    x_clean (np.array): fractional frequency timestream with cosmic rays removed
    poly (np.array): polynomial fit results for x versus theta fit
    theta_range (tuple): [lower, upper] range of theta over which the fit was
        performed
    cr_indices (list): values (int) are indices at which cosmic rays were found
        and removed
    fig_cal (plt.figure): plot of the IQ loop with noise balls and calibration
    fig_psd (plt.figure): plot of the PSDs
    fig_timestream (plt.figure): plot of the timestream data
    """
    # Prepare data
    ffine, zfine= np.array(ffine), np.array(zfine)
    ix = np.argsort(ffine)
    ffine, zfine = ffine[ix], zfine[ix]
    # Fit circle
    popt_circle, _ = fit_iq_circle(zfine, plotq = False)
    origin = popt_circle[0] + 1j * popt_circle[1]
    # Extract theta and x
    if znoise_offres is not None:
        znoise_offres = np.array(znoise_offres)
        theta_fine, theta_offres =\
        calibrate_timestreams(origin, ffine, zfine, fnoise_offres, znoise_offres, dt_offres, deglitch, offres = True)

        spar_offres, sper_offres = get_par_per_psd(dt_offres, znoise_offres, origin)
        f_psd_offres = np.fft.rfftfreq(len(znoise_offres), d = dt)
    else:
        theta_offres = None
        f_psd_offres, spar_offres, sper_offres = None, None, None
    if znoise is not None:
        znoise = np.array(znoise)
        theta_fine, theta, theta_range, poly, x, cr_indices, znoise_clean =\
        calibrate_timestreams(origin, ffine, zfine, fnoise, znoise, dt, deglitch, offres = False)

        sxx = get_psd(x, dt)
        spar, sper = get_par_per_psd(dt, znoise_clean, origin)
        f_psd = np.fft.rfftfreq(len(znoise_clean), d = dt)
    else:
        theta, poly, x, cr_indices, theta_range = None, None, None, None, None
        f_psd, spar, sper, sxx = None, None, None, None
    # Make PSDs

    # Plots
    if plot_calq:
        fig_cal = plot_cal(ffine, zfine, popt_circle, fnoise, znoise,
                           znoise_offres, theta_range, theta_fine, theta, poly)
    else:
        fig_cal = None
    if plot_timestreamq:
        fig_timestream = plot_timestream(dt, theta, dt_offres, theta_offres, poly, x, fnoise)
    else:
        fig_timestream = None
    if plot_psdq:
        fig_psd = plot_psd(f_psd, spar, sper, sxx, f_psd_offres, spar_offres, sper_offres)
    return f_psd, spar, sper, sxx, f_psd_offres, spar_offres, sper_offres 


def get_par_per_psd(dt, znoise, origin):
    """
    Calculates the parallel and perpendicular power spectral densities of the
    S21 noise

    Parameters:
    dt (float): noise sample time in s
    znoise (np.array): S21 noise timestream
    origin (complex): center of the IQ loop

    Returns:
    spar <np.array>: logarithmic parallel noise PSD
    sper <np.array>: logarithmic perpendicular noise PSD
    """
    znoise_shift = znoise - origin
    znoise_mean = np.mean(znoise_shift)
    angle = np.arctan2(np.imag(znoise_mean), np.real(znoise_mean))
    znoise_shift_rot = np.exp(-1j*angle)*znoise_shift
    # After centering and rotating, the par/perp components
    # to the IQ loop are the imaginary/real parts.
    zpar = np.imag(znoise_shift_rot)
    zper = np.real(znoise_shift_rot)

    spar = 10 * np.log10(get_psd(zpar, dt))
    sper = 10 * np.log10(get_psd(zper, dt))
    return spar, sper

def get_psd(x, dt):
    """
    Given a timeseries with a constant sample rate, calculates the power
    spectral density.

    Parameters:
    x (np.array): timeseries data
    dt (float): sample rate of timeseries

    Returns:
    psd (np.array): power spectral density
    """
    a = pyfftw.interfaces.numpy_fft.rfft(x)
    psd = 2 * np.abs(a) ** 2 * dt / len(x)
    return psd

def calibrate_timestreams(origin, ffine, zfine, fnoise, znoise, dt, deglitch,
                          offres = False):
    """
    Calculates theta and x timestreams given complex IQ noise timestreams.
    1) calculate theta of the sweep data an noise timestream
    2) flag and remove cosmic rays
    3) deglitch data and perform polynomial fit to get x from theta, if not offres

    May want to center this on the noise frequency, rather than the mean of the calibrated f data

    Parameters:
    origin (complex): origin of the iq loop
    ffine (np.array): frequency data in Hz
    zfine (np.array): complex IQ data
    fnoise (float): noise tone frequency in Hz
    znoise (np.array): complex noise timestream
    dt (float): timestream sample time
    deglitch (float or None): deglitching threshold, or None to bypass deglitching
    offres (bool): if True, only calibrates theta and bypasses x calibration

    Returns:
    theta_fine (np.array): theta of the complex sweep data
    theta (np.array): theta timestream
    # If not offres:
    theta_range (np.array): [lower, upper] bound on theta used in the polynomial
        fit to theta_fine versus ffine
    poly (np.array): polynomial fit parameters to theta_fine versus ffine
    x (np.array): deglitched x timestream
    cr_indices (np.array): cosmic ray indices
    """
    theta_fine, theta = calculate_theta(zfine, znoise, origin)
    if offres:
        return theta_fine, theta
    # Remove cosmic rays from theta timestream
    print('Need to implement cosmic ray removal')
    cr_indices = np.array([])
    theta_rmvd = theta
    # Calibrate x
    theta_deglitch, poly, theta_range = \
        calibrate_x(ffine, theta_fine, theta, deglitch = deglitch)

    fs_deglitch = np.polyval(poly, theta_deglitch)
    x = 1 - fs_deglitch / fnoise
    znoise_clean = deglitch_timestream(znoise, deglitch)
    return theta_fine, theta, theta_range, poly, x, cr_indices, znoise_clean

def calibrate_x(ffine, theta_fine, theta, deglitch = None, poly_deg = 3, min_cal_points = 5):
    """
    Fit fine scan frequency to phase

    Parameters:
    ffine (np.array): fine scan frequency data in Hz
    theta_fine (np.array): fine scan theta data
    theta (np.array): theta noise timestream data
    deglitch (float or None): threshold for deglitching data, or None to bypass
        deglitching
    poly_deg (int): degree of the polynomial fit
    min_cal_points (int): minimum number of points for the polynomial fit

    Returns:
    theta_deglitch (np.array): deglitched theta data
    poly (np.array): polynomial fit parameters
    theta_range (list): [lower, upper] bounds on theta used for the fit
    """
    if deglitch is not None:
        theta_deglitch = deglitch_timestream(theta, deglitch)
    else:
        theta_deglitch = theta.copy()

    ix0 = np.argmin(abs(min(theta_deglitch) - theta_fine))
    ix1 = np.argmin(abs(max(theta_deglitch) - theta_fine))
    if theta_fine[ix0] > min(theta_deglitch):
        ix0 -= np.sign(ix1 - ix0)
    if theta_fine[ix1] < max(theta_deglitch):
        ix1 += np.sign(ix1 - ix0)
    if ix0 > ix1:
        ix0, ix1 = ix1, ix0
    npoints_missing = min_cal_points + 1 - (ix1 - ix0)
    if npoints_missing > 0:
        half = int(np.ceil(npoints_missing / 2))
        ix0 -= half
        ix1 + half
    if len(theta_fine) < min_cal_points:
        raise Exception('theta_fine must be at least 5 points long')
    if ix0 < 0:
        ix1 -= ix0
        ix0 = 0
    if ix1 > len(theta_fine):
        ix0 -= len(theta_fine) - ix1
        ix1 = len(theta_fine)
    # Fit to data in range
    poly = np.polyfit(theta_fine[ix0:ix1 + 1], ffine[ix0:ix1 + 1],
                      deg = poly_deg)
    theta_range = np.array([theta_fine[ix0], theta_fine[ix1]])

    return theta_deglitch, poly, theta_range

def deglitch_timestream(x, deglitch):
    """
    Replaces points above a certain threshold from data with the mean of the data

    Parameters:
    x (np.array): data from which glitches are removed
    deglitch (float): threshold for glitch removal. points above this threshold
        times the standard deviation of the data are removed
    Returns:
    x_deglitch (np.array): data with glitches removed
    """
    x_mean = np.mean(x)
    x_deglitch = x.copy()

    devs = np.abs(x - x_mean)
    x_std = np.std(devs)
    x_deglitch[devs > deglitch * x_std] = x_mean
    return x_deglitch

def calculate_theta(zfine, znoise, origin):
    """
    Convert an IQ loop and timestream to theta using the origin of the circle

    Parameters:
    zfine (np.array): fine scan complex S21 data
    znoise (np.array): complex S21 noise timestream
    origin (complex): center of the IQ loop

    Returns:
    theta_fine (np.array): values of theta corresponding to the fine scan data
    theta_noise (np.array): theta timestream corresponding the the noise data
    """
    zn_mean = np.mean(znoise)
    # Get x, y basic vectors, where x is the vector that passes through the
    # origin of the circle and the mean of the noise ball, and y is
    # perpendicular to x
    x_complex = (zn_mean - origin)
    x_complex /= np.abs(x_complex)
    y_complex = 1j * x_complex
    x_vec = np.array([np.real(x_complex), np.imag(x_complex)])
    y_vec = np.array([np.real(y_complex), np.imag(y_complex)])
    # Calculate theta of the fine scan
    zfine_vec = np.transpose(np.array([np.real(zfine - origin),
                                       np.imag(zfine - origin)]))
    zfine_z = np.dot(zfine_vec, x_vec) + 1j * np.dot(zfine_vec, y_vec)
    theta_fine = np.angle(zfine_z)
    # Unwrap fine scan theta
    theta_fine = np.unwrap(2 * theta_fine) / 2
    while theta_fine[0] < 0:
        theta_fine += 2 * np.pi
    # Calculate theta of the noise
    noise_vec = np.transpose(np.array([np.real(znoise - origin),
                                       np.imag(znoise - origin)]))
    noise_z = np.dot(noise_vec, x_vec) + 1j * np.dot(noise_vec, y_vec)
    theta_noise = np.angle(noise_z)
    # Make sure the range of theta matches the range of the fine scan
    theta_noise = np.where(theta_noise > 1, theta_noise - 2 * np.pi,
                           theta_noise)

    return theta_fine, theta_noise

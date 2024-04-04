from scipy.stats import binned_statistic
import numpy as np
import pyfftw

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

################################################################################
############################ Binning and filtering #############################
################################################################################
def bin_psd(f, data, nbins = 500, fmin = 3, filter_pt_n = None):
    """
    Bins noise data logarithmically. Optionally filters pulse tubes before
    binning and leaves frequencies below fmin unbinned.

    Parameters:
    f (np.array): frequency data in Hz
    data (list): values (np.array) are 1D arrays of data corresponding to
        f to bin
    nbins (int): number of bins
    fmin (float): minimum frequency to bin. Data below fmin is kept without
        binning
    filter_pt_n (int or None): number of pulse tube harmonics to filter out of
        the data before binning, or None to bypass pulse tube filtering

    Returns:
    binned_data (list): values (np.array) are binned data corresponding to
        each array in input parameter data
    """
    if not type(f) == np.ndarray:
        f = np.array(f)
    for i, x in enumerate(data):
        if not type(x) == np.ndarray:
            data[i] = np.array(x)
    ix = f > 0
    f, data = f[ix], [d[ix] for d in data]
    if filter_pt_n is not None:
        data = [filter_pt(f, d, filter_pt_n) for d in data]
    ix = f < fmin
    f0, data0 = f[ix], [d[ix] for d in data]
    # Create logarithmically spaced bins, and remove bins that don't have data
    bins = np.geomspace(fmin, max(f), nbins)
    bin_counts, _, _ = binned_statistic(f, [], statistic='count', bins = bins)
    bin_counts = np.concatenate([bin_counts, [1]])
    bins  = bins[bin_counts != 0]
    # Bin data, and append unbinned data
    binned_data = binned_statistic(f, data, bins = bins)[0]
    binned_data = [np.concatenate([data0[i],
                            binned_data[i]]) for i in range(len(binned_data))]
    return binned_data

def filter_pt(f, y, n = 20):
    """
    Filter pulse tube spikes out of noise data

    Parameters:
    f (np.array): frequency data in Hz
    y (np.array): data to filter
    n (int): number of pulse tube harmonics to filter

    Returns:
    y_filt (np.array): y data with pulse tube spikes removed
    """
    f0s = [1.39296 * i for i in range(1, n)]
    for f0 in f0s:
        # Get width from typical values
        d = np.interp(f0, [1, 21, 32.1], [0.1, 0.08, 0.01])
        ix = np.where(abs(f - f0) < d)[0]
        N = len(ix)
        if N:
            y[ix] = np.mean([y[ix.min() - N:ix.min()],
                             y[ix.max():ix.max() + N]], axis = 0)
    return y

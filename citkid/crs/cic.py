import numpy as np
def cic2_response(f, fsample):
    '''
    Modified from hidfmux
    Compute the response of the second-stage cascaded-integrator comb (CIC)
    filter, as a function of frequency. This filter defines the shape of each
    channel's frequency response, so the frequencies here are relative to the
    central frequency of the channel.


    Parameters:
    f (array-like): frequencies in Hz within one channel bandwidth
        Must be less than the demodulated channel sampling frequency
    fsample (float): sample frequency in Hz

    Returns:
    response (np.array): cic2 response corresponding to f
    '''
    f = np.asarray(f)
    fir_stage = np.log10(550e6 / (256 * 64 * fsample)) / np.log10(2)
    cic_factor = 6 - fir_stage
    response = np.sinc(f / fsample) ** (cic_factor + fir_stage)
    return response

def apply_cic2_comp_psd(f, psd, fsample, trim = 0.25):
    '''
    Modified from hidfmux
    Compensate for the response of the second-stage cascaded-integrator comb
    (CIC) filter as a function of frequency in a power spectral density
    measurement

    Note: it is not completely correct to apply this correction to power data,
    as any random scatter in the voltage data will be forced positive,
    which may slightly skew the results!

    Because the channel response drops off quickly towards and above Nyquist,
    results will be less trustworthy at high frequencies, and so this trims
    the last trim-percent of the datapoints at high frequency

    Parameters:
    f (array-like): frequencies in Hz within one channel bandwidth.
        Must be less than the demodulated channel sampling frequency.
    psd (array-like): power spectral density values corresponding to f
    fsample (float): sample frequency in Hz
    trim (float): percent length of array to trim off at high frequencies, to
        avoid returning data where numerical uncertainties are higher in the
        compensation function.

    Returns:
    f_trim (np.array): trimmed frequencies
    psd_trim (np.array): trimmed and CIC2 filter compensated PSD
    '''
    warnings.warn(f"This function is an approximation. The exact version will be available in rfmux soon.", UserWarning)
    f, psd = np.asarray(f), np.asarray(psd)
    psd_comp = np.asarray(psd) / (cic2_response(np.asarray(f), fsample)) ** 2
    ix = int(np.ceil((1 - trim) * len(psd_comp)))
    f_trim, psd_trim = f[:ix], psd_comp[:ix]
    return f_trim, psd_trim

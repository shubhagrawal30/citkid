def flag_pulses(x, tsample, pulse_nstd, cr_nstd, amp_window):
    """
    Flag pulses in a timestream. Return indices of cosmic rays, pulses, and
    non-pulse events, and amplitudes of pulses (max value in a window around
    the pulse).
    What are the criteria for pulse vs cr vs nonpulse??

    Parameters:
    x (array-like): fractional frequency shift timestream
    tsample (float): sample rate in seconds
    amp_window (float): window in seconds around peak within which the amplitude
        is extracted from the maximum value of the data

    Returns:
    pulse_indices (np.array): array of indices into x where pulses occur
    cr_indices (np.array): array of indices into x where cosmic rays occur
    nonpulse_indices (np.array): array of indices into x where non-pulses occur
    pulse_amplitudes (np.array): max value in amp_window around each pulse
    cr_amplitudes (np.array): max value in amp_window around each cosmic ray
    nonpulse_amplitudes (np.array): max value in amp_window around each
        non-pulse
    """

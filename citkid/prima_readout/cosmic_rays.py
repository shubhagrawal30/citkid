from scipy.signal import find_peaks
import numpy as np

def remove_cosmic_rays(x, tsample, cr_nstd = 5, time_constant = 1e-3):
    """
    Remove cosmic rays from a timestream using a peak finding algorithm.
    Flags the cosmic rays and sets the data in the timestream equal to the
    average before and after the cosmic ray. In the case of overlapping cosmic
    rays, treats the overlap as a single cosmic ray.

    Parameters:
    x (np.array): timestream with cosmic rays
    tsample (float): sample time of x array
    cr_nstd (float): standard deviation threshold for peak finding
    time_constant(float): cosmic ray decay time constant in seconds

    Returns:
    cr_indices (np.array): array of indices at which cosmic rays were found
    x_rmvd (np.array): timestream with cosmic rays removed
    """
    # Create cr removal parameters from detector time constant
    cr_width_time = time_constant # minimum width of cosmic ray hit
    cr_peak_spacing = time_constant # minimum spacing between peaks
    cr_removal_time = time_constant * 20 # time to remove around peaks
    cr_width = int(cr_width_time / tsample)
    if cr_width < 1:
        cr_width = 1
    height = np.mean(-x) + cr_nstd * np.std(x)
    distance = int(cr_peak_spacing / tsample)
    if distance < 1:
        distance = 1
    # find cosmic rays
    cr_indices, _ = find_peaks(-x, width = cr_width,
                               distance = distance,
                               height = height)
    # remove cosmic rays
    start_offset = int(time_constant / 4 / tsample)
    istarts = cr_indices - start_offset
    i_removal = int(cr_removal_time / tsample)
    iends = istarts + i_removal
    iranges = [[istarts[i], iends[i]] for i in range(len(istarts))]
    # Merge overlapping iranges
    iranges = remove_overlaps(iranges)
    # fix iranges on edge of data
    for index, ir in enumerate(iranges):
        if ir[0] < 0:
            iranges[index][0] = 0
        if ir[1] >= len(x):
            iranges[index][1] = len(x) - 1
    # Average data before and after iranges
    x_rmvd = x.copy()
    for irange in iranges:
        # determine indices of data before and after the data to cut
        ilen = irange[1] - irange[0]
        ibef = [irange[0] - ilen - 1, irange[0] - 1]
        iaft = [irange[1], irange[1] + ilen]
        if ibef[0] < 0:
            ibef[0] = 0
        if iaft[1] > len(x):
            iaft[1] = len(x)
        # Make x_bef and x_aft arrays
        x_bef = x[ibef[0]: ibef[1]]
        x_aft = x[iaft[0]: iaft[1]]
        length_diff = len(x_aft) - len(x_bef)
        if length_diff > 0: # data before is too short
            x_bef = np.concatenate([x_bef, x_aft[:length_diff]])
        elif length_diff < 0: # data before is too short
            x_aft = np.concatenate([x_aft, x_bef[:-length_diff]])
        x_rmvd[irange[0]: irange[1]] = np.mean([x_bef, x_aft],
                                                    axis = 0)
    return cr_indices, x_rmvd

################################################################################
######################### Utility functions ####################################
################################################################################

def remove_overlaps(iranges):
    """
    Given a list of index ranges, return a list where overlapping ranges are
    concatenated. Overlaps are defined by clearances between ranges equal to
    the lengths of the ranges

    Parameter:
    iranges (list): values (list) are [lower, upper] values

    Returns:
    iranges_concat (list): iranges where overlapping ranges are concatenated
    """
    overlaps = find_overlaps(iranges)
    if not len(overlaps):
        return iranges
    # Remove overlaps
    iranges_concat = []
    i = 0
    while i < len(iranges):
        if i in overlaps:
            iranges_concat.append([min(iranges[i] + iranges[i + 1]),
                                   max(iranges[i] + iranges[i + 1])])
            ix = [overlap - 1 for overlap in overlaps]
            i += 2
        else:
            iranges_concat.append(iranges[i])
            i += 1
    # Repeat until done
    return remove_overlaps(iranges_concat)

def find_overlaps(iranges):
    """
    Find overlapping ranges, where overlaps are defined by a clearance between
    ranges equal to the max length of the two ranges

    Parameters:
    iranges (list): values (list) are [lower, upper] values

    Returns:
    overlaps (list): values (int) are indices of overlapping ranges, where the
        two ranges that overlap are iranges[overlaps[i]] and
        iranges[overlaps[i + 1]]
    """
    sorted_ranges = sorted(iranges, key = lambda x: x[0])
    range_lens = [0] + [ir[1] - ir[0] for ir in iranges] + [0]
    range_lens = [ir[1] - ir[0] for ir in iranges]
    required_clearances = list(np.max([range_lens[:-1], range_lens[1:]],
                               axis = 0))
    actual_clearances = []
    for i in range(len(iranges) - 1):
        actual_clearances.append(iranges[i + 1][0] - iranges[i][1])
    overlaps = []
    for i in range(len(actual_clearances)):
        if actual_clearances[i] <= required_clearances[i]:
            overlaps.append(i)
    return overlaps

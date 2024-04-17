import os
import numpy as np

def update_ares_pscale(f, a, a_nl, dbm_change_high = 1, dbm_change_low = 1,
                     a_max = 1000):
    """
    Updates the amplitude of a tone to target a_nl = 0.5 by scaling the output
    power of the RFSoC linearly with a_nl. If a_nl < 0.1 or a_nl > 0.7, shifts
    the output amplitude by a fixed value instead.

    Parameters:
    f (float): frequency in Hz
    a (float): amplitude
    a_nl (float): nonlinearity parameter
    dbm_change_high (float): number of dBm to decrease the power if a_nl > 0.7
    dbm_change_low (float) : number of dBm to increase the power if a_nl < 0.1
        We should explore if we can expand the range
    a_max (float): maximum value of the amplitude

    Returns:
    a_new (float): updated value of a
    """
    dbm = get_dbm(a, f)
    mW_power = 10 ** (dbm / 10)
    if a_nl > 0.7:
        new_dbm = dbm - dbm_change_high
    elif a_nl > 0.1:
        new_dbm = 10 * np.log10(mW_power * 0.5 / a_nl)
    else:
        new_dbm = dbm + dbm_change_low
    a_new = get_rfsoc_power(new_dbm, f)
    if a_new > a_max:
        a_new = a_max
    return a_new
update_ares_pscale = np.vectorize(update_ares_pscale)

def update_ares_addonly(f, a, a_nl, dbm_change_high = 1, dbm_change_low = 1,
                        a_max = 1000):
    """
    Updates the amplitude of a tone to target 0.4 < a_nl < 0.6 by adding or
    subtracting a fixed power in dB.

    Parameters:
    f (float): frequency in Hz
    a (float): amplitude
    a_nl (float): nonlinearity parameter
    dbm_change_high (float): number of dBm to decrease the power if a_nl > 0.6
    dbm_change_low (float) : number of dBm to increase the power if a_nl < 0.4
        We should explore if we can expand the range
    a_max (float): maximum value of the amplitude

    Returns:
    a_new (float): updated value of a
    """
    dbm = get_dbm(a, f)
    mW_power = 10 ** (dbm / 10)
    if a_nl > 0.6:
        new_dbm = dbm - dbm_change_high
    elif a_nl < 0.4:
        new_dbm = dbm + dbm_change_low
    else:
        new_dbm = dbm
    a_new = get_rfsoc_power(new_dbm, f)
    if a_new > a_max:
        a_new = a_max
    return a_new
update_ares_addonly = np.vectorize(update_ares_addonly)

################################################################################
########################## power conversion functions ##########################
################################################################################
cal_directory = os.path.dirname(os.path.realpath(__file__)) + '/cal_data/'
ref_freqs      = np.load(cal_directory + 'ref_freqs.npy')
ref_dbm_pdiffs = np.load(cal_directory + 'ref_dbm_pdiffs.npy')
a_cal          = np.load(cal_directory + 'rfsoc_powers.npy')
dbm_cal        = np.load(cal_directory + 'dbm_powers.npy')

def get_dbm(a, f):
    """
    Converts RFSoC power units to dBm. Uses two interpolations to predict the
    dBm power based on a conversion of RFSoc unit --> dBm power at frequency
    300MHz, plus a correction due to the fact that the RFSoC's output power
    decreases as frequency increases.

    Parameters:
    a (float): A power level in RFSoC units, i.e. what you supply for ares
    f (float): the frequency where the tone will be output

    Returns:
    dbm (float): the predicted output power in dBm
    """
    ix = np.argsort(a_cal)
    dbm = np.interp(a, a_cal[ix], dbm_cal[ix])
    ix = np.argsort(ref_freqs)
    freq_correction = np.interp(f, ref_freqs[ix], ref_dbm_pdiffs[ix])
    dbm += freq_correction
    return dbm

def get_rfsoc_power(dbm, f):
    '''
    Converts power in dBm units to RFSoC units, using the same interpolations
    from get_dbm.

    Parameters:
    dbm (float): power in dBm
    f (float): tone frequency in Hz

    Returns:
    a (float): power in RFSoC units
    '''
    ix = np.argsort(ref_freqs)
    freq_correction = np.interp(f, ref_freqs[ix], ref_dbm_pdiffs[ix])
    dbm -= freq_correction
    ix = np.argsort(dbm_cal)
    a = np.interp(dbm, dbm_cal[ix], a_cal[ix])
    return a

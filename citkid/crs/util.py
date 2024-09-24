import numpy as np 
from hidfmux.core.utils import transferfunctions

def volts_to_dbm(volts):
    """
    Converts voltage in V to power in dBm 

    Parameters:
    volts (float or array-like): voltage value(s) in V 

    Returns:
    dbm (float or np.array): power value(s) in dBm 
    """
    termination = 50 # ohms
    v_rms = volts / np.sqrt(2.)
    watts = v_rms ** 2 / termination
    return 10. * np.log10(watts * 1e3)

def remove_internal_phaseshift(f, z, zcal):
    """
    Corrects measured samples by the corresponding loopback measurement
    
    Parameters:
    f (array-like): frequency array 
    z (array-like): complex S21 data array from the ADC 
    zcal (array-like): complex S21 calibration data array from the carrier 
    
    Returns:
    zcor (np.array): z corrected for the carrier calibration data 
    """ 
    f, z, zcal = np.asarray(f), np.asarray(z), np.asarray(zcal)
    latency = transferfunctions.get_latency() 
    zcal_angle =  np.angle(zcal)
    latency_adjustment = np.pi * (1 - (2 * latency) * (f % (1 / latency)))
    adj = np.exp(- 1j * (zcal_angle + latency_adjustment))
    zcor = z * adj
    return zcor 
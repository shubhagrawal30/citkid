import numpy as np 
import os 
from hidfmux.core.utils import transferfunctions
import scipy.interpolate as interpolate
volts_per_roc = np.sqrt(2) * np.sqrt(50* (10**(-1.75 / 10)) / 1000) / 1880796.4604246316

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

def remove_internal_phaseshift_noise(f, z, ffine, zfine):
    """
    Corrects measured samples by the corresponding loopback measurement for 
    a noise measurement. 
    
    Parameters:
    f (array-like): frequency array 
    z (array-like): complex S21 data array from the ADC 
    ffine (array-like): fine S21 sweep 
    zfine (array-like): 
    
    Returns:
    zcor (np.array): z corrected for the carrier calibration data 
    """ 
    z = remove_internal_phaseshift(f, z, 0 + 0j)
    p = np.angle(np.mean(z, axis = 1))[:, np.newaxis]
    pfine = np.angle(zfine) 
    pcal = np.array([[np.interp(fi, ff, pf)]for fi, ff, pf in zip(f.flatten(), ffine, pfine)])
    z = z * np.exp(1j * (pcal - p)) 
    return z 


def convert_parser_to_z(path, crs_sn, module, ntones):
    """
    Import a parser file and convert the data to complex S21 in V 

    Parameters:
    path (str): path to the parser folder 
    crs_sn (int): CRS serial number 
    module (int): module number

    Returns:
    z (np.array): complex S21 data in V 
    """
    parser_batch_file ='m0%d_raw32'%(module)
    parser_dat = np.fromfile(os.path.join(path, f'serial_{crs_sn:04d}', parser_batch_file), 
                                np.dtype([('i', np.int32), ('q', np.int32)]))
    z = np.array([complex(*pi) for pi in parser_dat])
    z = np.array([z[i::1024] for i in range(ntones)])
    z = z * volts_per_roc / 256 # parser has additional factor of 256
    return z
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

def find_key_and_index(dictionary, j):
    """
    Finds the key in a dictionary where the numpy array (value) contains the integer 'j', 
    and returns both the key and the index of 'j' in that array.

    Parameters:
    dictionary (dict): A dictionary where keys are integers and values are numpy arrays of integers.
    j (int): The integer to search for in the numpy arrays.

    Returns:
    tuple: A tuple (key, index) where 'key' is the dictionary key and 'index' is the position of 'j' in the array.
           If 'j' is not found, returns (None, None).
    """
    for key, value_array in dictionary.items():
        indices = np.where(value_array == j)[0]
        if indices.size > 0:  # Check if 'j' was found
            return key, indices[0]
    return None, None  # Return (None, None) if 'j' is not found in any array

def convert_freq_to_nyq(target_frequency, nyquist_zone, 
                        adc_sampling_rate=5e9):
    """
    Converts an output frequency to the value to send to the DAC given the 
    Nyquist zone 

    Parameters:
    target_frequency (float): target frequency in Hz 
    nyquist_zone (int): Nyquist zone number (1 or 2) 
    adc_sample_rate (float): ADC sample rate in Hz 
    """
    if nyquist_zone == 1:
        return target_frequency 
    elif nyquist_zone == 2:
        return adc_sampling_rate - target_frequency 
    else:
        raise ValueError('nyquist_zone must be in [1, 2]')
    

def convert_freq_from_nyq(target_frequency, nyquist_zone, 
                        adc_sampling_rate=5e9):
    """
    Converts a frequency from the CRS board to the actual frequency given 
    the nyquist zone 

    Parameters:
    target_frequency (float): target frequency in Hz 
    nyquist_zone (int): Nyquist zone number (1 or 2) 
    adc_sample_rate (float): ADC sample rate in Hz 
    """
    if nyquist_zone == 1:
        return target_frequency 
    elif nyquist_zone == 2:
        return adc_sampling_rate - target_frequency
    else:
        raise ValueError('nyquist_zone must be in [1, 2]')
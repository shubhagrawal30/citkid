import numpy as np
import os
import rfmux

def convert_parser_to_z(path, crs_sn, module, ntones):
    """
    Import a parser file and convert the data to complex S21 in V

    Parameters:
    path (str): path to the parser folder
    crs_sn (int): CRS serial number
    module (int): module number
    ntones (int): number of tones

    Returns:
    z (np.array): complex S21 data in V
    """
    parser_batch_file ='m0%d_raw32'%(module)
    parser_dat = np.fromfile(os.path.join(path, f'serial_{crs_sn:04d}',
                                          parser_batch_file),
                                          np.dtype([('i', np.int32),
                                                    ('q', np.int32)]))
    z = np.array([complex(*pi) for pi in parser_dat])
    z = np.array([z[i::1024] for i in range(ntones)])
    z = z * rfmux.core.utils.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
    return z

def find_key_and_index(dictionary, j):
    """
    Finds the key in a dictionary where the numpy array (value) contains the
    integer 'j', and returns both the key and the index of 'j' in that array

    Parameters:
    dictionary (dict): A dictionary where keys are integers and values are numpy
        arrays of integers
    j (int): The integer to search for in the numpy arrays

    Returns:
    tuple: A tuple (key, index) where 'key' is the dictionary key and 'index'
        is the position of 'j' in the array If 'j' is not found, returns
        (None, None)
    """
    for key, value_array in dictionary.items():
        indices = np.where(value_array == j)[0]
        if indices.size > 0:  # Check if 'j' was found
            return key, indices[0]
    return None, None  # Return (None, None) if 'j' is not found in any array

def get_modules(crs, module_indices):
    """
    Gets the subset of crs ReadoutModule objects given the module indices

    Parameters:
    crs (rfmux.CRS): rfmux CRS system module
    module_indices (array-like, int): module indices

    Returns:
    modules (array-like, rfmux.ReadoutModule): crs system object readout modules
        corresponding to the module indices
    """
    modules_generic = rfmux.ReadoutModule.module.in_(module_indices)
    modules = crs.modules.filter(modules_generic)
    return modules

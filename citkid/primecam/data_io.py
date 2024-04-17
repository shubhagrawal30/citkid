import numpy as np

def import_iq_noise(out_directory, file_suffix):
    """
    Imports data from primecam.procedures.take_iq_noise

    Parameters:
    out_directory (str): directory containing the saved data
    file_index (int): file index

    Returns:
    fres_initial (np.array): initial frequency array in Hz
    fres (np.array): noise frequency array in Hz
    ares (np.array): RFSoC amplitude array
    Qres (np.array): resonance Q array for cutting data
    fcal_indices (np.array): calibration tone indices
    frough, zrough (np.array): rough sweep frequency and complex S21 data
    fgain, zgain (np.array): gain sweep frequency and complex S21 data
    ffine, zfine (np.array): fine sweep frequency and complex S21 data
    znoise (np.array): complex S21 noise timestream array
    noise_dt (float): noise sample time in s
    """
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    fres_initial = np.load(out_directory + f'fres_initial{file_suffix}.npy')
    fres = np.load(out_directory + f'fres{file_suffix}.npy')
    ares = np.load(out_directory + f'ares{file_suffix}.npy')
    Qres = np.load(out_directory + f'Qres{file_suffix}.npy')
    fcal_indices = np.load(out_directory + f'fcal_indices{file_suffix}.npy')
    # sweeps
    path = out_directory + f's21_rough{file_suffix}.npy'
    if os.path.exists(path):
        frough, irough, zrough = np.load(path)
        zrough = irough + 1j * qrough
    else:
        frough, zrough = None, None
    path = out_directory + f's21_gain{file_suffix}.npy'
    fgain, igain, zgain = np.load(path)
    zgain = igain + 1j * qgain
    path = out_directory + f's21_fine{file_suffix}.npy'
    ffine, ifine, zfine = np.load(path)
    zfine = ifine + 1j * qfine
    path = out_directory + f'noise{file_suffix}.npy'
    if os.path.exists(path):
        inoise, qnoise = np.load(path)
        znoise = inoise + 1j * qnoise
        noise_dt = np.load(out_directory + f'noise{file_suffix}_tsample.npy' )
    else:
        znoise, noise_dt = None, None
    return fres_initial, fres, ares, Qres, fcal_indices, frough, zrough,\
           fgain, zgain, ffine, zfine, znoise, noise_dt

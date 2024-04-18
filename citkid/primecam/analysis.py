import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm
from ..res.fitter import fit_nonlinear_iq_with_gain
from ..res.gain import fit_and_remove_gain_phase
from ..res.data_io import make_fit_row
from ..util import fix_path, save_fig
from .update_ares import get_dbm
import matplotlib.pyplot as plt

def fit_iq(fgain, zgain, ffine, zfine, fres, ares, Qres, fcal_indices,
           directory, out_directory, power_number, in_atten,
           constant_atten, temperature_index, temperature,
           resonator_indices = None, file_suffix = '',
           extra_fitdata_values = {}, plotq = False, plot_factor = 1,
           overwrite = False, verbose = True):
    """
    Fits all IQ loops in a target scan

    Parameters:
    fgain, zgain (np.array): gain sweep frequency and complex S21 data
    ffine, zfine (np.array): fine sweep frequency and complex S21 data
    fres (np.array): array of resonance frequencies in Hz
    ares (np.array): RFSoC power settings in RFSoC units for each tone, for
        logging
    Qres (np.array): array of Qs corresponding to fres to cut from the fine and
        gain scan data. Spans of f / Q are cut from the gain data and f / 2Q
        are cut from the fine data. Spans of f / 2Q are left in the fine data
        for the tone corresponding to that dataset
    fcal_indices (np.array): calibration tone indices into fres. For these
        tones, only the gain scan is fit
    directory (str): directory containing the data for logging
    out_directory (str): directory to save the plots and data, or None to bypass
        saving data
    power_number (int): power index for logging
    in_atten (np.array): variable input attenuations for logging
    constant_atten (np.array): constant input attenuations for logging. The
        total attenuation between the RFSoC and the device is
        in_atten + constant_atten. Any amplification on the input to the
        cryostat should be taken into account here
    temperature_index (int): temperature index for logging
    temperature (float): temperature in K for logging
    resonator_indices (np.array or None): If np.array, list of resonator
        indices corresponding to each resonator in the target sweep. If
        None, resonator indices are assigned by their index into fres
    file_suffix (str): suffix for file names
    extra_fitdata_values (dict): keys (str) are data column names and values
        (single value or np.array with same length as number of targets) are set
        to that data column
    plotq (bool): If True, plots IQ fits and saves them
    plot_factor (int): for plotting a subset of resonators. Plots every
        plot_factor resonators
    overwrite (bool): if not True, raises an exception if the output data file
        already exists
    verbose (bool): If True, displays a progress bar as data is taken

    Returns:
    data (pd.DataFrame): DataFrame of fit data
    """
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    directory = fix_path(directory)
    if out_directory is not None:
        out_directory = fix_path(out_directory)
        fit_plot_directory = out_directory + 'plots_iq/'
        if not os.path.exists(fit_plot_directory) and plotq:
            os.makedirs(fit_plot_directory)
        out_path = out_directory + f'fitdata{file_suffix}.csv'
        if not overwrite and os.path.exists(out_path):
            raise Exception(f'{out_path} already exists!!!')
    # Split data
    fgains, zgains = split_sweep(fgain, zgain, len(fgain) // len(fres))
    ffines, zfines = split_sweep(ffine, zfine, len(ffine) // len(fres))
    Qres = np.array(Qres, dtype = float)
    Qres[fcal_indices] = np.inf
    # Assign resonator indices if they are not given
    data = pd.DataFrame([])
    if resonator_indices is None:
        resonator_indices = list(range(len(fres)))
    # Iterate through resonators and fit
    pbar = resonator_indices
    if verbose:
        pbar = tqdm(pbar, leave = False)
        pbar.set_description('Fitting IQ Loops')
    for pbar_index, resonator_index in enumerate(pbar):
        plotq_single = ((resonator_index % plot_factor) == 0) and plotq
        ffine, zfine = ffines[pbar_index], zfines[pbar_index]
        fgain, zgain = fgains[pbar_index], zgains[pbar_index]
        fr, Qr = fres[pbar_index], Qres[pbar_index]
        # Cut adjacent resonators from data before fitting
        for index in range(len(fres)):
            fr, Qr = fres[index], Qres[index]
            if len(ffine):
                span = fr / (2 * Qr)
                if abs((fr - np.mean(ffine)) > 1e3):
                    ix0 = ffine < fr - span
                    ix1 = ffine > fr + span
                    ix_fine = ix0|ix1
                    ffine, zfine = ffine[ix_fine], zfine[ix_fine]

        file_prefix = f'Tn{temperature_index}Fn{resonator_index}'
        file_prefix += f'Pn{power_number}{file_suffix}'
        if plotq_single:
            plot_path = fit_plot_directory + file_prefix + '_fit.png'
        else:
            plot_path = ''
        if resonator_index not in fcal_indices:
            # For on-resonance, fit IQ loops
            fitrow, fig = \
                fit_nonlinear_iq_with_gain(fgain, zgain, ffine, zfine, fres,
                                           Qres, plotq = plotq_single,
                                           return_dataframe = True)
            fitrow['plotpath'] = plot_path
        else:
            # for off-resonance, just fit gain
            p_amp, p_phase, z_rmvd, (fig, axs) = \
                fit_and_remove_gain_phase(fgain, zgain, ffine, zfine, fres,
                                          Qres, plotq = plotq_single)
            p = [np.nan] * 7
            res = np.nan
            fitrow = make_fit_row(p_amp, p_phase, p, p, p, res,
                                  plot_path = plot_path)
        if not fig is None:
            save_fig(fig, file_prefix + '_fit', fit_plot_directory)
            plt.close(fig)
        fitrow['resonatorIndex'] = resonator_index
        fitrow['f0'] = np.mean(ffine) # Mean of ffine is the noise frequency
        fitdf = pd.DataFrame(fitrow).T
        data = pd.DataFrame(pd.concat([data, fitdf]))

    data['dataDirectory'] = directory
    data['temperature'] = temperature
    data['temperatureIndex'] = temperature_index

    data['powerNumber'] = power_number
    data['rfsocPower'] = ares
    data['inAtten'] = in_atten
    data['constantAtten'] = constant_atten
    data['outputPower'] = get_dbm(ares, fres)
    data['power'] = data.outputPower - data.inAtten - data.constantAtten
    for key in extra_fitdata_values:
        data[key] = extra_fitdata_values[key]
    data = data.reset_index(drop = True)
    if out_directory is not None:
        data.to_csv(out_path, index = False)
    return data

def split_sweep(f, z, npoints):
    """
    Splits an S21 sweep into subarrays corresponding to each tone

    Parameters:
    f, z (np.array): multi-tone sweep frequency and complex S21 data
    npoints (int): number of points

    Returns:
    fs, zs (list): values (np.array) are single-tone sweep frequency and complex
        S21 data for each tone
    """
    fs = [f[npoints * i: npoints * (i + 1)] for i in range(len(f) // npoints)]
    zs = [z[npoints * i: npoints * (i + 1)] for i in range(len(f) // npoints)]
    return fs, zs

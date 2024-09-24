import os
import numpy as np
from tqdm.auto import tqdm
from ..multitone.ares import update_ares_pscale, update_ares_addonly
from ..multitone.fres import update_fres
from ..multitone.analysis import fit_iq
from ..multitone.plot import plot_ares_opt
from ..util import save_fig
import matplotlib.pyplot as plt
from time import sleep
    
### This is a work in progress -> Don't try to use it yet
async def take_iq_noise(inst, fres, ares, qres, fcal_indices, out_directory, file_suffix,
                  noise_time = 200, take_noise = False,
                  npoints_fine = 600, npoints_gain = 100, npoints_rough = 300, nsamps = 10,
                  take_rough_sweep = False, fres_update_method = 'distance', fir_stage = 6):
    """
    Takes multitone IQ sweeps and noise.

    Parameters:
    inst (multitone instrument): initialized multitone instrument class, with
        'sweep', 'write_tones', and 'capture_noise' methods
    fres (np.array): array of resonance frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of span factors for cutting out of adjacent datasets.
        Resonances should span fres / qres
    fcal_indices (np.array): indices (into fres, ares, qres) of calibration tones
    out_directory (str): directory to save the data
    file_suffix (str): suffix for file names
    noise_time (float or None): noise timestream length in seconds
    if_bw (float): IF bandwidth. 1 / if_bw is the averaging time per data point
        in the IQ loops
    fine_bw (float): fine sweep bandwidth in Hz. Gain bandwidth is 10 X fine
        bandwidth
    rough_bw (float): rough sweep bandwidth in Hz
    npoints_fine (int): number of points per resonator in the fine sweep
    npoints_gain (int): number of points per resonator in the gain sweep
    npoints_rough (int): number of points per resonator in the rough sweep
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies
    fres_update_method (str): method for updating the tone frequencies, if
        take_rough_sweep is True. See .fres.update_fres for methods
    nnoise_timestreams (int): number of noise timestreams to take sequentially.
        Set to 0 to bypass noise acquisition
    fir_stage (int): fir_stage frequency downsampling factor.
            6 ->   596.05 Hz 
            5 -> 1,192.09 Hz 
            4 -> 2,384.19 Hz, might crash 
            3 -> will definetely crash 
    """
    data_path = 'tmp/parser_data_00/'
    if os.path.exists(data_path) and take_noise:
        raise FileExistsError(f'{data_path} already exists')
    os.makedirs(out_directory, exist_ok = True)
    fres = np.asarray(fres, dtype = float)
    ares= np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    fcal_indices = np.asarray(fcal_indices, dtype = int)
    spans = fres / qres
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    if take_rough_sweep:
        np.save(out_directory + f'fres_initial{file_suffix}.npy', fres)
    np.save(out_directory + f'ares{file_suffix}.npy', ares)
    np.save(out_directory + f'qres{file_suffix}.npy', qres)
    np.save(out_directory + f'fcal_indices{file_suffix}.npy',
            fcal_indices)
    # write initial target comb
    await inst.write_tones(1, fres, ares)
    # rough sweep
    if take_rough_sweep:
        filename = f's21_rough{file_suffix}.npy'
        f, z = await inst.sweep_qres(1, fres, ares, qres, npoints = npoints_rough, 
                                     nsamps = nsamps, verbose = True, pbar_description = 'Rough sweep')
        np.save(out_directory + filename, [f, np.real(z), np.imag(z)])
        fres = update_fres(f, z, fres, spans, fcal_indices,
                            method = fres_update_method)
        await inst.write_tones(1, fres, ares)
    np.save(out_directory + f'fres{file_suffix}.npy', fres)

    # Gain Sweep
    filename = f's21_gain{file_suffix}.npy'
    f, z = await inst.sweep_qres(1, fres, ares, qres / 10, npoints = npoints_gain, nsamps = nsamps,
                                   verbose = True, pbar_description = 'Gain sweep')
    np.save(out_directory + filename, [f, np.real(z), np.imag(z)])

    # Fine Sweep
    filename = f's21_fine{file_suffix}.npy'
    f, z = await inst.sweep_qres(1, fres, ares, qres, npoints = npoints_fine, nsamps = nsamps,
                                   verbose = True, pbar_description = 'Fine sweep')
    np.save(out_directory + filename, [f, np.real(z), np.imag(z)])

    # Noise
    if take_noise:
        fsample_noise = 625e6 / (256 * 64 * 2**fir_stage)
        filename = f'noise{file_suffix}_tsample.npy'
        np.save(out_directory + filename, 1 / fsample_noise)
    
        filename = f'noise{file_suffix}_00.npy'
        z = await inst.capture_noise(1, fres, ares, noise_time, f, z, fir_stage = fir_stage,
                                parser_loc='/home/daq1/github/citkid/citkid/crs/parser',
                                interface='enp2s0', delete_parser_data = True)
        np.save(out_directory + filename, [np.real(z), np.imag(z)])


# Haven't started adapting this one yet
async def optimize_ares(inst, out_directory, fres, ares, qres, fcal_indices, 
                        dbm_max = -50, a_target = 0.5, n_iterations = 10, n_addonly = 3,
                        fres_update_method = 'distance',
                        npoints_gain = 50, npoints_fine = 400, plot_directory = None,
                        verbose = False, nsamps = 10, dbm_change_pscale = 2,
                        dbm_change_addonly = 1, addonly_threshold = 0.2):
    """
    Optimize tone powers using by iteratively fitting IQ loops and using a_nl
    of each fit to scale each tone power

    Parameters:
    inst (citkid.primecam.instrument.RFSOC): RFSOC instance
    fres (np.array): array of center frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    qres (np.array): array of resonators Qs for cutting data. Resonances should
        span fres / qres
    fcal_indices (np.array): calibration tone indices
    max_dbm (float): maximum power per tone in dBm
    a_target (float): target value for a_nl. Must be in range (0, 0.77]
    n_iterations (int): total number of iterations
    n_addonly (int): number of iterations at the end to optimize using
        update_ares_addonly. Iterations before these use update_ares_pscale
    fine_bw (float): fine sweep bandwidth in MHz. See take_iq_noise
    fres_update_method (str): method for updating frequencies. See update_fres
    npoints_gain (int): number of points in the gain sweep
    npoints_fine (int): number of points in the fine sweep
    plot_directory (str or None): path to save histograms as the optimization is
        running. If None, doesn't save plots
    verbose (bool): if True, displays a progress bar of the iteration number
    N_accums (int): number of accumulations for the target sweeps
    threshold (float): optimization will occur within (1-threshold) and 
        (1+threshold) of the target during the addonly phase of power optimization
    """
    if plot_directory is not None:
        os.makedirs(plot_directory, exist_ok = True)
    fres, ares, qres = np.array(fres), np.array(ares), np.array(qres)
    pbar0 = list(range(n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices]
    a_nls = []
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('sweeping')
        file_suffix = f'{idx0:02d}'
        await take_iq_noise(inst, fres, ares, qres, fcal_indices, out_directory, file_suffix,
                            take_noise = False, take_rough_sweep = False, npoints_gain = npoints_gain,
                            npoints_fine = npoints_fine, nsamps = nsamps)

        # Fit IQ loops
        if verbose:
            pbar0.set_description('fitting')
        data = fit_iq(out_directory, None, file_suffix, 0, 0, 0, 0, 0, plotq = False, verbose = False)
        a_nl = np.array(data.sort_values('resonatorIndex').iq_a, dtype = float)
        if len(a_nls):
            a_nl[a_nl == np.nan] = a_nls[-1][a_nl == np.nan]
        else:
            a_nl[a_nl == np.nan] = 2
        a_nls.append(a_nl)
        np.save(out_directory + f'a_nl_{file_suffix}.npy', a_nl)
        if plot_directory is not None:
            fig_hist, fig_opt = plot_ares_opt(a_nls, fcal_indices)
            save_fig(fig_hist, 'ares_hist', plot_directory)
            save_fig(fig_opt, 'ares_opt', plot_directory)
        # Update ares
        if idx0 <= n_addonly:
            ares[fit_idx] = update_ares_pscale(fres[fit_idx], ares[fit_idx],
                                           a_nl[fit_idx], a_target = a_target,
                                           dbm_max = dbm_max, dbm_change_high = dbm_change_pscale,
                                           dbm_change_low = dbm_change_pscale)
        else:
            ares[fit_idx] = update_ares_addonly(fres[fit_idx], ares[fit_idx],
                                                a_nl[fit_idx],
                                                a_target = a_target,
                                                dbm_max = dbm_max,
                                                dbm_change_high = dbm_change_addonly,
                                                dbm_change_low = dbm_change_addonly,
                                                threshold = addonly_threshold)
        # update fres
        f, i, q = np.load(out_directory + f's21_fine_{file_suffix}.npy')
        fres = update_fres(f, i + 1j * q, fres, qres,
                           fcal_indices = fcal_indices, method = fres_update_method)
        # for the last iteration, save the updated ares list
        if idx0 == len(fres) - 1:
            np.save(out_directory + f'ares_{idx0 + 1:02d}', ares)

################################################################################
######################### Utility functions ####################################
################################################################################
def make_cal_tones(fres, ares, qres, max_n_tones = 1000,
                   resonator_indices = None, fcal_power = -55):
    '''
    Adds calibration tones to the given resonator list. Fills in largest spaces
    between resonators, up to max_n_tones.

    Parameters:
    fres (np.array): frequency array in Hz
    ares (np.array): amplitude array in Hz
    qres (np.array): span factor array
    max_n_tones (int): maximum number of tones after adding cal tones
    resonator_indices (np.array or None): resonator indices corresponding to
        fres
    fcal_power (float): calibration tone power, in the same units as ares

    Returns:
    fres, ares, qres (np.array): frequency, amplitude, and span factor arrays
        with calibration tones added
    fcal_indices (np.array): calibration tone indices into fres, ares, qres
    new_resonator_indices (np.array): new resonator index list with the new
        calibration tones. Calibration tone resonator indices are negative
    '''
    fres = np.asarray(fres, dtype = float)
    ares = np.asarray(ares, dtype = float)
    qres = np.asarray(qres, dtype = float)
    ix = np.argsort(fres)
    fres, ares, qres = fres[ix], ares[ix], qres[ix]
    if resonator_indices is not None:
        resonator_indices = resonator_indices[ix]
    if resonator_indices is None:
        resonator_indices = np.asarray(range(len(fres)))
    new_resonator_indices = np.asarray(resonator_indices, dtype = int)

    ix = np.flip(np.argsort(np.diff(fres)))[:max_n_tones - len(fres)]
    fcal = np.sort([np.mean(fres[i:i+2]) for i in ix])
    fcal_indices = np.searchsorted(fres, fcal)
    fcal_indices += np.asarray(range(len(fcal_indices)), dtype = int)
    for fcal_index, fres_index in enumerate(fcal_indices):
        fres = np.insert(fres, fres_index, fcal[fcal_index])
        ares = np.insert(ares, fres_index, fcal_power)
        qres = np.insert(qres, fres_index, np.inf)
        new_index = -fcal_index
        new_resonator_indices = np.insert(new_resonator_indices, fres_index,
                                          new_index)
    return fres, ares, qres, fcal_indices, new_resonator_indices

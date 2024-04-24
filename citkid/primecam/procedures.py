import numpy as np
from tqdm.auto import tqdm
from .update_ares import get_rfsoc_power, update_ares_pscale, update_ares_addonly
from .update_fres import update_fres
from .data_io import import_iq_noise
from .analysis import fit_iq
from .plot import plot_ares_opt
from ..util import save_fig
import os 

def take_iq_noise(rfsoc, fres, ares, Qres, fcal_indices, file_suffix,
                  noise_time = 200, fine_bw = 0.2, rough_bw = 0.2,
                  take_rough_sweep = False, fres_update_method = 'distance',
                  npoints_rough = 300, npoints_gain = 100, npoints_fine = 600,
                  nnoise_timestreams = 1):
    """
    Takes IQ sweeps and noise. The LO frequency must already be set.

    Parameters:
    rfsoc (citkid.primecam.instrument.RFSOC): RFSOC instance
    fres (np.array): array of center frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    Qres (np.array): array of resonators Qs for cutting data. Resonances should
        span fres / Qres
    fcal_indices (np.array): indices into fres of calibration tones
    file_suffix (str): suffix for file names
    noise_time (float or None): noise timestream length in seconds, or None to 
        bypass noise acquisition 
    fine_bw (float): fine sweep bandwidth in MHz. Gain bandwidth is 10 X fine
        bandwidth
    rough_bw (float): rough sweep bandwidth in MHz
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies
    fres_update_method (str): method for updating the tone frequencies. 'minS21'
        for the minimum of |S21|, 'distance' for the point of furthest distance
        in IQ space from the off-resonance point, or 'spacing' for the point
        of largest spacing in IQ space
    nnoise_timestreams (int): number of noise timestreams to take sequentially 
    """
    fres, ares, Qres = np.array(fres), np.array(ares), np.array(Qres)
    if file_suffix != '':
        file_suffix = '_' + file_suffix
    if take_rough_sweep:
        np.save(rfsoc.out_directory + f'fres_initial{file_suffix}.npy', fres)
    np.save(rfsoc.out_directory + f'ares{file_suffix}.npy', ares)
    np.save(rfsoc.out_directory + f'Qres{file_suffix}.npy', Qres)
    np.save(rfsoc.out_directory + f'fcal_indices{file_suffix}.npy',
            fcal_indices)
    # write initial target comb
    rfsoc.write_targ_comb_from_custom(fres, ares, pres = None)
    # rough sweep
    if take_rough_sweep:
        filename = f's21_rough{file_suffix}.npy'
        npoints = npoints_rough
        bw = rough_bw 
        rfsoc.target_sweep(filename, npoints = npoints, bandwidth = bw)
        f, i, q = np.load(rfsoc.out_directory + filename)
        z = i + 1j * q
        fres = update_fres(f, z, npoints = npoints,
                               fcal_indices = fcal_indices,
                               method = fres_update_method)
        rfsoc.write_targ_comb_from_custom(fres, ares, pres = None)
    np.save(rfsoc.out_directory + f'fres{file_suffix}.npy', fres)

    # Gain Sweep
    filename = f's21_gain{file_suffix}.npy'
    npoints = npoints_gain
    bw = 10 * fine_bw
    rfsoc.target_sweep(filename, npoints = npoints, bandwidth = bw)

    # Fine Sweep
    filename = f's21_fine{file_suffix}.npy'
    npoints = npoints_fine
    bw = fine_bw
    rfsoc.target_sweep(filename,  npoints = npoints, bandwidth = bw)

    # Noise
    if noise_time is not None:
        if nnoise_timestreams == 1:
            filename = f'noise{file_suffix}_00.npy'
            rfsoc.capture_save_noise(noise_time, filename)
        else:
            for nindex in range(nnoise_timestreams):
                filename = f'noise{file_suffix}_{nindex:02d}.npy'
                rfsoc.capture_save_noise(noise_time, filename)

def make_cal_tones(fres, ares, Qres, max_n_tones = 1000):
    '''
    Adds calibration tones to the given resonator list. Fills in largest spaces
    between resonators, up to max_n_tones

    Parameters:
    fres, ares, Qres (np.array): frequency, amplitude, and Q arrays
    max_n_tones (int): maximum number of tones

    Returns:
    fres, ares, Qres (np.array): frequency, amplitude, and Q arrays with
        calibration tones added
    fcal_indices (np.array): calibration tone indices
    '''
    Qres = [float(Q) for Q in Qres]
    ix = np.flip(np.argsort(np.diff(fres)))[:max_n_tones - len(fres)]
    fcal = np.sort([np.mean(fres[i:i+2]) for i in ix])
    fres = np.sort(np.concatenate([fres, fcal]))
    fcal_indices = [np.where(abs(fres - fcali) < 1)[0][0] for fcali in fcal]
    for ix in fcal_indices:
        ares = np.insert(ares, ix, 260)
        Qres = np.insert(Qres, ix, np.inf)
    return fres, ares, Qres, fcal_indices

def optimize_ares(rfsoc, fres, ares, Qres, fcal_indices, max_dbm = -50,
                  n_iterations = 10, n_addonly = 3, fine_bw = 0.2, 
                  fres_update_method = 'distance', npoints_gain = 50, npoints_fine = 400,
                  plot_directory = None, verbose = False):
    """
    Optimize tone powers using by iteratively fitting IQ loops and using a_nl
    of each fit to scale each tone power

    Parameters:
    rfsoc (citkid.primecam.instrument.RFSOC): RFSOC instance
    fres (np.array): array of center frequencies in Hz
    ares (np.array): array of amplitudes in RFSoC units
    Qres (np.array): array of resonators Qs for cutting data. Resonances should
        span fres / Qres
    fcal_indices (np.array): calibration tone indices
    max_dbm (float): maximum power per tone in dBm
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
    """
    if plot_directory is not None:
        os.makedirs(plot_directory, exist_ok = True)
    fres, ares, Qres = np.array(fres), np.array(ares), np.array(Qres)
    pbar0 = list(range(n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices]
    a_max = get_rfsoc_power(max_dbm, np.mean(fres))
    a_nls = []
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('sweeping')
        file_suffix = f'{idx0:02d}'
        take_iq_noise(rfsoc, fres, ares, Qres, fcal_indices, file_suffix,
                      noise_time = None, fine_bw = fine_bw,
                      take_rough_sweep = False, npoints_gain = npoints_gain,
                      npoints_fine = npoints_fine)
        fres_initial, fres, ares, Qres, fcal_indices, frough, zrough,\
               fgain, zgain, ffine, zfine, znoise, noise_dt =\
        import_iq_noise(rfsoc.out_directory, file_suffix)
        # Fit IQ loops
        if verbose:
            pbar0.set_description('fitting')
        data =\
        fit_iq(fgain, zgain, ffine, zfine, fres, ares, Qres, fcal_indices, '',
               None, 0, 0, 0, 0, 0, file_suffix = file_suffix,
               plotq = False, verbose = False)
        a_nl = np.array(data.sort_values('resonatorIndex').iq_a, dtype = float)
        a_nls.append(a_nl)
        np.save(rfsoc.out_directory + f'a_nl_{file_suffix}.npy', a_nl)
        if plot_directory is not None:
            fig_hist, fig_opt = plot_ares_opt(a_nls, fcal_indices)
            save_fig(fig_hist, 'ares_hist', plot_directory)
            save_fig(fig_opt, 'ares_opt', plot_directory)
        # Update ares
        if idx0 <= n_addonly:
            ares[fit_idx] = update_ares_pscale(fres[fit_idx], ares[fit_idx],
                                           a_nl[fit_idx], a_max = a_max,
                                           dbm_change_high = 2,
                                           dbm_change_low = 2)
        else:
            ares[fit_idx] = update_ares_addonly(fres[fit_idx], ares[fit_idx],
                                                a_nl[fit_idx], a_max = a_max,
                                                dbm_change_high = 1,
                                                dbm_change_low = 1)
        # update fres
        f, i, q = np.load(rfsoc.out_directory + f's21_fine_{file_suffix}.npy')
        fres = update_fres(f, i + 1j * q, len(f) // len(fres), 
                           fcal_indices = fcal_indices, method = fres_update_method,
                        cut_other_resonators =True, fres = fres, Qres = Qres)
        # for the last iteration, save the updated ares list
        if idx0 == len(fres) - 1:
            np.save(rfsoc.out_directory + f'ares_{idx0 + 1:02d}', ares)

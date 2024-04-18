import numpy as np
from tqdm.auto import tqdm
from .update_ares import get_rfsoc_power
from .update_fres import update_fres
from .data_io import import_iq_noise
from .analysis import fit_iq

def take_iq_noise(rfsoc, fres, ares, Qres, fcal_indices, file_suffix,
                  take_noise = False, noise_time = 200, bw_factor = 1,
                  take_rough_sweep = False, fres_update_method = 'distance'):
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
    take_noise (bool): if True, takes noise data. Otherwise, only takes S21
        sweeps
    noise_time (float): noise timestream length in seconds
    bw_factor (float): factor by which all bandwidths are multiplied. Default
        bandwidths for bw_factor = 1 are 0.2 GHz rough, 0.2 GHz fine, 2 GHz gain
    take_rough_sweep (bool): if True, first takes a rough sweep and optimizes
        the tone frequencies
    fres_update_method (str): method for updating the tone frequencies. 'minS21'
        for the minimum of |S21|, 'distance' for the point of furthest distance
        in IQ space from the off-resonance point, or 'spacing' for the point
        of largest spacing in IQ space
    """
    if file_suffix != '':
        file_suffix = '_' + file_suffix
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
        npoints = 300
        bw = 0.2 * bw_factor
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
    npoints = 100
    bw = 2 * bw_factor
    rfsoc.target_sweep(filename, npoints = npoints, bandwidth = bw)

    # Fine Sweep
    filename = f's21_fine{file_suffix}.npy'
    npoints = 600
    bw = 0.2 * bw_factor
    rfsoc.target_sweep(filename,  npoints = npoints, bandwidth = bw)

    # Noise
    if take_noise:
        filename = f'noise{file_suffix}.npy'
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
                  n_iterations = 10, n_addonly = 3, bw_factor = 1,
                  fres_update_method = 'distance', plotq = False,
                  plot_directory, verbose = False):
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
    bw_factor (float): bandwidth factor for IQ loops (see take_iq_noise)
    fres_update_method (str): method for updating frequencies. See update_fres
    plotq (bool): if True, plots and saves histograms as the optimization is
        running
    verbose (bool): if True, displays a progress bar of the iteration number
    """
    pbar0 = list(range(n_iterations))
    if verbose:
        pbar0 = tqdm(pbar0, leave = False)
    fit_idx = [i for i in range(len(fres)) if i not in fcal_indices]
    a_max = get_rfsoc_power(max_dbm, fres)
    a_nls = []
    for idx0 in pbar0:
        if verbose:
            pbar0.set_description('sweeping')
        file_suffix = f'{idx0:02d}'
        take_iq_noise(rfsoc, fres, ares, Qres, fcal_indices, file_suffix,
                      take_noise = False, noise_time = 200,
                      bw_factor = bw_factor, take_rough_sweep = False)
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
        a_nl = np.array(data.sort_values('resonatorIndex').iq_a)
        a_nl.append(a_nls)
        np.save(rfsoc.out_directory + f'a_nl{file_suffix}.npy', a_nl)
        if plotq:
            fig, ax = plot_ares_opt(a_nls)
            save_fig(fig, 'ares_hist', plot_directory)

        # Update ares
        if idx0 <= n_addonly:
            ares[fit_idx] = update_ares_pscale(fres[fit_idx], ares[fit_idx],
                                           a_nl[fit_idx], a_max = a_max,
                                           dbm_change_high = dbm_change_high,
                                           dbm_change_low = dbm_change_low)
        else:
            ares[fit_idx] = update_ares_addonly(fres[fit_idx], ares[fit_idx],
                                                a_nl[fit_idx], a_max = a_max,
                                                dbm_change_high = 1,
                                                dbm_change_low = 1)
        # for the last iteration, save the updated ares list
        if idx0 == len(fres) - 1:
            np.save(rfsoc.out_directory + f'ares{idx0 + 1:02d}', ares)

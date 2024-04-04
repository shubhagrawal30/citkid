import pandas as pd
import numpy as np
from ..util import fix_path, save_fig

def save_psd(psd_onres, psd_offres, timestream_onres, timestream_offres,
             cr_indices, theta_range, poly, xcal_data, figs, dt, dt_offres,
             out_directory, plot_directory, prefix = '', iq_fit_row = None):
     """
     Saves the output of .analysis.compute_psd

     Parameters:
     psd_onres (tuple): on-resonance psd data, or None
         f_psd (np.array): frequency array for PSDs
         spar  (np.array): PSD of noise parallel to IQ loop
         sper  (np.array): PSD of noise perpendicular to IQ loop
         sxx   (np.array): PSD of fractional frequency noise
     psd_offres (tuple): off-resonance psd data, or None
         f_psd_offres (np.array): frequency array for PSDs
         spar_offres  (np.array): PSD of noise parallel to IQ loop
         sper_offres  (np.array): PSD of noise perpendicular to IQ loop
     timestream_onres (tuple): on-resonance timestream data, or None
         theta (np.array): theta timestream data, with no cosmic ray removal or
             deglitching
         x (np.array): fractional frequency shift timestream with cosmic rays
             removed and deglitching
     timestream_offres (tuple): off-resonance timestream data, or None
         theta_offres (np.array): theta timestream data, with no cosmic ray
             removal or deglitching
     cr_indices (np.array): indices into theta where cosmic rays were found
     theta_range (list): [lower, upper] range of theta over which x vs theta was
         fit to calibrate x
     poly (np.array): x vs theta polynomial fit parameters
     xcal_data (tuple): x vs theta calibration data. Not cut to theta_range
         x (np.array): fractional frequency shift data
         theta (np.array): theta data
     figs (tuple): plots
         fig_cal (plt.figure): plot of the IQ loop with noise balls and
             calibration
         fig_psd (plt.figure): plot of the PSDs
         fig_timestream (plt.figure): plot of the timestream data
     dt (float): sample time of the on-resonance noise timestream in s
     dt_offres (float): sample time of the off-resonance noise timestream in s
     out_directory (str): directory to save the data
     plot_directory (str): directory to save the plots
     prefix (str): prefix for the file names
     iq_fit_row (pd.Series or None): row of IQ fit data to which the noise fit
        data is added, or None to create a new row

     Returns:
     iq_fit_row (pd.Series): fit row with noise data appended, or new fit row
        with noise data
     """
     out_directory = fix_path(out_directory)
     plot_directory = fix_path(plot_directory)
     if iq_fit_row is None:
         iq_fit_row = pd.Series([], dtype = object)
     if len(prefix):
         prefix += '_'
     predir = out_directory + prefix
     # Save psds
     if psd_onres[0] is not None:
         path = predir + 'psd.npy'
         np.save(path, psd_onres)
         iq_fit_row['psdPath'] = path
     if psd_offres[0] is not None:
         path = predir + 'psd_offres.npy'
         np.save(path, psd_offres)
         iq_fit_row['psdOffPath'] = path
     if timestream_onres[0] is not None:
         path = predir + 'timestream.npy'
         np.save(path, timestream_onres)
         iq_fit_row['timestreamPath'] = path
         path = predir + 'timestream_dt.npy'
         np.save(path, dt)
         iq_fit_row['timestream_dt'] = dt
     if timestream_offres[0] is not None:
         path = predir + 'timestream_offres.npy'
         np.save(path, timestream_offres)
         iq_fit_row['timestreamOffPath'] = path
         path = predir + 'timestream_offres_dt.npy'
         np.save(path, dt_offres)
         iq_fit_row['timestream_dtOff'] = dt_offres
     if cr_indices is not None:
         path = predir + 'cr_indices.npy'
         np.save(path, cr_indices)
         iq_fit_row['crIndexPath'] = path
     if theta_range is not None:
         iq_fit_row['thetaMin'] = min(theta_range)
         iq_fit_row['thetaMax'] = max(theta_range)
     if poly is not None:
         path = predir + 'xpoly.npy'
         np.save(path, poly)
         for i, pi in enumerate(poly):
             iq_fit_row[f'xpoly_{i}'] = pi
     if xcal_data[0] is not None:
         path = predir + 'xcal_data.npy'
         np.save(path, xcal_data)
         iq_fit_row['xcalPath'] = path
     for fig, name in zip(figs, ['cal', 'psd', 'timestream']):
         if fig is not None:
             path = plot_directory + prefix + name + '.png'
             save_fig(fig, prefix + name, plot_directory, ftype = 'png')
             iq_fit_row[name + 'FitPath'] = path
     return iq_fit_row

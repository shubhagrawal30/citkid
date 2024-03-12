import pandas as pd
import numpy as np
nonlinear_iq_p_names = ['fr', 'Qr', 'amp', 'phi', 'a', 'i0', 'q0', 'tau']

def make_iq_fit_row(p_amp, p_phase, p0, popt, popt_err, res, plot_path = '',
                    prefix = 'iq'):
    """
    Wraps the output of nonlinear_iq fitting into a pd.Series instance

    Parameters:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB
    p_phase (np.array): 1st-order polynomial fit parameters to phase
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter
    popt_err (np.array): standard errors on fit parameters
    res (float): fit residuals
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'iq'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(nonlinear_iq_p_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(nonlinear_iq_p_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(nonlinear_iq_p_names, popt_err):
        row[prefix + key + '_err'] = pi
    for i, pi in enumerate(p_amp):
        row[prefix + f'pamp_{i:02d}'] = pi
    for i, pi in enumerate(p_phase):
        row[prefix + f'pphase_{i:02d}'] = pi
    row[prefix + 'res'] = res
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_iq_fit_row(row, prefix = 'iq'):
    """
    Performs nearly the inverse function of make_iq_fit_row

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data

    Returns:
    p_amp (np.array): 2nd-order polynomial fit parameters to dB
    p_phase (np.array): 1st-order polynomial fit parameters to phase
    p0 (np.array): fit parameter guess.
    popt (np.array): fit parameters. See p0 parameter
    popt_err (np.array): standard errors on fit parameters
    res (float): fit residuals
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'iq'
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in nonlinear_iq_p_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in nonlinear_iq_p_names:
        popt.append(row[prefix + key])
    popt_err = []
    for key in nonlinear_iq_p_names:
        popt_err.append(row[prefix + key + '_err'])
    p_amp = []
    s = prefix + 'pamp' + '_'
    indices = [int(key.replace(s, '')) for key in row.keys() if s in key]
    for index in range(max(indices) + 1):
        key = s + f'{index:02d}'
        p_amp.append(row[key])
    p_phase = []
    s = prefix + 'pphase' + '_'
    indices = [int(key.replace(s, '')) for key in row.keys() if s in key]
    for index in range(max(indices) + 1):
        key = s + f'{index:02d}'
        p_phase.append(row[key])
    res = row[prefix + 'res']
    plot_path = row[prefix + 'plotpath']
    p_amp, p_phase = np.array(p_amp), np.array(p_phase)
    p0, popt, popt_err = np.array(p0), np.array(popt), np.array(popt_err)
    return p_amp, p_phase, p0, popt, popt_err, res, plot_path

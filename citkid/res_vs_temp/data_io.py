import pandas as pd
import numpy as np

fr_vs_temp_names = ['fr0', 'D', 'alpha', 'Tc']
fr_vs_temp_labels = [r'$f_r^0$', r'$D$', r'$\alpha$', r'$T_c$']

def make_fit_row(p0, popt, perr, plot_path = '', prefix = 'fr_vs_temp'):
    """
    Wraps the output of fit_fr_vs_temp fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'fr_vs_temp'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(fr_vs_temp_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(fr_vs_temp_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(fr_vs_temp_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row(row, prefix = 'fr_vs_temp'):
    """
    Performs the inverse function of make_fit_row

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'fr_vs_temp'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in fr_vs_temp_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in fr_vs_temp_names:
        popt.append(row[prefix + key])
    perr = []
    for key in fr_vs_temp_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    return p0, popt, perr, plot_path

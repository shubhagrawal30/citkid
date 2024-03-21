import pandas as pd
import numpy as np

responsivity_int_names = ['R0', 'P0', 'c']
repsonsivity_int_labels = [r'$R_0$', r'$P_0$', r'$c$']

def make_fit_row(p0, popt, perr, f1, f0, f0err, plot_path = '', prefix = 'resp'):
    """
    Wraps the output of fit_responsivity_int fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    f1 (float): frequency at P = 0 that was assumed when creating x
    f0 (float): frequency at P = 0 from fit
    f0err (float): uncertainty in f0
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'resp'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(responsivity_int_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(responsivity_int_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(responsivity_int_names, perr):
        row[prefix + key + '_err'] = pi
    row['resp_f1'] = f1
    row['resp_f0'] = f0
    row['resp_f0_err'] = f0err
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row(row, prefix = 'resp'):
    """
    Performs the inverse function of make_fit_row

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'resp'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    f1 (float): frequency at P = 0 that was assumed when creating x
    f0 (float): frequency at P = 0 from fit
    f0err (float): uncertainty in f0
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in responsivity_int_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in responsivity_int_names:
        popt.append(row[prefix + key])
    perr = []
    for key in responsivity_int_names:
        perr.append(row[prefix + key + '_err'])
    f1 = row['resp_f1']
    f0 = row['resp_f0']
    f0err = row['resp_f0_err']
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    return p0, popt, perr, f1, f0, f0err, plot_path

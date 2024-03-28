import pandas as pd
import numpy as np

fr_vs_temp_names = ['fr0', 'D', 'alpha', 'Tc']
fr_vs_temp_labels = [r'$f_r^0$', r'$D$', r'$\alpha$', r'$T_c$']
fr_vs_temp_notls_names = ['fr0', 'alpha', 'Tc']
fr_vs_temp_notls_labels = [r'$f_r^0$', r'$\alpha$', r'$T_c$']
fr_vs_temp_tls_names = ['fr0', 'D']
fr_vs_temp_tls_labels = [r'$f_r^0$', r'$D$']

Q_vs_temp_notls_names = ['fr0', 'alpha', 'Tc', 'delta_z']
Q_vs_temp_notls_labels = [r'$f_r^0$', r'$\alpha$', r'$T_c$', r'$\delta_z$']


def make_fit_row_fr_vs_temp(p0, popt, perr, gamma, plot_path = '',
                            prefix = 'fr_vs_temp'):
    """
    Wraps the output of fit_fr_vs_temp fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
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
    row[prefix + 'gamma'] = gamma
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_fr_vs_temp(row, prefix = 'fr_vs_temp'):
    """
    Performs the inverse function of make_fit_row_fr_vs_temp

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'fr_vs_temp'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
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
    gamma = row[prefix + 'gamma']
    return p0, popt, perr, gamma, plot_path

################################################################################
######################### Funcs without TLS component ##########################
################################################################################
def make_fit_row_fr_vs_temp_notls(p0, popt, perr, gamma, plot_path = '',
                                  prefix = 'fr_vs_temp_notls'):
    """
    Wraps the output of fit_fr_vs_temp_notls fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'fr_vs_temp_notls'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(fr_vs_temp_notls_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(fr_vs_temp_notls_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(fr_vs_temp_notls_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'gamma'] = gamma
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_fr_vs_temp_notls(row, prefix = 'fr_vs_temp_notls'):
    """
    Performs the inverse function of make_fit_row_fr_vs_temp_notls

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'fr_vs_temp_notls'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in fr_vs_temp_notls_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in fr_vs_temp_notls_names:
        popt.append(row[prefix + key])
    perr = []
    for key in fr_vs_temp_notls_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    gamma = row[prefix + 'gamma']
    return p0, popt, perr, gamma, plot_path

def make_fit_row_Q_vs_temp_notls(p0, popt, perr, gamma, N0, plot_path = '',
                                 prefix = 'Q_vs_temp_notls'):
    """
    Wraps the output of fit_Q_vs_temp_notls fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    N0 (float): single-spin density of states at the Fermi Level
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'Q_vs_temp_notls'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(Q_vs_temp_notls_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(Q_vs_temp_notls_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(Q_vs_temp_notls_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'gamma'] = gamma
    row[prefix + 'N0'] = N0
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_Q_vs_temp_notls(row, prefix = 'Q_vs_temp_notls'):
    """
    Performs the inverse function of make_fit_row_Q_vs_temp_notls

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'Q_vs_temp_notls'

    Returns:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    gamma (float): 1, 1/2, or 1/3 for thin-film, local, or anomalous limits
    N0 (float): single-spin density of states at the Fermi Level
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    """
    if len(prefix):
        prefix += '_'
    p0 = []
    for key in Q_vs_temp_notls_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in Q_vs_temp_notls_names:
        popt.append(row[prefix + key])
    perr = []
    for key in Q_vs_temp_notls_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    gamma = row[prefix + 'gamma']
    N0 = row[prefix + 'N0']
    return p0, popt, perr, gamma, N0, plot_path

################################################################################
######################## Funcs with only TLS component #########################
################################################################################
def make_fit_row_fr_vs_temp_tls(p0, popt, perr, plot_path = '',
                            prefix = 'fr_vs_temp_tls'):
    """
    Wraps the output of fit_fr_vs_temp_tls fitting into a pd.Series instance

    Parameters:
    p0 (np.array): fit parameter guess
    popt (np.array): fit parameters
    perr (np.array): standard errors on fit parameters
    plot_path (str): path to the saved plot, or empty string if it does not
        exists
    prefix (str): prefix for the column names. default is 'fr_vs_temp_tls'

    Returns:
    row (pd.Series): pd.Series object that includes all of the input data
    """
    if len(prefix):
        prefix += '_'
    row = pd.Series(dtype = float)
    for key, pi in zip(fr_vs_temp_tls_names, p0):
        row[prefix + key + '_guess'] = pi
    for key, pi in zip(fr_vs_temp_tls_names, popt):
        row[prefix + key] = pi
    for key, pi in zip(fr_vs_temp_tls_names, perr):
        row[prefix + key + '_err'] = pi
    row[prefix + 'plotpath'] = plot_path
    return row

def separate_fit_row_fr_vs_temp_tls(row, prefix = 'fr_vs_temp_tls'):
    """
    Performs the inverse function of make_fit_row_fr_vs_temp_tls

    Parameters:
    row (pd.Series): pd.Series object that includes all of the input data
    prefix (str): prefix for the column names. default is 'fr_vs_temp_tls'

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
    for key in fr_vs_temp_tls_names:
        p0.append(row[prefix + key + '_guess'])
    popt = []
    for key in fr_vs_temp_tls_names:
        popt.append(row[prefix + key])
    perr = []
    for key in fr_vs_temp_tls_names:
        perr.append(row[prefix + key + '_err'])
    plot_path = row[prefix + 'plotpath']
    p0, popt, perr = np.array(p0), np.array(popt), np.array(perr)
    return p0, popt, perr, plot_path

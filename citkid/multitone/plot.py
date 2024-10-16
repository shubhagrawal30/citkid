import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from ..util import fix_path

def plot_ares_opt(a_nls, fcal_indices):
    """
    Plots the current status of the power optimization procedure

    Parameters:
    a_nls (list): each value is an array of the values of the nonlinearity
        parameter for each resonator for the given iteration
    fcal_indices (array-like): indices that are calibration tones

    Returns:
    fig_hist (pyplot.fig): histogram of nonlinearity parameters for each
        iteration
    fig_opt (pyplot.fig): plot of the percent of the array that is optimized
        versus iteration number
    """
    fig_hist, ax_hist = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_hist.set(xlim = (0,1.5), xlabel = 'Nonlinearity parameter', ylabel = 'Number of KIDs')
    bins = np.linspace(0, 1.5, 20)
    for index, a_nl in enumerate(a_nls):
        [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        if index == len(a_nls):
            alpha = 0.8
        else:
            alpha = 0.5
        ax_hist.hist(a_nl, bins, label = index, edgecolor = color, color = color, alpha = alpha)
    ax_hist.legend(title = 'Iteration', framealpha = 1)

    fig_opt, ax_opt = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_opt.set(xlabel = 'Iteration', ylabel = 'Percentage of array optimized')
    ax_opt.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax_opt.grid()
    indices, percents = [], []
    for index, a_nl in enumerate(a_nls):
        a_nl = [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        a_opt = [a for a in a_nl if a <= 0.6 and a >= 0.4]
        percent = len(a_opt) / (len(a_nl)) * 100
        ax_opt.scatter(index, percent, marker = 's', color = color)
        indices.append(index)
        percents.append(percent)
    ax_opt.plot(indices, percents, '--k')
    return fig_hist, fig_opt

def plot_update_fres(fs, zs, fres, fcal_indices, resonator_indices):
    """
    Plots the results of update_fres in batches

    Parameters:
    fs (array-like): fine sweep frequency data in Hz for each resonator in fres
    zs (array-like): fine sweep complex S21 data for each resonator in fres
    fres (np.array or None): list of resonance frequencies in Hz
    fcal_indices (array-like): list of calibrations tone indices (index into
        fs, zs, fres, Qres). Calibration tone frequencies will not be updated
    resonator_indices (array-like): resonator indices for plotting
    plot_directory (str): directory to save plots
    """
    fs, zs, fres = np.array(fs), np.array(zs), np.array(fres)
    fcal_indices = np.array(fcal_indices)
    resonator_indices = np.array(resonator_indices)
    plot_directory = fix_path(plot_directory)
    os.makedirs(plot_directory, exist_ok = True)
    num_plots = len(fres)

    plots_per_fig = 50
    num_figs = (num_plots - 1) // plots_per_fig + 1

    for fig_index in num_figs:
        ix0, ix1 = fig_index * plots_per_fig, fig_index * (plots_per_fig + 1)
        fs0, zs0 = fs0[ix0:ix1], zs0[ix0:ix1], fres[ix0:ix1]
        # Left off here. I was debating about whether I should plot cal indices,
        # but I think I should to make sure they don't overlap with other tones
        # However, they should be a different color to be immediately recognizable

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

def plot_ares_opt(a_nls, fcal_indices):
    fig_hist, ax_hist = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax_hist.set(xlim = (0,1.5), xlabel = 'Nonlinearity parameter', ylabel = 'Number of KIDs')
    bins = np.linspace(0, 1.5, 20)
    for index, a_nl in enumerate(a_nls):
        [a_nl[i] for i in range(len(a_nl)) if i not in fcal_indices]
        color = plt.cm.viridis(index / len(a_nls))
        ax_hist.hist(a_nl, bins, label = index, edgecolor = color, fill = False, histtype = 'step')
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
        percent = len(a_opt) / len(a_nl)
        ax_opt.scatter(index, percent, marker = 's', color = color)
        indices.append(index)
        percents.append(percent)
    ax_opt.plot(indices, percents, '--k')
    return fig_hist, fig_opt
    # for a_nl in a_nls:
        
    # # plot optimization history
    # isets = list(range(0,idx0+1))
    # a_arrs = []
    # for ii in isets:
    #     a_arr = np.load(rfsoc.out_directory+f'iq_a_{lo}MHz_{ii:02d}.npy')
    #     a_arrs.append(a_arr)
    # a_arrs = np.array(a_arrs)
    # opt_nums = np.array([len(a_arr[(a_arr>0.4)&(a_arr<0.6)]) for a_arr in a_arrs])

    # fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    # ax.plot(isets, opt_nums/len(fres), 'k.')
    # ax.plot(isets, opt_nums/len(fres), 'k--')
    # plt.tight_layout()
    # ax.set(xlabel='set #', ylabel='fraction of KIDs with 0.4<a<0.6')
    # plt.tight_layout()
    # fig.savefig(plots+f'{lo}MHz_number_optimized.png')
    # plt.close(fig)

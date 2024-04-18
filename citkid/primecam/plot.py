import matplotlib.pyplot as plot
import numpy as np

def plot_ares_opt(a_nlss):
    a_nls = [[0.1, 0.2], [0.3, 0.4]]
    fig, ax = plt.subplots(figsize = (6, 4), dpi = 200, layout = 'tight')
    ax.set(xlim = (0,1.5), xlabel = 'Nonlinearity parameter', ylabel = 'Number of KIDs')
    bins = np.linspace(0, 1.5, 20)
    for index, a_nl in enumerate(a_nls):
        color = plt.cm.viridis(index / len(a_nls))
        ax.hist(a_nl, bins, label = index, edgecolor = color, fill = False)
    ax.legend(title = 'Iteration', framealpha = 1)
    return fig, ax

    # plot optimization history
    isets = list(range(0,idx0+1))
    a_arrs = []
    for ii in isets:
        a_arr = np.load(rfsoc.out_directory+f'iq_a_{lo}MHz_{ii:02d}.npy')
        a_arrs.append(a_arr)
    a_arrs = np.array(a_arrs)
    opt_nums = np.array([len(a_arr[(a_arr>0.4)&(a_arr<0.6)]) for a_arr in a_arrs])

    fig, ax = plt.subplots(figsize=(6,4), dpi=100)
    ax.plot(isets, opt_nums/len(fres), 'k.')
    ax.plot(isets, opt_nums/len(fres), 'k--')
    plt.tight_layout()
    ax.set(xlabel='set #', ylabel='fraction of KIDs with 0.4<a<0.6')
    plt.tight_layout()
    fig.savefig(plots+f'{lo}MHz_number_optimized.png')
    plt.close(fig)

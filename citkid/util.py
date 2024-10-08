import matplotlib.pyplot as plt
from io import BytesIO
import warnings

def fix_path(path):
    """
    Given a path, convert \ to / and ensure the path ends in / if it is a folder

    Parameters:
    path (str): path to a folder or file

    Returns:
    fixed_path (str): fixed path
    """
    if path == '':
        return path
    fixed_path = path.replace('\\', '/')
    if fixed_path[-1] != '/' and ('.' not in fixed_path.split('/')[-1]):
        fixed_path += '/'
    return fixed_path

def save_fig(fig, filename, plot_directory, ftype = 'png',
             tight_layout = False, close_fig = True):
    """
    Saves a pyplot figure with standard settings

    Parameters:
    fig (plt.figure): figure to save
    filename (str): name of the file to save, without extension
    plot_directory (str): directory to save the file
    ftype (str): file type to save. Common types are 'png' and 'eps'
    tight_layout (bool): if True, sets the figure layout to 'tight'
    """
    if fig is not None:
        plot_directory = fix_path(plot_directory)
        fig.set_facecolor('white')
        if tight_layout:
            fig.tight_layout()
        plt.figure(fig.number)
        try:
            plt.savefig(plot_directory + filename + '.' + ftype,
                        bbox_inches='tight', pad_inches = 0.05)
        except Exception as e:
            plt.savefig(plot_directory + filename + '.' + ftype,
                        pad_inches = 0.05)
        if close_fig:
            plt.close(fig)

def save_figure_to_memory(fig):
    """
    Saves a matplotlib figure to memory. Use this to easily stitch together
    multiple figures without saving extra files

    Parameters:
    fig (pyplot.figure): figure to save

    Returns:
    buf (BytesIO): memory buffer of saved figure
    """
    buf = BytesIO()
    fig.set_facecolor('white')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0.05)
    buf.seek(0)
    return buf

def combine_figures_vertically(fig1, fig2):
    """
    Combine two matplotlib figures vertically for saving as a single file

    Parameters:
    fig1, fig2 (pyplot.figure): figures to combine

    Returns:
    fig (pyplot.figure): combined figure
    """
    buf1 = save_figure_to_memory(fig1)
    buf2 = save_figure_to_memory(fig2)
    plt.close(fig1)
    plt.close(fig2)
    fig, axs = plt.subplots(2, 1, dpi = 200, layout = 'tight')
    for ax in axs:
        ax.set_axis_off()
    axs[0].imshow(plt.imread(buf1))
    axs[1].imshow(plt.imread(buf2))
    fig.tight_layout()
    return fig

def combine_figures_horizontally(fig1, fig2):
    """
    Combine two matplotlib figures horizontally for saving as a single file

    Parameters:
    fig1, fig2 (pyplot.figure): figures to combine

    Returns:
    fig (pyplot.figure): combined figure
    """
    buf1 = save_figure_to_memory(fig1)
    buf2 = save_figure_to_memory(fig2)
    plt.close(fig1)
    plt.close(fig2)
    fig, axs = plt.subplots(1, 2, dpi = 200, layout = 'tight')
    for ax in axs:
        ax.set_axis_off()
    axs[0].imshow(plt.imread(buf1))
    axs[1].imshow(plt.imread(buf2))
    fig.tight_layout()
    return fig

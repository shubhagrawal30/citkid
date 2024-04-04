import matplotlib.pyplot as plt

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

def save_fig(fig, filename, plot_directory, ftype = 'png'):
    """
    Saves a pyplot figure with standard settings

    Parameters:
    fig (plt.figure): figure to save
    filename (str): name of the file to save, without extension
    plot_directory (str): directory to save the file
    ftype (str): file type to save. Common types are 'png' and 'eps'
    """
    if fig is not None:
        plot_directory = fix_path(plot_directory)
        fig.set_facecolor('white')
        fig.tight_layout()
        plt.figure(fig.number)
        plt.savefig(plot_directory + filename + '.' + ftype,
                    bbox_inches='tight', pad_inches = 0.05)
        plt.close(fig)

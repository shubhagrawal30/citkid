import matplotlib.pyplot as plt
import numpy as np

def plot_circle(z, A, B, R):
    """
    Plots IQ data with a circular fit

    Parameters:
    z (np.array): complex IQ data
    A, B (float): circle origin
    R (float): circle radius

    Returns:
    fig, ax (pyplot figure and axis): data and fit plot 
    """
    fig, ax = plt.subplots(figsize = (4, 4), dpi = 300)
    ax.plot(np.real(z), np.imag(z), 'r.')
    ax.set_aspect('equal', adjustable='datalim')
    cir = plt.Circle((A, B), R, color='k', fill=False, label='IQ loop fit')
    ax.add_patch(cir)
    ax.set(xlabel='I', ylabel='Q')
    return fig, ax

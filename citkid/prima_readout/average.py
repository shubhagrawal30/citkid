import numpy as np
def average_x(x, navg = 10):
    """
    Reduce x data by averaging to a lower sample rate

    Parameters:
    x (np.array): array of timestream data
    navg (int): number of data points to average per bin

    Returns:
    x_avg (np.array): averaged x data with reduced sample rate
    """
    num_chunks = len(x) // navg
    x_avg = np.mean(x[:num_chunks * navg].reshape(-1, navg), axis = 1)
    return x_avg

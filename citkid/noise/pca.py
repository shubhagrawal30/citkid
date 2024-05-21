import numpy as np

def pca(y, n_components = 5):
    """
    Removes the largest principal components from a noise dataset using single
    value decomposition

    Parameters:
    y (array-like): array of timestream datasets, where each value is an array
        of timestream data
    n_components (int): number of principal components to remove

    Returns:
    z (np.array): array of timestream datasets corresponding to y, with
        principal components removed
    """
    # Normalize input data
    y = np.array(y).T
    mean = np.mean(y, axis=0)
    y_normalized = y - mean
    std_dev = np.std(y_normalized, axis=0)
    y_normalized /= std_dev
    y_normalized = y_normalized.T
    # Perform SVD
    U, S, eigenvectors = np.linalg.svd(y_normalized, full_matrices = False)
    S_rmvd = S.copy()
    S_rmvd[0:n_components] = 0. # set the components to remove to zero
    # Reconstruct data with modes removed
    z_normalized = (U * S_rmvd) @ eigenvectors
    # Remove normalization
    z = ((z_normalized.T * std_dev) + mean).T
    # Get eigenvalues
    eigenvalues = (U * S).T * std_dev + mean
    return z, eigenvectors, eigenvalues

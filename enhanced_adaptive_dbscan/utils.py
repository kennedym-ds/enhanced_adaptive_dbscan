# enhanced_adaptive_dbscan/utils.py

from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import numpy as np

def count_neighbors(kdtree, X, epsilon, i):
    """
    Count the number of neighbors within epsilon[i] for point i using KDTree.
    
    Parameters:
    - kdtree (KDTree): KDTree built on the dataset.
    - X (ndarray): Shape (n_samples, n_features).
    - epsilon (ndarray): Shape (n_samples,).
    - i (int): Index of the query point.
    
    Returns:
    - count (int): Number of neighbors within epsilon[i].
    """
    neighbors = kdtree.query_radius(X[i].reshape(1, -1), r=epsilon[i])[0]
    return len(neighbors)


def neighbors_for_point(kdtree: KDTree, X: np.ndarray, epsilon: np.ndarray, i: int):
    """Return neighbor indices for point i using KDTree.query_radius.

    Extracted as a top-level function to be picklable for joblib. This
    avoids using a lambda in Parallel on Windows (spawn), which would fail.

    Parameters
    ----------
    kdtree : KDTree
        Tree built on X.
    X : ndarray of shape (n_samples, n_features)
        Data matrix.
    epsilon : ndarray of shape (n_samples,)
        Per-point radius.
    i : int
        Index of the query point.

    Returns
    -------
    ndarray
        1D array of neighbor indices for point i.
    """
    return kdtree.query_radius(X[i].reshape(1, -1), r=epsilon[i])[0]
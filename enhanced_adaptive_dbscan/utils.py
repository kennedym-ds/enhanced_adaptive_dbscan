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
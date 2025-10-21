# enhanced_adaptive_dbscan/utils.py

from sklearn.neighbors import KDTree
from joblib import Parallel, delayed
import numpy as np
from typing import Tuple, Optional
import warnings

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


def compute_kdist_graph(X: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Compute k-distance graph for DBSCAN parameter estimation.
    
    The k-distance graph plots the distance to the k-th nearest neighbor
    for each point, sorted in ascending order. The "elbow" or "knee" point
    in this graph suggests an optimal epsilon value for DBSCAN.
    
    This implements the approach from X-DBSCAN (2024) research:
    "Improvement of DBSCAN Algorithm Based on K-Dist Graph for Adaptive..."
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input data points
    k : int, default=4
        The k-th nearest neighbor to compute distances for.
        Commonly set to min_samples - 1 for DBSCAN.
        
    Returns:
    --------
    k_distances : ndarray of shape (n_samples,)
        Distance to the k-th nearest neighbor for each point,
        sorted in ascending order
        
    Examples:
    ---------
    >>> import numpy as np
    >>> from enhanced_adaptive_dbscan.utils import compute_kdist_graph
    >>> X = np.random.randn(100, 2)
    >>> k_distances = compute_kdist_graph(X, k=4)
    >>> # Find elbow point for optimal eps
    >>> optimal_eps = find_kdist_elbow(k_distances)
    """
    n_samples = X.shape[0]
    
    if k >= n_samples:
        warnings.warn(
            f"k={k} is >= n_samples={n_samples}. "
            f"Setting k to {n_samples - 1}."
        )
        k = n_samples - 1
    
    # Build KDTree for efficient neighbor search
    tree = KDTree(X)
    
    # Query for k+1 nearest neighbors (includes point itself)
    distances, _ = tree.query(X, k=k+1)
    
    # Take k-th nearest neighbor distance (excluding the point itself)
    k_distances = distances[:, k]
    
    # Sort in ascending order for visualization
    k_distances_sorted = np.sort(k_distances)
    
    return k_distances_sorted


def find_kdist_elbow(k_distances: np.ndarray, 
                     method: str = 'kneedle',
                     sensitivity: float = 1.0) -> Tuple[float, int]:
    """
    Find the elbow/knee point in k-distance graph to suggest optimal epsilon.
    
    Implements multiple methods for elbow detection:
    - 'kneedle': Kneedle algorithm (finds max curvature)
    - 'derivative': Maximum of second derivative
    - 'distance': Maximum perpendicular distance from line
    
    Parameters:
    -----------
    k_distances : ndarray of shape (n_samples,)
        Sorted k-distances from compute_kdist_graph
    method : str, default='kneedle'
        Method to use for elbow detection
    sensitivity : float, default=1.0
        Sensitivity parameter (higher = more conservative)
        
    Returns:
    --------
    optimal_eps : float
        Suggested epsilon value
    elbow_index : int
        Index of the elbow point in k_distances array
        
    References:
    -----------
    Kneedle algorithm: Satopaa et al. "Finding a 'Kneedle' in a Haystack" (2011)
    X-DBSCAN: MDPI Electronics 12(15):3213 (2024)
    """
    n_points = len(k_distances)
    
    if n_points < 3:
        # Not enough points to find elbow
        return k_distances[-1], n_points - 1
    
    if method == 'kneedle':
        # Normalize to [0, 1] range
        x = np.arange(n_points) / (n_points - 1)
        y = (k_distances - k_distances[0]) / (k_distances[-1] - k_distances[0] + 1e-10)
        
        # Compute perpendicular distance from straight line
        distances = y - x
        
        # Find maximum distance point (adjusted by sensitivity)
        threshold = sensitivity * np.std(distances)
        candidates = np.where(distances > threshold)[0]
        
        if len(candidates) > 0:
            # Find point with maximum curvature among candidates
            elbow_index = candidates[np.argmax(distances[candidates])]
        else:
            # Fallback to simple maximum
            elbow_index = np.argmax(distances)
            
    elif method == 'derivative':
        # Use second derivative to find maximum curvature
        first_derivative = np.diff(k_distances)
        second_derivative = np.diff(first_derivative)
        
        # Find point of maximum curvature change
        elbow_index = np.argmax(second_derivative) + 1
        
    elif method == 'distance':
        # Maximum perpendicular distance from line connecting endpoints
        start = np.array([0, k_distances[0]])
        end = np.array([n_points - 1, k_distances[-1]])
        
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / (line_len + 1e-10)
        
        max_dist = 0
        elbow_index = 0
        
        for i in range(1, n_points - 1):
            point = np.array([i, k_distances[i]])
            vec_to_point = point - start
            
            # Perpendicular distance from point to line
            projection = np.dot(vec_to_point, line_unitvec)
            closest_point = start + projection * line_unitvec
            dist = np.linalg.norm(point - closest_point)
            
            if dist > max_dist:
                max_dist = dist
                elbow_index = i
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kneedle', 'derivative', or 'distance'.")
    
    # Ensure elbow_index is within valid range
    elbow_index = max(0, min(elbow_index, n_points - 1))
    optimal_eps = k_distances[elbow_index]
    
    return optimal_eps, elbow_index


def suggest_dbscan_parameters(X: np.ndarray, 
                              k_range: Tuple[int, int] = (4, 20),
                              n_trials: int = 5) -> dict:
    """
    Automatically suggest DBSCAN parameters (eps and min_samples) using k-distance analysis.
    
    This function computes k-distance graphs for multiple values of k and
    identifies consistent elbow points to suggest robust parameters.
    
    Based on X-DBSCAN (2024) and K-DBSCAN (2024) research methods.
    
    Parameters:
    -----------
    X : ndarray of shape (n_samples, n_features)
        Input data points
    k_range : tuple of (int, int), default=(4, 20)
        Range of k values to try
    n_trials : int, default=5
        Number of k values to sample from k_range
        
    Returns:
    --------
    parameters : dict
        Dictionary containing:
        - 'eps': Suggested epsilon value
        - 'min_samples': Suggested min_samples value
        - 'k_distances': K-distance arrays for each k tried
        - 'elbow_points': Elbow points found for each k
        - 'confidence': Confidence score (0-1) based on consistency
        
    Examples:
    ---------
    >>> import numpy as np
    >>> from enhanced_adaptive_dbscan.utils import suggest_dbscan_parameters
    >>> X = np.random.randn(1000, 2)
    >>> params = suggest_dbscan_parameters(X)
    >>> print(f"Suggested: eps={params['eps']:.3f}, min_samples={params['min_samples']}")
    """
    k_min, k_max = k_range
    k_values = np.linspace(k_min, k_max, n_trials, dtype=int)
    
    suggested_eps = []
    elbow_indices = []
    k_distances_all = []
    
    for k in k_values:
        # Compute k-distance graph
        k_distances = compute_kdist_graph(X, k=k)
        k_distances_all.append(k_distances)
        
        # Find elbow point
        eps, elbow_idx = find_kdist_elbow(k_distances, method='kneedle')
        suggested_eps.append(eps)
        elbow_indices.append(elbow_idx)
    
    # Compute statistics on suggested values
    median_eps = np.median(suggested_eps)
    std_eps = np.std(suggested_eps)
    
    # Confidence based on consistency (low std = high confidence)
    max_std = median_eps * 0.5  # 50% variation is low confidence
    confidence = max(0.0, 1.0 - (std_eps / max_std))
    
    # Suggest min_samples based on most common k
    suggested_min_samples = int(np.median(k_values))
    
    return {
        'eps': float(median_eps),
        'min_samples': suggested_min_samples,
        'k_distances': k_distances_all,
        'elbow_points': list(zip(k_values.tolist(), suggested_eps)),
        'confidence': float(confidence),
        'std_eps': float(std_eps),
        'suggested_eps_range': (float(median_eps - std_eps), float(median_eps + std_eps))
    }
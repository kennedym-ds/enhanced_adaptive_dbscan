# Core API Reference

This section documents the core components of the Enhanced Adaptive DBSCAN framework.

## EnhancedAdaptiveDBSCAN

The main clustering class that extends DBSCAN with adaptive capabilities.

### Class Definition

```python
class EnhancedAdaptiveDBSCAN:
    """
    Enhanced Adaptive DBSCAN clustering algorithm.
    
    This class implements an advanced version of DBSCAN that includes:
    - Adaptive parameter selection
    - Multi-density region handling
    - Hierarchical clustering analysis
    - Ensemble methods integration
    - Production-ready features
    
    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
        
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
        
    adaptive : bool, default=True
        Whether to use adaptive parameter selection.
        
    enable_mdbscan : bool, default=False
        Whether to enable multi-density DBSCAN features.
        
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        The algorithm to be used by NearestNeighbors.
        
    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.
        
    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances.
        
    n_jobs : int, default=None
        The number of parallel jobs to run.
        
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset given to fit().
        
    core_sample_indices_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    """
    
    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        adaptive=True,
        enable_mdbscan=False,
        algorithm='auto',
        leaf_size=30,
        metric='euclidean',
        n_jobs=None,
        random_state=None
    ):
        ...
```

### Methods

#### fit(X, y=None)

Fit the clustering algorithm to the data.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Training data.
- `y` : Ignored
  Not used, present for API consistency.

**Returns:**
- `self` : object
  Returns the instance itself.

#### predict(X)

Predict cluster labels for new data points.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  New data to predict.

**Returns:**
- `labels` : ndarray of shape (n_samples,)
  Cluster labels for each sample.

#### fit_predict(X, y=None)

Fit the model and predict cluster labels in one step.

**Parameters:**
- `X` : array-like of shape (n_samples, n_features)
  Training data.
- `y` : Ignored
  Not used, present for API consistency.

**Returns:**
- `labels` : ndarray of shape (n_samples,)
  Cluster labels for each sample.

#### plot(X=None, labels=None, title=None, figsize=(10, 8), **kwargs)

Visualize clustering results.

**Parameters:**
- `X` : array-like, optional
  Data to plot. If None, uses data from last fit.
- `labels` : array-like, optional
  Cluster labels. If None, uses labels from last fit.
- `title` : str, optional
  Plot title.
- `figsize` : tuple, default=(10, 8)
  Figure size.

#### evaluate(X=None, labels=None, metrics=None)

Evaluate clustering performance.

**Parameters:**
- `X` : array-like, optional
  Data to evaluate. If None, uses data from last fit.
- `labels` : array-like, optional
  True labels for evaluation. If None, uses internal metrics.
- `metrics` : list, optional
  Metrics to compute. If None, uses default metrics.

**Returns:**
- `results` : dict
  Dictionary containing evaluation metrics.

#### get_params(deep=True)

Get parameters for this estimator.

**Parameters:**
- `deep` : bool, default=True
  If True, return parameters for sub-estimators too.

**Returns:**
- `params` : dict
  Parameter names mapped to their values.

#### set_params(**params)

Set parameters for this estimator.

**Parameters:**
- `**params` : dict
  Estimator parameters.

**Returns:**
- `self` : estimator instance
  Estimator instance.

### Properties

#### labels_

```python
@property
def labels_(self):
    """Cluster labels for each point in the dataset."""
    return self._labels
```

#### core_sample_indices_

```python
@property
def core_sample_indices_(self):
    """Indices of core samples."""
    return self._core_sample_indices
```

#### n_clusters_

```python
@property
def n_clusters_(self):
    """Number of clusters found."""
    if hasattr(self, '_labels'):
        return len(set(self._labels)) - (1 if -1 in self._labels else 0)
    return 0
```

#### n_noise_

```python
@property
def n_noise_(self):
    """Number of noise points."""
    if hasattr(self, '_labels'):
        return list(self._labels).count(-1)
    return 0
```

## Multi-Density Components

### MultiDensityClusterEngine

Handles clustering in regions with varying densities.

```python
class MultiDensityClusterEngine:
    """
    Engine for handling multi-density clustering scenarios.
    
    This component automatically identifies regions with different
    densities and applies appropriate clustering parameters for each region.
    """
    
    def __init__(self, base_clusterer, density_threshold=0.1):
        self.base_clusterer = base_clusterer
        self.density_threshold = density_threshold
    
    def fit_cluster(self, X):
        """Fit clustering with multi-density support."""
        ...
    
    def identify_density_regions(self, X):
        """Identify regions with different densities."""
        ...
```

### HierarchicalDensityManager

Manages hierarchical clustering analysis across different scales.

```python
class HierarchicalDensityManager:
    """
    Manages hierarchical density analysis for stable cluster selection.
    """
    
    def __init__(self, hierarchy_method='single', stability_threshold=0.3):
        self.hierarchy_method = hierarchy_method
        self.stability_threshold = stability_threshold
    
    def build_hierarchy(self, X, labels):
        """Build hierarchical clustering structure."""
        ...
    
    def select_stable_clusters(self):
        """Select most stable clusters from hierarchy."""
        ...
```

## Adaptive Optimization Components

### AdaptiveTuningEngine

Main engine for parameter optimization.

```python
class AdaptiveTuningEngine:
    """
    Adaptive parameter tuning engine using various optimization strategies.
    """
    
    def __init__(self, optimization_strategy='bayesian'):
        self.optimization_strategy = optimization_strategy
    
    def optimize_parameters(self, X, max_iterations=50):
        """Optimize clustering parameters for given dataset."""
        ...
```

### BayesianOptimizer

Bayesian optimization for parameter tuning.

```python
class BayesianOptimizer:
    """
    Bayesian optimization for efficient parameter search.
    """
    
    def __init__(self, acquisition_function='expected_improvement'):
        self.acquisition_function = acquisition_function
    
    def optimize(self, X, bounds, n_iterations=30):
        """Run Bayesian optimization."""
        ...
```

## Ensemble Components

### ConsensusClusteringEngine

Main ensemble clustering engine.

```python
class ConsensusClusteringEngine:
    """
    Consensus clustering engine that combines multiple clustering results.
    """
    
    def __init__(self, base_clusterer, n_estimators=10):
        self.base_clusterer = base_clusterer
        self.n_estimators = n_estimators
    
    def fit_consensus_clustering(self, X):
        """Fit consensus clustering using ensemble of clusterers."""
        ...
```

## Utility Functions

### Data Preprocessing

```python
def preprocess_data(X, scaling='standard', handle_outliers=True):
    """
    Preprocess data for clustering.
    
    Parameters
    ----------
    X : array-like
        Input data
    scaling : str, default='standard'
        Scaling method ('standard', 'minmax', 'robust', 'none')
    handle_outliers : bool, default=True
        Whether to handle outliers
        
    Returns
    -------
    X_processed : array-like
        Processed data
    """
    ...
```

### Evaluation Metrics

```python
def compute_clustering_metrics(X, labels, true_labels=None):
    """
    Compute comprehensive clustering evaluation metrics.
    
    Parameters
    ----------
    X : array-like
        Data points
    labels : array-like
        Predicted cluster labels
    true_labels : array-like, optional
        True cluster labels for supervised metrics
        
    Returns
    -------
    metrics : dict
        Dictionary of computed metrics
    """
    ...
```

### Visualization Utilities

```python
def plot_clustering_results(X, labels, centers=None, title=None):
    """
    Plot clustering results with optional cluster centers.
    
    Parameters
    ----------
    X : array-like
        Data points
    labels : array-like
        Cluster labels
    centers : array-like, optional
        Cluster centers
    title : str, optional
        Plot title
    """
    ...

def plot_parameter_optimization(optimization_history):
    """
    Plot parameter optimization progress.
    
    Parameters
    ----------
    optimization_history : dict
        Optimization history containing parameters and scores
    """
    ...
```

This core API documentation provides comprehensive coverage of the main framework components. For detailed implementation examples, see the Usage Examples section.

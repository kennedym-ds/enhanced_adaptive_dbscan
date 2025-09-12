# Getting Started

## Installation

### From Source

```bash
git clone https://github.com/your-username/enhanced_adaptive_dbscan.git
cd enhanced_adaptive_dbscan
pip install -e .
```

### Requirements

- Python 3.8+
- NumPy
- Scikit-learn
- Matplotlib
- Pandas (optional, for data handling)

## Basic Usage

### Simple Clustering

```python
from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(200, 2)

# Create clustering instance
clusterer = EnhancedAdaptiveDBSCAN(
    eps=0.1,
    min_samples=5,
    adaptive=True
)

# Fit and predict clusters
labels = clusterer.fit_predict(X)

# Get cluster statistics
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
```

### With Visualization

```python
# Create and fit the model
clusterer = EnhancedAdaptiveDBSCAN()
labels = clusterer.fit_predict(X)

# Visualize results
clusterer.plot(
    title="Enhanced Adaptive DBSCAN Results",
    figsize=(10, 8)
)

# Show cluster metrics
clusterer.evaluate()
```

## Key Concepts

### Adaptive Parameters

The framework automatically adjusts clustering parameters based on local data density:

- **Adaptive Epsilon**: Adjusts the neighborhood radius based on local density
- **Dynamic MinPts**: Varies minimum points requirement across regions
- **Stability Analysis**: Selects the most stable clustering across parameter ranges

### Multi-Density Support

Handle datasets with varying density regions:

```python
# Enable multi-density clustering
clusterer = EnhancedAdaptiveDBSCAN(
    enable_mdbscan=True,
    density_adaptation=True
)
```

### Ensemble Clustering

Improve stability through consensus clustering:

```python
from enhanced_adaptive_dbscan.ensemble_clustering import ConsensusClusteringEngine

# Create ensemble clusterer
ensemble = ConsensusClusteringEngine(
    base_clusterer=EnhancedAdaptiveDBSCAN(),
    n_estimators=10,
    voting_strategy='quality_weighted'
)

# Fit ensemble
consensus_labels = ensemble.fit_consensus_clustering(X)
```

## Next Steps

- Explore [Usage Examples](usage_examples.md) for detailed scenarios
- Learn about [Advanced Features](advanced_features.md) like optimization and streaming
- Check the [API Reference](api/core.md) for complete documentation

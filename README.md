# Enhanced Adaptive DBSCAN

An Enhanced Adaptive DBSCAN clustering algorithm tailored for semiconductor wafer defect analysis.

## Features

- **Adaptive Parameter Selection:** Adjusts ε and MinPts based on local density.
- **Stability-Based Cluster Selection:** Retains only robust and persistent clusters.
- **Dynamic Cluster Centers:** Maintains up-to-date cluster centroids.
- **Partial Re-Clustering:** Efficiently updates affected clusters with new data points.
- **Incremental Clustering:** Handles streaming data seamlessly.
- **Interactive Visualization:** Utilizes Plotly for dynamic cluster plots.
- **Comprehensive Logging:** Tracks the algorithm's progress and decisions.

## Installation

```bash
pip install enhanced_adaptive_dbscan

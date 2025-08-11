# Usage

## Installation
```bash
pip install enhanced_adaptive_dbscan
```

## Quick start
```python
from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
import numpy as np

X = np.random.randn(1000, 2)
severity = np.random.randint(1, 11, size=(1000, 1))
X_full = np.hstack((X, severity))

model = EnhancedAdaptiveDBSCAN(
    wafer_shape='circular', wafer_size=100, k=20, density_scaling=1.0,
    buffer_ratio=0.1, min_scaling=5, max_scaling=10, n_jobs=-1,
    max_points=100000, subsample_ratio=0.1, random_state=42,
    additional_features=[2], feature_weights=[1.0], stability_threshold=0.6
)
model.fit(X_full, additional_attributes=severity)
labels = model.labels_
```

## Tips
- Scale inputs; outliers can bias local-density distances.
- For large datasets, tune subsample_ratio and k to balance accuracy vs. speed.
- Use evaluate_clustering for quick internal metrics.

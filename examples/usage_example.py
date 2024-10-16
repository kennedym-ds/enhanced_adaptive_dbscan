from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
import numpy as np

# Sample data
X = np.random.randn(1000, 2)
severity = np.random.randint(1, 11, size=1000).reshape(-1, 1)
X_full = np.hstack((X, severity))

# Initialize the model
model = EnhancedAdaptiveDBSCAN(
    wafer_shape='circular',
    wafer_size=100,
    k=20,
    density_scaling=1.0,
    buffer_ratio=0.1,
    min_scaling=5,
    max_scaling=10,
    n_jobs=-1,
    max_points=100000,
    subsample_ratio=0.1,
    random_state=42,
    additional_features=[2],  # Index of severity
    feature_weights=[1.0],
    stability_threshold=0.6
)

# Fit the model
model.fit(X_full, additional_attributes=severity)

# Retrieve cluster labels
labels = model.labels_

# Plot the clusters
model.plot_clusters(X_full, plot_all=False)

# Evaluate clustering
model.evaluate_clustering(X_full[:, :2], labels)
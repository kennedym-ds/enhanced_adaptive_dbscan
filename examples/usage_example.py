"""
Simple Usage Example for Enhanced Adaptive DBSCAN

This is a basic example showing how to use the Enhanced Adaptive DBSCAN
algorithm for clustering data with default settings.
"""

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def simple_clustering_example():
    """
    Demonstrate basic clustering with Enhanced Adaptive DBSCAN.
    """
    print("Enhanced Adaptive DBSCAN - Simple Usage Example")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X, true_labels = make_blobs(n_samples=300, centers=4, cluster_std=1.0, 
                               center_box=(-10.0, 10.0), random_state=42)
    
    print(f"Generated dataset: {len(X)} points, {len(set(true_labels))} true clusters")
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize the Enhanced Adaptive DBSCAN model
    model = EnhancedAdaptiveDBSCAN(
        eps=0.5,
        min_samples=5,
        enable_mdbscan=True,
        adaptive_eps=True,
        random_state=42
    )
    
    # Fit the model
    print("Fitting the model...")
    model.fit(X_scaled)
    
    # Retrieve cluster labels
    labels = model.labels_
    
    # Analyze results
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = list(labels).count(-1)
    
    print(f"\nClustering Results:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Outliers: {n_outliers}")
    print(f"  Outlier ratio: {n_outliers/len(X):.3f}")
    
    # Calculate basic metrics
    if n_clusters > 1:
        from sklearn.metrics import silhouette_score, adjusted_rand_score
        
        # Silhouette score
        silhouette = silhouette_score(X_scaled, labels)
        print(f"  Silhouette score: {silhouette:.3f}")
        
        # Adjusted Rand Index (compared to true labels)
        ari = adjusted_rand_score(true_labels, labels)
        print(f"  Adjusted Rand Index: {ari:.3f}")
    
    print(f"\nClustering completed successfully!")
    
    return X, X_scaled, labels, model

if __name__ == "__main__":
    # Run the simple clustering example
    X, X_scaled, labels, model = simple_clustering_example()
    
    print(f"\nFor more comprehensive examples, see:")
    print(f"  - basic_usage_guide.py: Detailed usage scenarios")
    print(f"  - real_world_use_cases.py: Practical applications")
    print(f"  - advanced_configuration.py: Parameter optimization")
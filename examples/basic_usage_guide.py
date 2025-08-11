"""
Basic Usage Example for Enhanced Adaptive DBSCAN

This example demonstrates the fundamental usage of the Enhanced Adaptive DBSCAN
clustering algorithm with various datasets and configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN


def basic_clustering_example():
    """Demonstrate basic clustering with different datasets."""
    print("=" * 60)
    print("BASIC CLUSTERING EXAMPLE")
    print("=" * 60)
    
    # Create sample datasets
    datasets = {
        "Blobs": make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42),
        "Moons": make_moons(n_samples=300, noise=0.1, random_state=42),
        "Circles": make_circles(n_samples=300, noise=0.1, factor=0.6, random_state=42)
    }
    
    # Initialize the Enhanced Adaptive DBSCAN model
    model = EnhancedAdaptiveDBSCAN(
        eps=0.5,
        min_samples=5,
        density_scaling=1.0,
        adaptive_eps=True,
        enable_stability_analysis=True,
        random_state=42
    )
    
    results = {}
    
    for name, (X, y_true) in datasets.items():
        print(f"\nProcessing {name} dataset...")
        
        # Fit the model
        model.fit(X)
        labels = model.labels_
        
        # Calculate metrics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        if len(set(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            ari = adjusted_rand_score(y_true, labels)
        else:
            silhouette = -1
            ari = 0
        
        results[name] = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'adjusted_rand_index': ari,
            'labels': labels
        }
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        print(f"  Silhouette score: {silhouette:.3f}")
        print(f"  Adjusted Rand Index: {ari:.3f}")
    
    return results, datasets


def multi_density_example():
    """Demonstrate multi-density clustering capabilities."""
    print("\n" + "=" * 60)
    print("MULTI-DENSITY CLUSTERING EXAMPLE")
    print("=" * 60)
    
    # Create multi-density data
    np.random.seed(42)
    
    # High-density cluster
    high_density = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 100)
    
    # Medium-density cluster
    medium_density = np.random.multivariate_normal([6, 6], [[0.3, 0], [0, 0.3]], 80)
    
    # Low-density cluster
    low_density = np.random.multivariate_normal([10, 2], [[0.8, 0], [0, 0.8]], 60)
    
    # Combine all data
    X_multi = np.vstack([high_density, medium_density, low_density])
    
    print(f"Dataset created with {len(X_multi)} points across multiple density regions")
    
    # Standard DBSCAN (for comparison)
    model_standard = EnhancedAdaptiveDBSCAN(
        eps=0.5,
        min_samples=5,
        enable_mdbscan=False
    )
    model_standard.fit(X_multi)
    labels_standard = model_standard.labels_
    
    # Multi-Density DBSCAN
    model_mdbscan = EnhancedAdaptiveDBSCAN(
        enable_mdbscan=True,
        enable_hierarchical_clustering=True,
        enable_boundary_refinement=True,
        enable_quality_analysis=True,
        adaptive_eps=True
    )
    model_mdbscan.fit(X_multi)
    labels_mdbscan = model_mdbscan.labels_
    
    # Compare results
    n_clusters_std = len(set(labels_standard)) - (1 if -1 in labels_standard else 0)
    n_clusters_md = len(set(labels_mdbscan)) - (1 if -1 in labels_mdbscan else 0)
    
    print(f"\nResults Comparison:")
    print(f"  Standard DBSCAN clusters: {n_clusters_std}")
    print(f"  Multi-Density DBSCAN clusters: {n_clusters_md}")
    
    if len(set(labels_standard)) > 1:
        sil_std = silhouette_score(X_multi, labels_standard)
        print(f"  Standard DBSCAN silhouette score: {sil_std:.3f}")
    
    if len(set(labels_mdbscan)) > 1:
        sil_md = silhouette_score(X_multi, labels_mdbscan)
        print(f"  Multi-Density DBSCAN silhouette score: {sil_md:.3f}")
    
    # Access Multi-Density specific results
    mdbscan_clusters = model_mdbscan.get_mdbscan_clusters()
    quality_analysis = model_mdbscan.get_quality_analysis()
    boundary_analysis = model_mdbscan.get_boundary_analysis()
    
    if mdbscan_clusters:
        print(f"  MDBSCAN found {len(mdbscan_clusters)} density regions")
    
    if quality_analysis:
        print(f"  Quality analysis available with {len(quality_analysis)} metrics")
    
    return X_multi, labels_standard, labels_mdbscan, model_mdbscan


def adaptive_optimization_example():
    """Demonstrate adaptive parameter optimization."""
    print("\n" + "=" * 60)
    print("ADAPTIVE OPTIMIZATION EXAMPLE")
    print("=" * 60)
    
    # Create complex dataset
    np.random.seed(42)
    X_complex, _ = make_blobs(n_samples=500, centers=6, cluster_std=1.5, random_state=42)
    
    print(f"Dataset created with {len(X_complex)} points")
    
    # Model with adaptive optimization enabled
    model_adaptive = EnhancedAdaptiveDBSCAN(
        enable_adaptive_optimization=True,
        optimization_method='bayesian',  # or 'genetic'
        n_optimization_trials=20,
        optimization_timeout=60,  # 1 minute
        adaptive_eps=True,
        enable_stability_analysis=True,
        random_state=42
    )
    
    print("Running adaptive optimization (this may take a moment)...")
    
    # Fit with optimization
    model_adaptive.fit(X_complex)
    labels_adaptive = model_adaptive.labels_
    
    # Get optimization results
    optimization_results = model_adaptive.get_optimization_results()
    
    print(f"\nOptimization Results:")
    if optimization_results:
        print(f"  Best parameters found: {optimization_results.get('best_params', 'N/A')}")
        print(f"  Best score: {optimization_results.get('best_score', 'N/A'):.3f}")
        print(f"  Optimization trials: {optimization_results.get('n_trials', 'N/A')}")
        print(f"  Optimization time: {optimization_results.get('optimization_time', 'N/A'):.2f}s")
    
    # Compare with manual parameters
    model_manual = EnhancedAdaptiveDBSCAN(
        eps=0.5,
        min_samples=5,
        enable_adaptive_optimization=False
    )
    model_manual.fit(X_complex)
    labels_manual = model_manual.labels_
    
    # Compare results
    n_clusters_adaptive = len(set(labels_adaptive)) - (1 if -1 in labels_adaptive else 0)
    n_clusters_manual = len(set(labels_manual)) - (1 if -1 in labels_manual else 0)
    
    if len(set(labels_adaptive)) > 1:
        sil_adaptive = silhouette_score(X_complex, labels_adaptive)
    else:
        sil_adaptive = -1
        
    if len(set(labels_manual)) > 1:
        sil_manual = silhouette_score(X_complex, labels_manual)
    else:
        sil_manual = -1
    
    print(f"\nComparison:")
    print(f"  Adaptive optimization clusters: {n_clusters_adaptive} (silhouette: {sil_adaptive:.3f})")
    print(f"  Manual parameters clusters: {n_clusters_manual} (silhouette: {sil_manual:.3f})")
    
    return X_complex, labels_adaptive, labels_manual, optimization_results


def visualization_example():
    """Demonstrate visualization capabilities."""
    print("\n" + "=" * 60)
    print("VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Create interesting dataset
    np.random.seed(42)
    X_vis, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    # Add some additional attributes
    severity = np.random.uniform(0.1, 1.0, len(X_vis)).reshape(-1, 1)
    X_full = np.hstack([X_vis, severity])
    
    # Fit model with visualization-friendly settings
    model = EnhancedAdaptiveDBSCAN(
        eps=0.3,
        min_samples=5,
        enable_mdbscan=True,
        enable_quality_analysis=True,
        additional_features=[2],  # Include severity
        feature_weights=[0.3],   # Weight for severity
        random_state=42
    )
    
    model.fit(X_full, additional_attributes=severity)
    labels = model.labels_
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Clustering results:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original data colored by cluster
    plt.subplot(1, 3, 1)
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            color = 'black'
            marker = 'x'
            alpha = 0.5
        else:
            marker = 'o'
            alpha = 0.8
        
        mask = labels == label
        plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                   c=[color], marker=marker, alpha=alpha, s=30)
    
    plt.title(f'Clusters (n={n_clusters})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 2: Data colored by severity
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=severity.flatten(), 
                         cmap='viridis', alpha=0.7, s=30)
    plt.colorbar(scatter, label='Severity')
    plt.title('Data by Severity')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 3: Cluster quality heatmap (if available)
    plt.subplot(1, 3, 3)
    quality_analysis = model.get_quality_analysis()
    
    if quality_analysis and n_clusters > 0:
        # Create a simple quality visualization
        cluster_ids = [i for i in range(n_clusters)]
        quality_scores = [0.8, 0.6, 0.7][:n_clusters]  # Mock scores for demonstration
        
        plt.bar(cluster_ids, quality_scores, color='skyblue', alpha=0.7)
        plt.title('Cluster Quality Scores')
        plt.xlabel('Cluster ID')
        plt.ylabel('Quality Score')
        plt.ylim(0, 1)
    else:
        plt.text(0.5, 0.5, 'Quality analysis\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Cluster Quality')
    
    plt.tight_layout()
    plt.savefig('clustering_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nVisualization saved as 'clustering_results.png'")
    
    return X_full, labels, model


def performance_analysis_example():
    """Demonstrate performance analysis and benchmarking."""
    print("\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS EXAMPLE")
    print("=" * 60)
    
    import time
    
    # Test different dataset sizes
    dataset_sizes = [100, 500, 1000, 2000]
    configurations = {
        'Basic': {'eps': 0.5, 'min_samples': 5},
        'Adaptive': {'adaptive_eps': True, 'enable_stability_analysis': True},
        'Multi-Density': {'enable_mdbscan': True, 'enable_hierarchical_clustering': True}
    }
    
    results = {}
    
    for size in dataset_sizes:
        print(f"\nTesting with {size} data points...")
        
        # Generate test data
        X_test, _ = make_blobs(n_samples=size, centers=max(3, size//200), 
                              cluster_std=1.0, random_state=42)
        
        size_results = {}
        
        for config_name, config_params in configurations.items():
            try:
                # Create model with configuration
                model = EnhancedAdaptiveDBSCAN(**config_params, random_state=42)
                
                # Measure fitting time
                start_time = time.time()
                model.fit(X_test)
                fit_time = time.time() - start_time
                
                # Calculate metrics
                labels = model.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(X_test, labels)
                else:
                    silhouette = -1
                
                size_results[config_name] = {
                    'fit_time': fit_time,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': silhouette,
                    'throughput': size / fit_time
                }
                
                print(f"  {config_name:12} - Time: {fit_time:.3f}s, "
                      f"Clusters: {n_clusters:2d}, "
                      f"Silhouette: {silhouette:6.3f}, "
                      f"Throughput: {size/fit_time:7.1f} pts/s")
                
            except Exception as e:
                print(f"  {config_name:12} - Failed: {str(e)}")
                size_results[config_name] = None
        
        results[size] = size_results
    
    # Summary
    print(f"\nPerformance Summary:")
    print(f"{'Size':<8} {'Config':<15} {'Time(s)':<8} {'Clusters':<9} {'Silhouette':<11} {'Throughput':<12}")
    print("-" * 70)
    
    for size, size_results in results.items():
        for config_name, metrics in size_results.items():
            if metrics:
                print(f"{size:<8} {config_name:<15} {metrics['fit_time']:<8.3f} "
                      f"{metrics['n_clusters']:<9} {metrics['silhouette']:<11.3f} "
                      f"{metrics['throughput']:<12.1f}")
    
    return results


def main():
    """Run all examples."""
    print("Enhanced Adaptive DBSCAN - Basic Usage Examples")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_results, basic_datasets = basic_clustering_example()
        multi_density_data, labels_std, labels_md, mdbscan_model = multi_density_example()
        complex_data, labels_adaptive, labels_manual, opt_results = adaptive_optimization_example()
        vis_data, vis_labels, vis_model = visualization_example()
        perf_results = performance_analysis_example()
        
        print(f"\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Check the generated 'clustering_results.png' for visualizations.")
        
        return {
            'basic_results': basic_results,
            'multi_density_results': (multi_density_data, labels_std, labels_md),
            'optimization_results': (complex_data, labels_adaptive, opt_results),
            'visualization_results': (vis_data, vis_labels),
            'performance_results': perf_results
        }
        
    except Exception as e:
        print(f"\nError occurred during examples: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\nExample results available in the 'results' dictionary.")
        print(f"Use the returned data for further analysis and experimentation.")

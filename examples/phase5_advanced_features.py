"""
Phase 5 Advanced Features Demonstration
========================================

This example demonstrates the new advanced clustering features added in Phase 5:
1. Deep Learning Clustering with Autoencoders
2. Scalable Indexing for Large Datasets
3. Complete HDBSCAN Hierarchical Clustering
4. Hybrid Approaches
5. Distributed Processing

Requirements:
- torch (for deep learning features)
- annoy or faiss-cpu (for scalable indexing, optional)
"""

import numpy as np
import time
from typing import Dict, Any

# Set random seed for reproducibility
np.random.seed(42)


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def generate_multi_density_data(n_samples: int = 1000) -> np.ndarray:
    """Generate synthetic multi-density wafer defect data."""
    # Dense cluster (high-severity defects)
    dense = np.random.randn(n_samples // 2, 10) * 0.3
    
    # Sparse cluster (low-severity defects)
    sparse = np.random.randn(n_samples // 4, 10) * 0.8 + 4
    
    # Medium density cluster
    medium = np.random.randn(n_samples // 4, 10) * 0.5 + [2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    
    return np.vstack([dense, sparse, medium])


def demo_scalable_clustering():
    """Demonstrate scalable clustering for large datasets."""
    print_section("1. Scalable Clustering with Approximate NN")
    
    from enhanced_adaptive_dbscan import (
        ScalableDBSCAN,
        ScalableIndexManager,
        IndexConfig
    )
    
    # Generate large dataset
    print("Generating large dataset (100,000 points)...")
    X = np.random.randn(100_000, 20)
    
    # Method 1: ScalableDBSCAN (simplest approach)
    print("\nMethod 1: ScalableDBSCAN")
    start_time = time.time()
    
    scalable_dbscan = ScalableDBSCAN(
        eps=0.5,
        min_samples=10,
        n_jobs=-1
    )
    
    labels = scalable_dbscan.fit_predict(X)
    elapsed = time.time() - start_time
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    
    print(f"  ✓ Clustered {len(X):,} points in {elapsed:.2f}s")
    print(f"  ✓ Found {n_clusters} clusters")
    print(f"  ✓ Noise points: {n_noise:,}")
    
    # Method 2: Manual index building for more control
    print("\nMethod 2: Manual Index Building")
    start_time = time.time()
    
    config = IndexConfig(
        method='auto',  # Automatically selects best method
        metric='euclidean'
    )
    
    index_manager = ScalableIndexManager(config)
    index_manager.build_index(X)
    build_time = time.time() - start_time
    
    print(f"  ✓ Built index in {build_time:.2f}s")
    
    # Query neighbors
    start_time = time.time()
    distances, indices = index_manager.query_neighbors(X[:1000], k=10)
    query_time = time.time() - start_time
    
    print(f"  ✓ Queried 1,000 x 10 neighbors in {query_time:.3f}s")
    print(f"  ✓ Average query time: {query_time * 1000 / 1000:.3f}ms per point")


def demo_hdbscan_clustering():
    """Demonstrate HDBSCAN hierarchical clustering."""
    print_section("2. HDBSCAN Hierarchical Clustering")
    
    from enhanced_adaptive_dbscan import HDBSCANClusterer
    
    # Generate multi-density data
    print("Generating multi-density wafer defect data...")
    X = generate_multi_density_data(n_samples=500)
    
    # HDBSCAN clustering
    print("\nRunning HDBSCAN...")
    start_time = time.time()
    
    hdbscan = HDBSCANClusterer(
        min_cluster_size=20,
        min_samples=5,
        metric='euclidean'
    )
    
    labels = hdbscan.fit_predict(X)
    elapsed = time.time() - start_time
    
    # Get detailed cluster information
    info = hdbscan.get_cluster_info()
    
    print(f"  ✓ Clustered {len(X)} points in {elapsed:.2f}s")
    print(f"  ✓ Found {info['n_clusters']} stable clusters")
    print(f"  ✓ Noise points: {info['n_noise']}")
    
    print("\n  Cluster Details:")
    for cluster_id, size in info['cluster_sizes'].items():
        persistence = info['cluster_persistence'].get(cluster_id, 0)
        stability = info.get('selected_stabilities', {}).get(
            hdbscan.selector.selected_clusters_[cluster_id]
            if cluster_id < len(hdbscan.selector.selected_clusters_) else 0,
            0
        )
        print(f"    Cluster {cluster_id}: {size} points, "
              f"persistence={persistence:.3f}, stability={stability:.3f}")


def demo_deep_clustering():
    """Demonstrate deep learning clustering."""
    print_section("3. Deep Learning Clustering")
    
    try:
        from enhanced_adaptive_dbscan import (
            DeepClusteringEngine,
            HybridDeepDBSCAN,
            TORCH_AVAILABLE
        )
        
        if not TORCH_AVAILABLE:
            print("  ⚠ PyTorch not available. Install with: pip install torch")
            return
        
        # Generate high-dimensional data
        print("Generating high-dimensional wafer data (500 samples, 50 features)...")
        X = generate_multi_density_data(n_samples=500)
        
        # Expand to 50 dimensions
        X = np.hstack([X, np.random.randn(500, 40) * 0.1])
        
        # Method 1: Autoencoder-based clustering
        print("\nMethod 1: Autoencoder + K-Means")
        start_time = time.time()
        
        deep_engine = DeepClusteringEngine(
            method='autoencoder',
            latent_dim=10,
            n_clusters=3,
            n_epochs=20,  # Reduced for demo speed
            batch_size=64
        )
        
        result = deep_engine.fit_transform(X)
        elapsed = time.time() - start_time
        
        print(f"  ✓ Trained autoencoder in {elapsed:.2f}s")
        print(f"  ✓ Reconstruction error: {result.reconstruction_error:.4f}")
        print(f"  ✓ Reduced {X.shape[1]}D → {result.embeddings.shape[1]}D")
        print(f"  ✓ Found {len(np.unique(result.labels))} clusters")
        
        # Method 2: Hybrid Deep+DBSCAN
        print("\nMethod 2: Hybrid Deep Learning + DBSCAN")
        start_time = time.time()
        
        hybrid = HybridDeepDBSCAN(
            latent_dim=10,
            n_epochs=20,
            dbscan_params={'min_samples': 5}
        )
        
        labels = hybrid.fit_predict(X)
        elapsed = time.time() - start_time
        
        embeddings = hybrid.get_embeddings(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        
        print(f"  ✓ Hybrid clustering in {elapsed:.2f}s")
        print(f"  ✓ Learned embeddings: {X.shape[1]}D → {embeddings.shape[1]}D")
        print(f"  ✓ Found {n_clusters} clusters + {n_noise} noise points")
        
    except ImportError as e:
        print(f"  ⚠ Deep clustering not available: {e}")
        print("  Install PyTorch with: pip install torch")


def demo_distributed_processing():
    """Demonstrate distributed processing."""
    print_section("4. Distributed Processing")
    
    from enhanced_adaptive_dbscan import (
        DistributedClusteringCoordinator,
        ScalableIndexManager
    )
    
    # Generate large dataset
    print("Generating large dataset (50,000 points)...")
    X = np.random.randn(50_000, 15)
    
    # Initialize coordinator
    print("\nInitializing distributed coordinator...")
    coordinator = DistributedClusteringCoordinator(n_workers=-1)
    
    # Build index
    print("Building index...")
    start_time = time.time()
    index = ScalableIndexManager()
    index.build_index(X)
    build_time = time.time() - start_time
    print(f"  ✓ Index built in {build_time:.2f}s")
    
    # Distributed density estimation
    print("\nComputing densities in parallel...")
    start_time = time.time()
    
    densities = coordinator.distributed_density_estimation(
        X, k=20, index=index
    )
    
    elapsed = time.time() - start_time
    
    print(f"  ✓ Computed densities for {len(X):,} points in {elapsed:.2f}s")
    print(f"  ✓ Using {coordinator.n_workers} workers")
    print(f"  ✓ Throughput: {len(X) / elapsed:,.0f} points/second")
    
    # Show density statistics
    print(f"\n  Density Statistics:")
    print(f"    Min: {np.min(densities):.4f}")
    print(f"    Mean: {np.mean(densities):.4f}")
    print(f"    Max: {np.max(densities):.4f}")
    print(f"    Std: {np.std(densities):.4f}")


def demo_chunked_processing():
    """Demonstrate chunked processing for memory efficiency."""
    print_section("5. Chunked Processing for Large Datasets")
    
    from enhanced_adaptive_dbscan import ChunkedProcessor
    
    # Simulate very large dataset
    print("Simulating processing of 500,000 points in chunks...")
    X = np.random.randn(500_000, 10)
    
    processor = ChunkedProcessor(chunk_size=50_000)
    
    # Example: compute mean in chunks
    def compute_mean_chunk(chunk, state):
        """Compute sum and count for chunk."""
        return {
            'sum': np.sum(chunk, axis=0),
            'count': len(chunk)
        }
    
    def combine_results(state, result):
        """Combine chunk results."""
        if state is None:
            return result
        return {
            'sum': state['sum'] + result['sum'],
            'count': state['count'] + result['count']
        }
    
    start_time = time.time()
    result = processor.process_chunked(
        X,
        compute_mean_chunk,
        combine_results,
        initial_state=None
    )
    elapsed = time.time() - start_time
    
    mean = result['sum'] / result['count']
    
    print(f"  ✓ Processed {len(X):,} points in {elapsed:.2f}s")
    print(f"  ✓ Used 10 chunks of 50,000 points each")
    print(f"  ✓ Computed mean: {mean[:3]} ... (first 3 features)")
    
    # Verify against numpy
    true_mean = np.mean(X, axis=0)
    error = np.max(np.abs(mean - true_mean))
    print(f"  ✓ Maximum error vs numpy: {error:.10f}")


def demo_performance_comparison():
    """Compare performance across different methods."""
    print_section("6. Performance Comparison")
    
    from enhanced_adaptive_dbscan import (
        ScalableDBSCAN,
        HDBSCANClusterer,
        EnhancedAdaptiveDBSCAN
    )
    
    # Generate test data
    print("Generating test dataset (10,000 points)...")
    X = generate_multi_density_data(n_samples=10_000)
    
    results: Dict[str, Dict[str, Any]] = {}
    
    # 1. Traditional Enhanced Adaptive DBSCAN
    print("\n1. Enhanced Adaptive DBSCAN (Phase 1)")
    start_time = time.time()
    
    adaptive_dbscan = EnhancedAdaptiveDBSCAN(
        k=20,
        stability_threshold=0.6
    )
    labels_adaptive = adaptive_dbscan.fit_predict(X)
    
    results['Adaptive DBSCAN'] = {
        'time': time.time() - start_time,
        'n_clusters': len(set(labels_adaptive)) - (1 if -1 in labels_adaptive else 0),
        'n_noise': np.sum(labels_adaptive == -1)
    }
    
    # 2. Scalable DBSCAN
    print("2. Scalable DBSCAN (Phase 5)")
    start_time = time.time()
    
    scalable = ScalableDBSCAN(eps=0.5, min_samples=10)
    labels_scalable = scalable.fit_predict(X)
    
    results['Scalable DBSCAN'] = {
        'time': time.time() - start_time,
        'n_clusters': len(set(labels_scalable)) - (1 if -1 in labels_scalable else 0),
        'n_noise': np.sum(labels_scalable == -1)
    }
    
    # 3. HDBSCAN
    print("3. HDBSCAN (Phase 5)")
    start_time = time.time()
    
    hdbscan = HDBSCANClusterer(min_cluster_size=50)
    labels_hdbscan = hdbscan.fit_predict(X)
    
    results['HDBSCAN'] = {
        'time': time.time() - start_time,
        'n_clusters': len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0),
        'n_noise': np.sum(labels_hdbscan == -1)
    }
    
    # Print comparison
    print("\n" + "-" * 70)
    print(f"{'Method':<25} {'Time (s)':<12} {'Clusters':<12} {'Noise':<12}")
    print("-" * 70)
    
    for method, stats in results.items():
        print(f"{method:<25} {stats['time']:<12.3f} {stats['n_clusters']:<12} {stats['n_noise']:<12}")
    
    print("-" * 70)


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  ENHANCED ADAPTIVE DBSCAN - PHASE 5 FEATURES DEMO")
    print("=" * 70)
    print("\nDemonstrating advanced clustering capabilities:")
    print("  • Scalable clustering for large datasets")
    print("  • HDBSCAN hierarchical clustering")
    print("  • Deep learning clustering")
    print("  • Distributed processing")
    print("  • Chunked processing")
    print("  • Performance comparison")
    
    try:
        # Run all demos
        demo_scalable_clustering()
        demo_hdbscan_clustering()
        demo_deep_clustering()
        demo_distributed_processing()
        demo_chunked_processing()
        demo_performance_comparison()
        
        # Summary
        print_section("Summary")
        print("✓ All Phase 5 features demonstrated successfully!")
        print("\nKey Achievements:")
        print("  • Scalable indexing handles millions of points efficiently")
        print("  • HDBSCAN provides hierarchical clustering with stability")
        print("  • Deep learning enables representation learning")
        print("  • Distributed processing leverages multi-core systems")
        print("  • Chunked processing manages memory for very large datasets")
        print("\nNext Steps:")
        print("  • Explore parameter tuning for your specific data")
        print("  • Try combining multiple approaches (hybrid methods)")
        print("  • Scale to your full production datasets")
        print("  • Integrate with existing MLOps pipelines")
        
    except Exception as e:
        print(f"\n⚠ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

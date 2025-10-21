"""
Quick Demo: Phase 5 Advanced Features
======================================

A quick demonstration of the new Phase 5 features with smaller datasets.
"""

import numpy as np
import time

np.random.seed(42)


def demo_hdbscan():
    """Quick HDBSCAN demo."""
    print("\n=== HDBSCAN Hierarchical Clustering ===")
    
    from enhanced_adaptive_dbscan import HDBSCANClusterer
    
    # Multi-density data
    X = np.vstack([
        np.random.randn(50, 2) * 0.3,       # Dense
        np.random.randn(30, 2) * 0.8 + 5,   # Sparse
    ])
    
    hdbscan = HDBSCANClusterer(min_cluster_size=15, min_samples=5)
    labels = hdbscan.fit_predict(X)
    
    info = hdbscan.get_cluster_info()
    print(f"✓ Found {info['n_clusters']} stable clusters")
    print(f"✓ Noise points: {info['n_noise']}")


def demo_scalable():
    """Quick scalable clustering demo."""
    print("\n=== Scalable Clustering ===")
    
    from enhanced_adaptive_dbscan import ScalableDBSCAN
    
    X = np.random.randn(5000, 10)
    
    start = time.time()
    scalable = ScalableDBSCAN(eps=0.5, min_samples=10)
    labels = scalable.fit_predict(X)
    elapsed = time.time() - start
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✓ Clustered {len(X)} points in {elapsed:.2f}s")
    print(f"✓ Found {n_clusters} clusters")


def demo_distributed():
    """Quick distributed processing demo."""
    print("\n=== Distributed Processing ===")
    
    from enhanced_adaptive_dbscan import (
        DistributedClusteringCoordinator,
        ScalableIndexManager
    )
    
    X = np.random.randn(2000, 10)
    
    coordinator = DistributedClusteringCoordinator(n_workers=2)
    index = ScalableIndexManager()
    index.build_index(X)
    
    start = time.time()
    densities = coordinator.distributed_density_estimation(X, k=10, index=index)
    elapsed = time.time() - start
    
    print(f"✓ Computed densities for {len(X)} points in {elapsed:.2f}s")
    print(f"✓ Using {coordinator.n_workers} workers")


def demo_deep():
    """Quick deep learning demo."""
    print("\n=== Deep Learning Clustering ===")
    
    try:
        from enhanced_adaptive_dbscan import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            print("⚠ PyTorch not available")
            return
        
        from enhanced_adaptive_dbscan import DeepClusteringEngine
        
        X = np.random.randn(200, 20)
        
        engine = DeepClusteringEngine(
            method='autoencoder',
            latent_dim=5,
            n_clusters=3,
            n_epochs=10
        )
        
        result = engine.fit_transform(X)
        print(f"✓ Reduced {X.shape[1]}D → {result.embeddings.shape[1]}D")
        print(f"✓ Found {len(np.unique(result.labels))} clusters")
        
    except Exception as e:
        print(f"⚠ Deep clustering demo skipped: {e}")


def main():
    print("\n" + "=" * 60)
    print("  PHASE 5 FEATURES - QUICK DEMO")
    print("=" * 60)
    
    demo_hdbscan()
    demo_scalable()
    demo_distributed()
    demo_deep()
    
    print("\n" + "=" * 60)
    print("✓ Quick demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

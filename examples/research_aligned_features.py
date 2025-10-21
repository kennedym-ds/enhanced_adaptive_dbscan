#!/usr/bin/env python3
"""
Example: Using New Research-Aligned Features (2024-2025)

This example demonstrates the new features added based on latest clustering research:
1. predict() method for new data points (HDBSCAN 2024 alignment)
2. K-distance graph analysis (X-DBSCAN 2024)
3. Automatic parameter suggestion

Author: Enhanced Adaptive DBSCAN Development Team
Date: October 2025
"""

import numpy as np
from enhanced_adaptive_dbscan import (
    EnhancedAdaptiveDBSCAN,
    compute_kdist_graph,
    find_kdist_elbow,
    suggest_dbscan_parameters
)

def main():
    print("=" * 80)
    print("Enhanced Adaptive DBSCAN - New Research-Aligned Features Demo")
    print("=" * 80)
    print()
    
    # =========================================================================
    # Example 1: Automatic Parameter Suggestion using K-Distance Graph
    # =========================================================================
    print("Example 1: Automatic Parameter Suggestion")
    print("-" * 80)
    
    # Generate synthetic data with two clusters
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.5, (100, 2)),
        np.random.normal(4, 0.5, (100, 2))
    ])
    
    print(f"Dataset: {len(X)} points, 2 dimensions")
    print()
    
    # Method 1: Compute k-distance graph manually
    print("Method 1: Manual k-distance analysis")
    k_distances = compute_kdist_graph(X, k=5)
    optimal_eps, elbow_idx = find_kdist_elbow(k_distances, method='kneedle')
    print(f"  K-distance graph computed for k=5")
    print(f"  Suggested eps (elbow detection): {optimal_eps:.4f}")
    print(f"  Elbow point at index: {elbow_idx}")
    print()
    
    # Method 2: Automatic parameter suggestion (recommended)
    print("Method 2: Automatic parameter suggestion (recommended)")
    params = suggest_dbscan_parameters(X, k_range=(4, 15), n_trials=5)
    print(f"  Suggested parameters:")
    print(f"    eps: {params['eps']:.4f}")
    print(f"    min_samples: {params['min_samples']}")
    print(f"    confidence: {params['confidence']:.2f}")
    print(f"    eps range: [{params['suggested_eps_range'][0]:.4f}, "
          f"{params['suggested_eps_range'][1]:.4f}]")
    print()
    
    # =========================================================================
    # Example 2: Train/Test Split with predict() Method
    # =========================================================================
    print("=" * 80)
    print("Example 2: Train/Test Split with predict() Method")
    print("-" * 80)
    
    # Split data into train and test
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    train_size = int(0.7 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    
    print(f"Training set: {len(X_train)} points")
    print(f"Test set: {len(X_test)} points")
    print()
    
    # Train model using suggested parameters
    print("Training model with suggested parameters...")
    model = EnhancedAdaptiveDBSCAN(
        k=params['min_samples'],
        density_scaling=params['eps'],
        stability_threshold=0.5
    )
    model.fit(X_train)
    
    train_labels = model.labels_
    n_clusters_train = len(set(train_labels)) - (1 if -1 in train_labels else 0)
    n_noise_train = np.sum(train_labels == -1)
    
    print(f"Training results:")
    print(f"  Clusters found: {n_clusters_train}")
    print(f"  Noise points: {n_noise_train}")
    print()
    
    # Predict on test data using new predict() method
    print("Predicting on test data using predict() method...")
    test_labels = model.predict(X_test)
    
    n_clusters_test = len(set(test_labels)) - (1 if -1 in test_labels else 0)
    n_noise_test = np.sum(test_labels == -1)
    
    print(f"Test results:")
    print(f"  Clusters assigned: {n_clusters_test}")
    print(f"  Noise points: {n_noise_test}")
    print(f"  Clustering rate: {100 * (1 - n_noise_test/len(X_test)):.1f}%")
    print()
    
    # =========================================================================
    # Example 3: Online/Streaming Scenario
    # =========================================================================
    print("=" * 80)
    print("Example 3: Online/Streaming Prediction Scenario")
    print("-" * 80)
    
    # Train on initial batch
    X_initial = np.vstack([
        np.random.normal(0, 0.5, (80, 2)),
        np.random.normal(4, 0.5, (80, 2))
    ])
    
    print("Training on initial batch...")
    model_streaming = EnhancedAdaptiveDBSCAN(k=10, density_scaling=0.6)
    model_streaming.fit(X_initial)
    print(f"  Initial clusters: {len(set(model_streaming.labels_)) - (1 if -1 in model_streaming.labels_ else 0)}")
    print()
    
    # Simulate streaming data arriving in batches
    print("Processing streaming data batches...")
    for batch_num in range(3):
        # Generate new batch
        new_batch = np.vstack([
            np.random.normal(0, 0.4, (5, 2)),
            np.random.normal(4, 0.4, (5, 2))
        ])
        
        # Predict cluster assignments for new batch
        batch_labels = model_streaming.predict(new_batch)
        n_assigned = np.sum(batch_labels != -1)
        
        print(f"  Batch {batch_num + 1}: {len(new_batch)} points, "
              f"{n_assigned} assigned to clusters, "
              f"{len(new_batch) - n_assigned} marked as noise")
    print()
    
    # =========================================================================
    # Example 4: Complete Workflow with Parameter Optimization
    # =========================================================================
    print("=" * 80)
    print("Example 4: Complete Workflow - Suggest → Train → Predict → Evaluate")
    print("-" * 80)
    
    # Generate labeled data for evaluation
    np.random.seed(123)
    X_cluster1 = np.random.normal(0, 0.5, (100, 2))
    X_cluster2 = np.random.normal(5, 0.5, (100, 2))
    X_cluster3 = np.random.normal([2.5, 4], 0.6, (100, 2))
    X_full = np.vstack([X_cluster1, X_cluster2, X_cluster3])
    true_labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    # Split into train/test
    indices = np.arange(len(X_full))
    np.random.shuffle(indices)
    train_size = int(0.8 * len(X_full))
    
    X_train_full = X_full[indices[:train_size]]
    X_test_full = X_full[indices[train_size:]]
    true_test_labels = true_labels[indices[train_size:]]
    
    print("Step 1: Suggest parameters...")
    suggested = suggest_dbscan_parameters(X_train_full, k_range=(5, 20), n_trials=7)
    print(f"  Suggested eps: {suggested['eps']:.4f}")
    print(f"  Suggested min_samples: {suggested['min_samples']}")
    print(f"  Confidence: {suggested['confidence']:.2f}")
    print()
    
    print("Step 2: Train model...")
    final_model = EnhancedAdaptiveDBSCAN(
        k=suggested['min_samples'],
        density_scaling=suggested['eps'],
        stability_threshold=0.4
    )
    final_model.fit(X_train_full)
    print(f"  Clusters found: {len(set(final_model.labels_)) - (1 if -1 in final_model.labels_ else 0)}")
    print()
    
    print("Step 3: Predict on test set...")
    final_test_labels = final_model.predict(X_test_full)
    n_test_clusters = len(set(final_test_labels)) - (1 if -1 in final_test_labels else 0)
    print(f"  Test clusters: {n_test_clusters}")
    print(f"  Test noise: {np.sum(final_test_labels == -1)}")
    print()
    
    print("Step 4: Evaluate predictions...")
    # Calculate clustering accuracy (ignoring noise)
    non_noise_mask = final_test_labels != -1
    if np.any(non_noise_mask):
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(
            true_test_labels[non_noise_mask],
            final_test_labels[non_noise_mask]
        )
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Clustering rate: {100 * np.mean(non_noise_mask):.1f}%")
    print()
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 80)
    print("Summary of New Features")
    print("=" * 80)
    print()
    print("✅ K-Distance Graph Analysis (X-DBSCAN 2024)")
    print("   - compute_kdist_graph(): Compute k-distances for all points")
    print("   - find_kdist_elbow(): Detect elbow point for optimal eps")
    print("   - suggest_dbscan_parameters(): Automatic parameter suggestion")
    print()
    print("✅ predict() Method (HDBSCAN 2024 Alignment)")
    print("   - Predict cluster labels for new data points")
    print("   - Uses adaptive k-NN voting for cluster assignment")
    print("   - Enables proper train/test split workflows")
    print("   - Supports streaming/online prediction scenarios")
    print()
    print("✅ Benefits")
    print("   - Automatic parameter selection reduces manual tuning")
    print("   - predict() enables proper ML pipeline integration")
    print("   - Compatible with scikit-learn cross-validation")
    print("   - Aligns with latest clustering research (2024-2025)")
    print()
    print("For more information, see RESEARCH_COMPARISON_ANALYSIS.md")
    print("=" * 80)

if __name__ == "__main__":
    main()

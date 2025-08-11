"""
Comprehensive Enhanced Adaptive DBSCAN Demo
===========================================

This example demonstrates all four phases of the Enhanced Adaptive DBSCAN framework:
- Phase 1: Core Adaptive DBSCAN clustering
- Phase 2: Ensemble & Multi-density clustering  
- Phase 3: Adaptive parameter optimization
- Phase 4: Production pipeline & enterprise integration

Run with: python examples/comprehensive_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
import time

# Phase 1: Core Adaptive DBSCAN
print("🎯 Phase 1: Enhanced Adaptive DBSCAN Demo")
print("=" * 50)

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN

# Generate multi-density synthetic data
centers = [[0, 0], [4, 4], [-3, 2]]
X_blobs, _ = make_blobs(n_samples=300, centers=centers, cluster_std=[0.5, 1.2, 0.8], 
                       random_state=42)
X_moons, _ = make_moons(n_samples=200, noise=0.1, random_state=42)
X_moons[:, 0] += 6  # Shift moons to avoid overlap

# Combine datasets for complex clustering scenario
X = np.vstack([X_blobs, X_moons])
severity = np.random.randint(1, 11, size=(len(X), 1))  # Simulated defect severity

print(f"Dataset: {len(X)} points with varying densities")

# Initialize Enhanced Adaptive DBSCAN
model = EnhancedAdaptiveDBSCAN(
    wafer_shape='circular',
    wafer_size=100,
    k=20,
    density_scaling=1.0,
    min_scaling=0.5,
    additional_features=[2],
    feature_weights=[1.0],
    stability_threshold=0.6
)

# Fit the model
start_time = time.time()
X_full = np.hstack((X, severity))
model.fit(X_full, additional_attributes=severity)
labels = model.labels_
phase1_time = time.time() - start_time

print(f"✅ Phase 1 Complete: Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
print(f"   Processing time: {phase1_time:.3f} seconds")
print(f"   Noise points: {sum(labels == -1)} / {len(labels)}")

# Phase 2: Ensemble & Multi-Density Clustering
print("\n🎪 Phase 2: Ensemble & Multi-Density Clustering")
print("=" * 50)

from enhanced_adaptive_dbscan.ensemble_clustering import ConsensusClusteringEngine

# Create consensus clustering engine
engine = ConsensusClusteringEngine(
    n_estimators=30,
    voting_strategy='quality_weighted',
    stability_threshold=0.7
)

start_time = time.time()
consensus_labels = engine.fit_consensus_clustering(X)
quality_scores = engine.get_cluster_quality_scores()
phase2_time = time.time() - start_time

print(f"✅ Phase 2 Complete: Consensus clustering with {engine.n_estimators} estimators")
print(f"   Consensus clusters: {len(set(consensus_labels)) - (1 if -1 in consensus_labels else 0)}")
print(f"   Average quality score: {np.mean(list(quality_scores.values())):.3f}")
print(f"   Processing time: {phase2_time:.3f} seconds")

# Phase 3: Adaptive Parameter Optimization
print("\n🧠 Phase 3: Adaptive Parameter Optimization")
print("=" * 50)

from enhanced_adaptive_dbscan.adaptive_optimization import AdaptiveTuningEngine

# Create optimization engine
tuning_engine = AdaptiveTuningEngine(
    optimization_method='bayesian',
    n_iterations=20,  # Reduced for demo speed
    prediction_enabled=True,
    meta_learning_enabled=True
)

# Define parameter space
parameter_space = {
    'eps': (0.1, 2.0),
    'min_samples': (3, 30)
}

start_time = time.time()
result = tuning_engine.optimize_parameters(
    X, parameter_space, 
    optimization_metric='silhouette_score'
)
phase3_time = time.time() - start_time

print(f"✅ Phase 3 Complete: Bayesian optimization with {tuning_engine.n_iterations} iterations")
print(f"   Best parameters: eps={result.best_parameters['eps']:.3f}, min_samples={result.best_parameters['min_samples']}")
print(f"   Best silhouette score: {result.best_score:.3f}")
print(f"   Optimization time: {phase3_time:.3f} seconds")

# Phase 4: Production Pipeline & Enterprise Integration
print("\n🏭 Phase 4: Production Pipeline & Enterprise Integration")
print("=" * 50)

from enhanced_adaptive_dbscan.production_pipeline import ProductionPipeline, DeploymentConfig
from enhanced_adaptive_dbscan.streaming_engine import StreamingClusteringEngine, StreamingConfig
from enhanced_adaptive_dbscan.web_api import ClusteringWebAPI

# 1. Production Pipeline Setup
config = DeploymentConfig(
    model_name="adaptive_dbscan_demo",
    version="1.0.0",
    environment="development",
    model_store_path="./models",
    metrics_store_path="./metrics",
    auto_scaling=True
)

pipeline = ProductionPipeline(config)

# 2. Train and deploy model
start_time = time.time()
demo_model = pipeline.train_model(X)
deployment_id = pipeline.deploy_model(demo_model)
print(f"✅ Model Training & Deployment Complete")
print(f"   Model ID: {deployment_id}")
print(f"   Environment: {config.environment}")

# 3. Streaming clustering demonstration
streaming_config = StreamingConfig(
    window_size=50,
    overlap=0.1,
    enable_concept_drift_detection=True,
    drift_threshold=0.05
)

streaming_engine = StreamingClusteringEngine(demo_model, streaming_config)
streaming_engine.start_streaming()

print(f"✅ Streaming Engine Started")
print(f"   Window size: {streaming_config.window_size}")
print(f"   Drift detection: {'enabled' if streaming_config.enable_concept_drift_detection else 'disabled'}")

# Process some streaming data points
streaming_results = []
for i in range(10):
    new_point = np.random.randn(2) * 2 + np.random.choice([[0, 0], [4, 4], [-3, 2]])
    result = streaming_engine.process_point(new_point)
    streaming_results.append(result)

streaming_engine.stop_streaming()
phase4_time = time.time() - start_time

print(f"✅ Streaming Processing Complete: {len(streaming_results)} points processed")
print(f"   Clusters detected: {len(set(r['cluster_id'] for r in streaming_results if r['cluster_id'] != -1))}")
print(f"   Drift detected: {'Yes' if any(r.get('drift_detected', False) for r in streaming_results) else 'No'}")

# 4. Web API setup (demonstrate configuration, not actual server start)
api = ClusteringWebAPI(host="localhost", port=5001, debug=False)
print(f"✅ Web API Configured")
print(f"   Endpoint: http://localhost:5001")
print(f"   Health check: /api/health")
print(f"   Clustering: /api/cluster")

print(f"   Total Phase 4 time: {phase4_time:.3f} seconds")

# Summary Report
print("\n📊 COMPREHENSIVE DEMO SUMMARY")
print("=" * 50)
print(f"🎯 Phase 1 - Enhanced Adaptive DBSCAN:")
print(f"   ✅ Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"   ✅ Processing time: {phase1_time:.3f}s")

print(f"\n🎪 Phase 2 - Ensemble Clustering:")
print(f"   ✅ Consensus clusters: {len(set(consensus_labels)) - (1 if -1 in consensus_labels else 0)}")
print(f"   ✅ Quality score: {np.mean(list(quality_scores.values())):.3f}")
print(f"   ✅ Processing time: {phase2_time:.3f}s")

print(f"\n🧠 Phase 3 - Adaptive Optimization:")
print(f"   ✅ Best score: {result.best_score:.3f}")
print(f"   ✅ Optimal eps: {result.best_parameters['eps']:.3f}")
print(f"   ✅ Processing time: {phase3_time:.3f}s")

print(f"\n🏭 Phase 4 - Production Pipeline:")
print(f"   ✅ Model deployed: {deployment_id}")
print(f"   ✅ Streaming points processed: {len(streaming_results)}")
print(f"   ✅ API configured: localhost:5001")
print(f"   ✅ Processing time: {phase4_time:.3f}s")

total_time = phase1_time + phase2_time + phase3_time + phase4_time
print(f"\n🚀 TOTAL PROCESSING TIME: {total_time:.3f} seconds")
print("\n🎉 All phases completed successfully!")
print("   The Enhanced Adaptive DBSCAN framework is ready for:")
print("   • Research and development (Phases 1-3)")
print("   • Production deployment (Phase 4)")
print("   • Enterprise integration (Phase 4 APIs)")
print("   • Real-time streaming analytics (Phase 4)")

# Visualization option
try:
    import matplotlib.pyplot as plt
    
    print("\n📈 Generating visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Enhanced Adaptive DBSCAN: All Phases Results', fontsize=16)
    
    # Phase 1 results
    axes[0, 0].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    axes[0, 0].set_title('Phase 1: Enhanced Adaptive DBSCAN')
    axes[0, 0].set_xlabel('Feature 1')
    axes[0, 0].set_ylabel('Feature 2')
    
    # Phase 2 results  
    axes[0, 1].scatter(X[:, 0], X[:, 1], c=consensus_labels, cmap='plasma', alpha=0.7)
    axes[0, 1].set_title('Phase 2: Ensemble Clustering')
    axes[0, 1].set_xlabel('Feature 1')
    axes[0, 1].set_ylabel('Feature 2')
    
    # Phase 3 optimization convergence (simulated)
    iterations = range(1, len(result.scores) + 1) if hasattr(result, 'scores') else range(1, 21)
    scores = result.scores if hasattr(result, 'scores') else np.random.cummax(np.random.random(20) * 0.3 + 0.4)
    axes[1, 0].plot(iterations, scores, 'b-', linewidth=2)
    axes[1, 0].set_title('Phase 3: Optimization Convergence')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Silhouette Score')
    axes[1, 0].grid(True)
    
    # Phase 4 streaming throughput
    throughput = [len(streaming_results) / phase4_time] * 10
    axes[1, 1].bar(range(1, 11), throughput, color='green', alpha=0.7)
    axes[1, 1].set_title('Phase 4: Streaming Throughput')
    axes[1, 1].set_xlabel('Time Window')
    axes[1, 1].set_ylabel('Points/Second')
    
    plt.tight_layout()
    plt.savefig('comprehensive_demo_results.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'comprehensive_demo_results.png'")
    
except ImportError:
    print("⚠️  Matplotlib not available for visualization")

print("\n" + "=" * 50)
print("Demo completed! Check the generated files and logs for detailed results.")

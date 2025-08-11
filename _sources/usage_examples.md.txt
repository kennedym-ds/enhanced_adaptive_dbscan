# Usage Examples

This guide provides comprehensive examples demonstrating the Enhanced Adaptive DBSCAN framework's capabilities.

## Basic Clustering

### Simple Dataset

```python
import numpy as np
import matplotlib.pyplot as plt
from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN

# Generate sample data with multiple density regions
np.random.seed(42)
data1 = np.random.normal(0, 0.5, (50, 2))
data2 = np.random.normal(3, 0.8, (80, 2))
data3 = np.random.normal([1, 3], 0.3, (40, 2))
X = np.vstack([data1, data2, data3])

# Basic clustering
clusterer = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=5)
labels = clusterer.fit_predict(X)

# Visualize
clusterer.plot(title="Basic Clustering Example")
plt.show()
```

### Parameter Optimization

```python
from enhanced_adaptive_dbscan.adaptive_optimization import AdaptiveTuningEngine

# Create optimization engine
optimizer = AdaptiveTuningEngine()

# Optimize parameters
best_params, optimization_history = optimizer.optimize_parameters(
    X, 
    max_iterations=20,
    optimization_strategy='bayesian'
)

print(f"Best parameters found: {best_params}")

# Use optimized parameters
clusterer = EnhancedAdaptiveDBSCAN(**best_params)
optimized_labels = clusterer.fit_predict(X)
```

## Multi-Density Clustering

### Varying Density Regions

```python
# Generate data with different density regions
np.random.seed(123)
dense_region = np.random.normal([0, 0], 0.2, (100, 2))
sparse_region = np.random.normal([3, 3], 0.8, (30, 2))
medium_region = np.random.normal([0, 3], 0.5, (60, 2))

X_multi = np.vstack([dense_region, sparse_region, medium_region])

# Enable multi-density clustering
clusterer = EnhancedAdaptiveDBSCAN(
    enable_mdbscan=True,
    density_adaptation=True,
    stability_threshold=0.3
)

labels = clusterer.fit_predict(X_multi)

# Analyze density regions
if hasattr(clusterer, 'get_density_analysis'):
    density_info = clusterer.get_density_analysis()
    print(f"Density regions identified: {len(density_info['regions'])}")
    
    for i, region in enumerate(density_info['regions']):
        print(f"Region {i}: density={region['density']:.3f}, points={region['n_points']}")
```

### Hierarchical Clustering Analysis

```python
# Access hierarchical clustering components (if available)
if hasattr(clusterer, 'hierarchy_manager_'):
    # Get hierarchical analysis
    hierarchy_info = clusterer.hierarchy_manager_.get_hierarchy_info()
    
    # Plot stability scores across different scales
    plt.figure(figsize=(10, 6))
    scales = hierarchy_info['scales']
    stability_scores = hierarchy_info['stability_scores']
    
    plt.plot(scales, stability_scores, 'bo-')
    plt.xlabel('Scale')
    plt.ylabel('Stability Score')
    plt.title('Hierarchical Clustering Stability Analysis')
    plt.grid(True)
    plt.show()
```

## Ensemble Clustering

### Consensus Clustering

```python
from enhanced_adaptive_dbscan.ensemble_clustering import ConsensusClusteringEngine

# Create ensemble with different parameter sets
ensemble = ConsensusClusteringEngine(
    base_clusterer=EnhancedAdaptiveDBSCAN(),
    n_estimators=15,
    parameter_diversity_threshold=0.3,
    voting_strategy='quality_weighted'
)

# Fit consensus clustering
consensus_labels = ensemble.fit_consensus_clustering(X)

# Get ensemble statistics
ensemble_stats = ensemble.get_ensemble_statistics()
print(f"Number of base clusterers: {ensemble_stats['n_estimators']}")
print(f"Average quality score: {ensemble_stats['avg_quality']:.3f}")
print(f"Consensus stability: {ensemble_stats['consensus_stability']:.3f}")

# Visualize ensemble results
ensemble.plot_consensus_results(X, consensus_labels)
```

### Parameter Diversity Analysis

```python
# Analyze parameter diversity in ensemble
parameter_sets = ensemble.get_parameter_sets()
diversity_matrix = ensemble.calculate_parameter_diversity()

# Plot parameter diversity
plt.figure(figsize=(12, 8))
plt.imshow(diversity_matrix, cmap='viridis')
plt.colorbar(label='Parameter Diversity')
plt.title('Ensemble Parameter Diversity Matrix')
plt.xlabel('Estimator Index')
plt.ylabel('Estimator Index')
plt.show()
```

## Streaming and Production

### Streaming Data Processing

```python
from enhanced_adaptive_dbscan.streaming_engine import StreamingEngine
from enhanced_adaptive_dbscan.streaming_config import StreamingConfig

# Configure streaming
config = StreamingConfig(
    batch_size=50,
    update_frequency=5,
    concept_drift_detection=True,
    drift_threshold=0.1
)

# Create streaming engine
streaming_engine = StreamingEngine(
    base_clusterer=EnhancedAdaptiveDBSCAN(),
    config=config
)

# Simulate streaming data
for batch_id in range(10):
    # Generate batch data
    batch_data = np.random.normal([batch_id * 0.1, 0], 0.5, (50, 2))
    
    # Process batch
    batch_labels = streaming_engine.process_batch(batch_data)
    
    # Check for concept drift
    if streaming_engine.concept_drift_detector.drift_detected:
        print(f"Concept drift detected at batch {batch_id}")
        streaming_engine.handle_concept_drift()
```

### Production Pipeline

```python
from enhanced_adaptive_dbscan.production_pipeline import ProductionPipeline
from enhanced_adaptive_dbscan.deployment_config import DeploymentConfig

# Configure deployment
deployment_config = DeploymentConfig(
    model_validation_threshold=0.7,
    performance_monitoring=True,
    auto_retraining=True,
    model_versioning=True
)

# Create production pipeline
pipeline = ProductionPipeline(deployment_config)

# Train and deploy model
model_id = pipeline.train_and_deploy(
    training_data=X,
    model_params={'eps': 0.5, 'min_samples': 5}
)

print(f"Model deployed with ID: {model_id}")

# Monitor performance
performance_metrics = pipeline.monitor_performance(model_id)
print(f"Model performance: {performance_metrics}")
```

## Advanced Optimization

### Grid Search with Validation

```python
from enhanced_adaptive_dbscan.adaptive_optimization import ParameterSpaceExplorer

# Define parameter space
param_space = {
    'eps': [0.1, 0.3, 0.5, 0.7],
    'min_samples': [3, 5, 7, 10],
    'adaptive': [True, False]
}

# Create explorer
explorer = ParameterSpaceExplorer()

# Explore parameter space
results = explorer.explore_parameter_space(
    X, 
    param_space, 
    exploration_strategy='grid_search',
    cv_folds=5
)

# Get best configuration
best_config = results['best_params']
best_score = results['best_score']

print(f"Best configuration: {best_config}")
print(f"Best score: {best_score:.3f}")

# Plot parameter space exploration
explorer.plot_parameter_space_results(results)
```

### Bayesian Optimization

```python
from enhanced_adaptive_dbscan.adaptive_optimization import BayesianOptimizer

# Create Bayesian optimizer
bayesian_opt = BayesianOptimizer(
    acquisition_function='expected_improvement',
    n_initial_points=10
)

# Define optimization bounds
bounds = {
    'eps': (0.05, 1.0),
    'min_samples': (2, 20)
}

# Run optimization
optimization_result = bayesian_opt.optimize(
    X, 
    bounds,
    n_iterations=30
)

# Plot optimization progress
bayesian_opt.plot_optimization_progress()
```

## Performance Analysis

### Comprehensive Benchmarking

```python
from enhanced_adaptive_dbscan.adaptive_optimization import DBSCANOptimizer

# Create optimizer for benchmarking
optimizer = DBSCANOptimizer()

# Run comprehensive benchmarking
benchmark_results = optimizer.performance_benchmarking_example(
    dataset_sizes=[100, 500, 1000, 2000],
    n_features_list=[2, 5, 10],
    n_trials=5
)

# Analyze results
for size, results in benchmark_results.items():
    mean_time = np.mean(results['execution_times'])
    mean_silhouette = np.mean(results['silhouette_scores'])
    print(f"Size {size}: Time={mean_time:.3f}s, Silhouette={mean_silhouette:.3f}")

# Plot performance scaling
optimizer.plot_performance_scaling(benchmark_results)
```

### Memory Usage Analysis

```python
import psutil
import os

def monitor_memory_usage(func, *args, **kwargs):
    """Monitor memory usage during function execution."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func(*args, **kwargs)
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_delta = final_memory - initial_memory
    
    return result, memory_delta

# Monitor clustering memory usage
clusterer = EnhancedAdaptiveDBSCAN()
labels, memory_used = monitor_memory_usage(clusterer.fit_predict, X)

print(f"Memory used during clustering: {memory_used:.2f} MB")
```

## Real-World Applications

### Anomaly Detection

```python
# Simulate network traffic data
np.random.seed(456)
normal_traffic = np.random.normal([100, 50], [10, 5], (1000, 2))
anomalous_traffic = np.random.normal([200, 150], [50, 20], (50, 2))
network_data = np.vstack([normal_traffic, anomalous_traffic])

# Configure for anomaly detection
anomaly_detector = EnhancedAdaptiveDBSCAN(
    eps=15,
    min_samples=10,
    adaptive=True
)

# Fit and identify anomalies
labels = anomaly_detector.fit_predict(network_data)
anomalies = network_data[labels == -1]

print(f"Anomalies detected: {len(anomalies)} out of {len(network_data)} points")
print(f"Anomaly rate: {len(anomalies)/len(network_data)*100:.2f}%")

# Visualize anomalies
plt.figure(figsize=(12, 8))
plt.scatter(network_data[:, 0], network_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(anomalies[:, 0], anomalies[:, 1], c='red', marker='x', s=100, label='Anomalies')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Network Traffic Anomaly Detection')
plt.legend()
plt.show()
```

### Customer Segmentation

```python
# Simulate customer behavior data
np.random.seed(789)
customers = []

# High-value customers
high_value = np.random.normal([80, 90], [5, 10], (200, 2))
customers.append(high_value)

# Medium-value customers  
medium_value = np.random.normal([50, 60], [10, 15], (500, 2))
customers.append(medium_value)

# Low-value customers
low_value = np.random.normal([20, 30], [8, 12], (300, 2))
customers.append(low_value)

customer_data = np.vstack(customers)

# Segment customers
segmentation = EnhancedAdaptiveDBSCAN(
    adaptive=True,
    enable_mdbscan=True
)

segments = segmentation.fit_predict(customer_data)

# Analyze segments
unique_segments = set(segments) - {-1}
print(f"Customer segments identified: {len(unique_segments)}")

for segment_id in unique_segments:
    segment_mask = segments == segment_id
    segment_customers = customer_data[segment_mask]
    avg_value = np.mean(segment_customers, axis=0)
    print(f"Segment {segment_id}: {len(segment_customers)} customers, "
          f"avg value=[{avg_value[0]:.1f}, {avg_value[1]:.1f}]")

# Visualize segmentation
segmentation.plot(title="Customer Segmentation Analysis")
```

## Error Handling and Debugging

### Robust Error Handling

```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_clustering(data, max_retries=3):
    """Robust clustering with error handling and retries."""
    
    for attempt in range(max_retries):
        try:
            clusterer = EnhancedAdaptiveDBSCAN(
                eps=0.5,
                min_samples=5,
                adaptive=True
            )
            
            labels = clusterer.fit_predict(data)
            
            # Validate results
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters == 0:
                raise ValueError("No clusters found")
            
            logger.info(f"Clustering successful: {n_clusters} clusters found")
            return labels, clusterer
            
        except Exception as e:
            logger.warning(f"Clustering attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error("All clustering attempts failed")
                raise
            
    return None, None

# Use robust clustering
try:
    labels, clusterer = robust_clustering(X)
    print("Clustering completed successfully")
except Exception as e:
    print(f"Clustering failed: {e}")
```

### Performance Debugging

```python
import time
from contextlib import contextmanager

@contextmanager
def timer(description):
    """Context manager for timing operations."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"{description}: {elapsed:.3f} seconds")

# Profile different components
with timer("Data preprocessing"):
    # Preprocessing steps
    X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

with timer("Clustering execution"):
    clusterer = EnhancedAdaptiveDBSCAN()
    labels = clusterer.fit_predict(X_scaled)

with timer("Visualization"):
    clusterer.plot()

# Memory profiling
if hasattr(clusterer, 'get_memory_usage'):
    memory_stats = clusterer.get_memory_usage()
    print(f"Memory usage: {memory_stats}")
```

This comprehensive guide covers the main usage patterns for the Enhanced Adaptive DBSCAN framework. For more specific use cases or advanced features, refer to the API documentation.

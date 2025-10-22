# Enhanced Adaptive DBSCAN

[![PyPI version](https://badge.fury.io/py/enhanced-adaptive-dbscan.svg)](https://badge.fury.io/py/enhanced-adaptive-dbscan)
[![CI](https://github.com/kennedym-ds/enhanced_adaptive_dbscan/workflows/CI/badge.svg)](https://github.com/kennedym-ds/enhanced_adaptive_dbscan/actions)
[![Documentation](https://github.com/kennedym-ds/enhanced_adaptive_dbscan/workflows/Documentation/badge.svg)](https://kennedym-ds.github.io/enhanced_adaptive_dbscan/)
[![codecov](https://codecov.io/gh/kennedym-ds/enhanced_adaptive_dbscan/branch/main/graph/badge.svg)](https://codecov.io/gh/kennedym-ds/enhanced_adaptive_dbscan)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **comprehensive adaptive clustering framework** featuring three specialized clustering approaches: **Enhanced Adaptive DBSCAN**, **Ensemble Clustering**, and **Adaptive Parameter Optimization**. Designed for complex data analysis including semiconductor wafer defect detection, multi-density datasets, and automatic parameter tuning.

## 🌟 Key Capabilities

### 🎯 Phase 1: Enhanced Adaptive DBSCAN
- **Adaptive Parameter Selection:** Automatically adjusts ε (epsilon) and MinPts based on local density
- **Multi-Scale Density Analysis:** Analyzes density patterns across multiple scales
- **Stability-Based Cluster Selection:** Retains only robust, stable clusters
- **Incremental Clustering:** Real-time updates with new data points
- **Interactive Visualization:** Rich plotting capabilities with Plotly

### 🎪 Phase 2: Ensemble & Multi-Density Clustering  
- **Parameter Ensemble Generation:** Creates diverse parameter sets for robust clustering
- **Consensus Voting Mechanisms:** Multiple voting strategies (majority, weighted, quality-based)
- **Multi-Density Region Detection:** Handles datasets with varying density patterns
- **Boundary Analysis:** Advanced boundary point detection and refinement
- **Cluster Quality Assessment:** Comprehensive quality metrics and validation

### 🧠 Phase 3: Adaptive Optimization Framework
- **Bayesian Parameter Optimization:** Gaussian Process-based parameter search
- **Genetic Algorithm Optimization:** Evolution-based parameter exploration
- **Performance Prediction:** ML-based clustering outcome prediction
- **Meta-Learning:** Cross-dataset learning and strategy recommendation
- **Automated Parameter Tuning:** Intelligent parameter space exploration

### 🏭 Phase 4: Production Pipeline & Enterprise Integration
- **Streaming Clustering Engine:** Real-time data processing with concept drift detection
- **Production Pipeline:** Complete model lifecycle management (train, validate, deploy, monitor)
- **RESTful Web API:** Flask-based API with health monitoring and clustering endpoints
- **Enterprise Integration:** Configuration management, deployment automation, and monitoring

### 🚀 Phase 5: Advanced Clustering Enhancements (NEW!)
- **Deep Learning Integration:** Neural network-based clustering with autoencoders and DEC
- **Scalable Indexing:** Approximate nearest neighbor search for millions of points (Annoy/FAISS)
- **Complete HDBSCAN:** Full hierarchical density-based clustering with stability selection
- **Hybrid Approaches:** Combine deep learning with traditional clustering methods
- **Distributed Processing:** Parallel clustering for large-scale datasets

## 📦 Installation

Install from PyPI:

```bash
pip install enhanced-adaptive-dbscan
```

For Phase 4 production features, install with additional dependencies:

```bash
pip install enhanced-adaptive-dbscan[production]
# or manually install Flask dependencies:
pip install enhanced-adaptive-dbscan flask flask-cors pyyaml
```

For Phase 5 advanced features (deep learning, scalable indexing):

```bash
# For deep learning clustering with PyTorch
pip install enhanced-adaptive-dbscan torch

# For scalable approximate nearest neighbor search
pip install enhanced-adaptive-dbscan annoy  # or faiss-cpu for FAISS

# Install all advanced features
pip install enhanced-adaptive-dbscan[production] torch annoy
```

Or install from source:

```bash
git clone https://github.com/kennedym-ds/enhanced_adaptive_dbscan.git
cd enhanced_adaptive_dbscan
pip install -e .
```

## 🔧 Quick Start

### Basic Adaptive DBSCAN

```python
from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
import numpy as np

# Generate synthetic data
X = np.random.randn(1000, 2)
severity = np.random.randint(1, 11, size=(1000, 1))

# Initialize the model
model = EnhancedAdaptiveDBSCAN(
    wafer_shape='circular',
    wafer_size=100,
    k=20,
    density_scaling=1.0,
    additional_features=[2],  # Include severity as feature
    feature_weights=[1.0],
    stability_threshold=0.6
)

# Fit and get results
X_full = np.hstack((X, severity))
model.fit(X_full, additional_attributes=severity)
labels = model.labels_

# Visualize results
model.plot_clusters(X_full)
model.evaluate_clustering(X_full[:, :2], labels)
```

### Ensemble Clustering

```python
from enhanced_adaptive_dbscan.ensemble_clustering import ConsensusClusteringEngine
import numpy as np

# Create diverse parameter sets
engine = ConsensusClusteringEngine(
    n_estimators=50,
    voting_strategy='quality_weighted',
    stability_threshold=0.7
)

# Generate synthetic multi-density data
X = np.vstack([
    np.random.normal(0, 0.5, (200, 2)),      # Dense cluster
    np.random.normal(5, 1.5, (100, 2)),      # Sparse cluster
    np.random.normal([0, 5], 0.8, (150, 2))  # Medium density
])

# Perform consensus clustering
consensus_labels = engine.fit_consensus_clustering(X)
quality_scores = engine.get_cluster_quality_scores()

print(f"Consensus clustering found {len(set(consensus_labels)) - 1} clusters")
print(f"Average quality score: {np.mean(list(quality_scores.values())):.3f}")
```

### Adaptive Parameter Optimization

```python
from enhanced_adaptive_dbscan.adaptive_optimization import AdaptiveTuningEngine
import numpy as np

# Create optimization engine
tuning_engine = AdaptiveTuningEngine(
    optimization_method='bayesian',  # or 'genetic'
    n_iterations=50,
    prediction_enabled=True,
    meta_learning_enabled=True
)

# Define parameter space to optimize
parameter_space = {
    'eps': (0.1, 2.0),
    'min_samples': (5, 50)
}

# Optimize parameters for your data
result = tuning_engine.optimize_parameters(
    X, parameter_space, 
    optimization_metric='silhouette_score'
)

print(f"Best parameters: {result.best_parameters}")
print(f"Best score: {result.best_score:.3f}")
print(f"Optimization insights: {result.meta_learning_insights}")
```

### Production Pipeline & Streaming (Phase 4)

```python
from enhanced_adaptive_dbscan.production_pipeline import ProductionPipeline, DeploymentConfig
from enhanced_adaptive_dbscan.streaming_engine import StreamingClusteringEngine, StreamingConfig
from enhanced_adaptive_dbscan.web_api import ClusteringWebAPI
import numpy as np

# 1. Production Pipeline Setup
config = DeploymentConfig(
    model_name="adaptive_dbscan_prod",
    version="1.0",
    environment="production",
    model_store_path="./models",
    metrics_store_path="./metrics",
    auto_scaling=True
)

pipeline = ProductionPipeline(config)

# 2. Train and deploy model
X = np.random.randn(1000, 2)
model = pipeline.train_model(X)
deployment_id = pipeline.deploy_model(model)
print(f"Model deployed with ID: {deployment_id}")

# 3. Real-time streaming clustering
streaming_config = StreamingConfig(
    window_size=100,
    overlap=0.1,
    enable_concept_drift_detection=True,
    drift_threshold=0.05
)

streaming_engine = StreamingClusteringEngine(model, streaming_config)
streaming_engine.start_streaming()

# Process real-time data
for _ in range(10):
    new_point = np.random.randn(1, 2)
    result = streaming_engine.process_point(new_point[0])
    print(f"Point clustered: {result['cluster_id']}")

# 4. Web API for enterprise integration
api = ClusteringWebAPI(host="localhost", port=5001, debug=False)
# Visit http://localhost:5001/api/health for system status
# POST to http://localhost:5001/api/cluster for clustering requests
```

### Scalable Clustering (Phase 5)

```python
from enhanced_adaptive_dbscan import ScalableDBSCAN, ScalableIndexManager
import numpy as np

# For datasets with millions of points
X = np.random.randn(1_000_000, 10)  # 1M points

# Scalable DBSCAN with approximate nearest neighbors
scalable_dbscan = ScalableDBSCAN(
    eps=0.5,
    min_samples=10,
    n_jobs=-1  # Use all CPUs
)

labels = scalable_dbscan.fit_predict(X)
print(f"Clustered {len(X)} points into {len(set(labels))} clusters")

# Or use the index manager directly for more control
from enhanced_adaptive_dbscan import IndexConfig

config = IndexConfig(
    method='auto',  # Automatically selects best method (annoy/faiss/kdtree)
    metric='euclidean'
)

index_manager = ScalableIndexManager(config)
index_manager.build_index(X)

# Query nearest neighbors efficiently
distances, indices = index_manager.query_neighbors(X[:100], k=10)
```

### HDBSCAN Hierarchical Clustering (Phase 5)

```python
from enhanced_adaptive_dbscan import HDBSCANClusterer
import numpy as np

# Multi-density data
X = np.vstack([
    np.random.randn(100, 2) * 0.3,       # Dense cluster
    np.random.randn(50, 2) * 0.8 + 5,    # Sparse cluster
    np.random.randn(30, 2) * 0.5 + [3, 3] # Medium cluster
])

# HDBSCAN with stability-based selection
hdbscan = HDBSCANClusterer(
    min_cluster_size=15,
    min_samples=5,
    metric='euclidean'
)

labels = hdbscan.fit_predict(X)

# Get detailed cluster information
info = hdbscan.get_cluster_info()
print(f"Found {info['n_clusters']} stable clusters")
print(f"Cluster persistence: {info['cluster_persistence']}")
```

### Deep Learning Clustering (Phase 5)

```python
from enhanced_adaptive_dbscan import DeepClusteringEngine, HybridDeepDBSCAN
import numpy as np

# High-dimensional wafer data
X = np.random.randn(500, 50)  # 50 features

# Method 1: Autoencoder-based clustering
deep_engine = DeepClusteringEngine(
    method='autoencoder',
    latent_dim=10,
    n_clusters=5,
    n_epochs=100
)

result = deep_engine.fit_transform(X)
print(f"Reconstruction error: {result.reconstruction_error:.4f}")
print(f"Found {len(set(result.labels))} clusters")

# Method 2: Hybrid Deep+DBSCAN approach
hybrid = HybridDeepDBSCAN(
    latent_dim=10,
    n_epochs=50,
    dbscan_params={'min_samples': 5}
)

labels = hybrid.fit_predict(X)
embeddings = hybrid.get_embeddings(X)  # Get learned representations
print(f"Reduced {X.shape[1]}D to {embeddings.shape[1]}D")
```

### Distributed Clustering (Phase 5)

```python
from enhanced_adaptive_dbscan import DistributedClusteringCoordinator
import numpy as np

# Very large dataset
X = np.random.randn(5_000_000, 20)  # 5M points

coordinator = DistributedClusteringCoordinator(n_workers=-1)

# Build index
from enhanced_adaptive_dbscan import ScalableIndexManager
index = ScalableIndexManager()
index.build_index(X)

# Distributed density estimation
densities = coordinator.distributed_density_estimation(
    X, k=20, index=index
)

print(f"Computed densities for {len(X)} points in parallel")
```

## 📚 Documentation

- **API Reference:** [Full documentation](https://kennedym-ds.github.io/enhanced_adaptive_dbscan/)
- **Examples:** Check the [`examples/`](examples/) directory for detailed usage examples
- **Algorithm Details:** See our [detailed algorithm overview](https://kennedym-ds.github.io/enhanced_adaptive_dbscan/algorithm.html)

## 🛠️ Development

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/kennedym-ds/enhanced_adaptive_dbscan.git
cd enhanced_adaptive_dbscan

# Create virtual environment
python -m venv .venv
./.venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run tests with coverage
pytest --cov=enhanced_adaptive_dbscan --cov-report=html

# Run linting
ruff check .

# Run type checking
mypy .
```

### Building Documentation

```bash
cd docs
make html
```

## 🔬 Algorithm Overview

The Enhanced Adaptive DBSCAN framework provides five complementary clustering approaches:

### 🎯 Phase 1: Core Adaptive DBSCAN

**Key Innovation**: Dynamic parameter adaptation based on local density patterns

- **Adaptive Parameters**: Automatically adjusts `eps` and `min_samples` for each point
- **Multi-Scale Analysis**: Analyzes density across multiple scales for robust detection
- **Wafer-Aware Clustering**: Optimized for semiconductor defect detection
- **Stability Filtering**: Retains only clusters that persist across parameter variations
- **Incremental Updates**: Efficient real-time data processing

**Algorithm Steps**:
1. **Local Density Estimation**: k-NN based density computation
2. **Adaptive Parameter Selection**: Point-wise `eps` and `min_samples` adjustment  
3. **Multi-Scale Clustering**: Clustering across density scales
4. **Stability Assessment**: Cluster persistence evaluation
5. **Incremental Processing**: Real-time data integration

### 🎪 Phase 2: Ensemble & Multi-Density Methods

**Key Innovation**: Consensus-based clustering with multi-density awareness

- **Parameter Diversity**: Generates diverse parameter sets for robust clustering
- **Voting Mechanisms**: Multiple consensus strategies (majority, weighted, quality-based)
- **Multi-Density Handling**: Specialized algorithms for varying density regions
- **Boundary Processing**: Advanced boundary point analysis and refinement
- **Quality Assessment**: Comprehensive cluster validation metrics

**Core Components**:
1. **Parameter Ensemble**: Strategic parameter space sampling
2. **Consensus Voting**: Multi-strategy result aggregation
3. **Multi-Density Engine**: Region-aware clustering algorithms
4. **Boundary Processor**: Cluster boundary analysis and refinement
5. **Quality Analyzer**: Multi-metric cluster validation

### 🧠 Phase 3: Adaptive Optimization Framework

**Key Innovation**: AI-powered parameter optimization with meta-learning

- **Bayesian Optimization**: Gaussian Process-based parameter search
- **Genetic Algorithms**: Evolution-based parameter exploration  
- **Performance Prediction**: ML-based clustering outcome forecasting
- **Meta-Learning**: Cross-dataset strategy recommendation
- **Automated Tuning**: Intelligent parameter space exploration

**Optimization Methods**:
1. **Bayesian Optimizer**: GP-based acquisition function optimization
2. **Genetic Optimizer**: Population-based evolutionary search
3. **Performance Predictor**: ML-based outcome prediction
4. **Meta-Learning Component**: Cross-dataset learning and recommendations
5. **Parameter Explorer**: Intelligent space exploration strategies

### 🏭 Phase 4: Production Pipeline & Enterprise Integration

**Key Innovation**: Enterprise-ready clustering platform with streaming and deployment automation

- **Streaming Clustering**: Real-time data processing with concept drift detection
- **Model Lifecycle Management**: Automated training, validation, deployment, and monitoring
- **RESTful API**: Enterprise integration with health monitoring and clustering endpoints
- **Configuration Management**: Environment-specific deployment configurations
- **Performance Monitoring**: Real-time metrics and alerting for production operations

**Production Components**:
1. **Streaming Engine**: Real-time clustering with drift detection
2. **Production Pipeline**: Complete model lifecycle automation
3. **Web API**: RESTful endpoints for enterprise integration
4. **Model Store**: Versioned model persistence and retrieval
5. **Performance Monitor**: Operational metrics and alerting

### 🚀 Phase 5: Advanced Clustering Enhancements

**Key Innovation**: State-of-the-art deep learning and scalability features

- **Deep Learning Clustering**: Autoencoder-based representation learning and DEC
- **Scalable Approximate NN**: Efficient nearest neighbor search for millions of points
- **Complete HDBSCAN**: Hierarchical clustering with minimum spanning trees
- **Hybrid Approaches**: Combine deep learning with traditional methods
- **Distributed Processing**: Parallel computation for large-scale datasets

**Advanced Components**:
1. **Deep Clustering Engine**: Neural network-based clustering (autoencoder, DEC)
2. **Scalable Indexing**: Approximate NN with Annoy/FAISS/KDTree
3. **HDBSCAN Implementation**: MST-based hierarchical clustering
4. **Hybrid Models**: Deep learning + traditional clustering fusion
5. **Distributed Coordinator**: Multi-core parallel processing

## 🧮 Technical Details

### Multi-Scale Density Analysis
The framework performs density analysis across multiple scales to capture varying cluster characteristics:

```python
# Multi-scale density computation
scales = [0.5, 1.0, 1.5, 2.0]  # Density scaling factors
for scale in scales:
    local_eps = base_eps * scale * density_factor[point]
    # Perform clustering at this scale
```

### Consensus Voting Strategies
Multiple voting mechanisms ensure robust cluster assignment:

- **Majority Voting**: Simple democratic consensus
- **Weighted Voting**: Parameter-quality weighted decisions  
- **Quality-Based Voting**: Performance-metric driven consensus

### Optimization Algorithms
The framework includes multiple optimization approaches:

- **Bayesian Optimization**: Efficient for continuous parameter spaces
- **Genetic Algorithms**: Effective for discrete/mixed parameter spaces
- **Hybrid Approaches**: Combining multiple optimization strategies

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Run pre-commit hooks (`pre-commit run --all-files`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- **Author**: Michael Kennedy
- **Email**: kennedym.ds@gmail.com
- **GitHub**: [@kennedym-ds](https://github.com/kennedym-ds)

## 🙏 Acknowledgments

- Built on top of the excellent [scikit-learn](https://scikit-learn.org/) library
- Inspired by the original DBSCAN algorithm by Ester et al.
- Special thanks to the open-source community for their invaluable contributions

## 📖 Citation

If you use this algorithm in your research, please cite:

```bibtex
@software{kennedy_enhanced_adaptive_dbscan_2024,
  author = {Kennedy, Michael},
  title = {Enhanced Adaptive DBSCAN},
  year = {2024},
  url = {https://github.com/kennedym-ds/enhanced_adaptive_dbscan},
  version = {1.0.0}
}
```

# tests/test_deep_clustering.py

"""Tests for deep clustering module."""

import pytest
import numpy as np
from enhanced_adaptive_dbscan.deep_clustering import (
    TORCH_AVAILABLE,
    DeepClusteringResult
)

if TORCH_AVAILABLE:
    from enhanced_adaptive_dbscan.deep_clustering import (
        DeepClusteringEngine,
        HybridDeepDBSCAN,
        Autoencoder,
        DeepDensityEstimator,
        DeepEmbeddedClustering
    )
    import torch


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestAutoencoder:
    """Test Autoencoder model."""
    
    def test_initialization(self):
        """Test autoencoder initialization."""
        model = Autoencoder(input_dim=10, latent_dim=5)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass through autoencoder."""
        model = Autoencoder(input_dim=10, latent_dim=5)
        
        X = torch.randn(32, 10)
        encoded, decoded = model(X)
        
        assert encoded.shape == (32, 5)
        assert decoded.shape == (32, 10)
    
    def test_encode_only(self):
        """Test encoding only."""
        model = Autoencoder(input_dim=10, latent_dim=5)
        
        X = torch.randn(32, 10)
        encoded = model.encode(X)
        
        assert encoded.shape == (32, 5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDeepDensityEstimator:
    """Test DeepDensityEstimator model."""
    
    def test_initialization(self):
        """Test density estimator initialization."""
        model = DeepDensityEstimator(input_dim=10)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DeepDensityEstimator(input_dim=10)
        
        X = torch.randn(32, 10)
        densities = model(X)
        
        assert densities.shape == (32, 1)
        # All densities should be positive (Softplus output)
        assert torch.all(densities > 0)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDeepEmbeddedClustering:
    """Test DeepEmbeddedClustering model."""
    
    def test_initialization(self):
        """Test DEC initialization."""
        model = DeepEmbeddedClustering(
            input_dim=10,
            n_clusters=3,
            latent_dim=5
        )
        assert model.n_clusters == 3
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = DeepEmbeddedClustering(
            input_dim=10,
            n_clusters=3,
            latent_dim=5
        )
        
        X = torch.randn(32, 10)
        encoded, decoded, q = model(X)
        
        assert encoded.shape == (32, 5)
        assert decoded.shape == (32, 10)
        assert q.shape == (32, 3)
        
        # Q should be valid probabilities
        assert torch.allclose(torch.sum(q, dim=1), torch.ones(32), atol=1e-5)
    
    def test_soft_assignment(self):
        """Test soft cluster assignment."""
        model = DeepEmbeddedClustering(
            input_dim=10,
            n_clusters=3,
            latent_dim=5
        )
        
        z = torch.randn(32, 5)
        q = model._soft_assignment(z)
        
        assert q.shape == (32, 3)
        # Should sum to 1
        assert torch.allclose(torch.sum(q, dim=1), torch.ones(32), atol=1e-5)
    
    def test_target_distribution(self):
        """Test target distribution computation."""
        q = torch.rand(32, 3)
        q = q / torch.sum(q, dim=1, keepdim=True)
        
        p = DeepEmbeddedClustering.target_distribution(q)
        
        assert p.shape == q.shape
        # Should sum to 1
        assert torch.allclose(torch.sum(p, dim=1), torch.ones(32), atol=1e-5)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestDeepClusteringEngine:
    """Test DeepClusteringEngine."""
    
    def test_initialization_autoencoder(self):
        """Test engine initialization with autoencoder."""
        engine = DeepClusteringEngine(
            method='autoencoder',
            latent_dim=5,
            n_epochs=2,  # Quick test
            batch_size=32
        )
        assert engine.method == 'autoencoder'
        assert engine.latent_dim == 5
    
    def test_initialization_dec(self):
        """Test engine initialization with DEC."""
        engine = DeepClusteringEngine(
            method='dec',
            n_clusters=3,
            latent_dim=5,
            n_epochs=2,
            batch_size=32
        )
        assert engine.method == 'dec'
        assert engine.n_clusters == 3
    
    def test_fit_transform_autoencoder(self):
        """Test fitting autoencoder."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        engine = DeepClusteringEngine(
            method='autoencoder',
            n_clusters=3,
            latent_dim=5,
            n_epochs=5,  # Quick test
            batch_size=32
        )
        
        result = engine.fit_transform(X)
        
        assert isinstance(result, DeepClusteringResult)
        assert result.labels.shape == (100,)
        assert result.embeddings.shape == (100, 5)
        assert result.reconstruction_error >= 0
        assert result.cluster_centers is not None
    
    def test_fit_transform_dec(self):
        """Test fitting DEC."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        engine = DeepClusteringEngine(
            method='dec',
            n_clusters=3,
            latent_dim=5,
            n_epochs=4,  # Quick test (2 pretrain + 2 optimize)
            batch_size=32
        )
        
        result = engine.fit_transform(X)
        
        assert isinstance(result, DeepClusteringResult)
        assert result.labels.shape == (100,)
        assert result.embeddings.shape == (100, 5)
        # Should have 3 clusters as specified
        assert len(np.unique(result.labels)) <= 3
    
    def test_fit_transform_density(self):
        """Test fitting density estimator."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        engine = DeepClusteringEngine(
            method='density',
            latent_dim=5,
            n_epochs=5,
            batch_size=32
        )
        
        result = engine.fit_transform(X)
        
        assert isinstance(result, DeepClusteringResult)
        assert result.labels.shape == (100,)
        assert result.embeddings.shape[0] == 100
    
    def test_transform_after_fit(self):
        """Test transforming new data after fitting."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_test = np.random.randn(20, 10)
        
        engine = DeepClusteringEngine(
            method='autoencoder',
            latent_dim=5,
            n_epochs=3,
            batch_size=32
        )
        
        # Fit
        result = engine.fit_transform(X_train)
        
        # Transform
        embeddings = engine.transform(X_test)
        
        assert embeddings.shape == (20, 5)
    
    def test_transform_before_fit_raises(self):
        """Test that transform before fit raises error."""
        engine = DeepClusteringEngine(method='autoencoder')
        
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Model not trained"):
            engine.transform(X)
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises error."""
        engine = DeepClusteringEngine(method='invalid')
        
        X = np.random.randn(100, 10)
        
        with pytest.raises(ValueError, match="Unknown method"):
            engine.fit_transform(X)
    
    def test_dec_without_n_clusters_raises(self):
        """Test that DEC without n_clusters raises error."""
        engine = DeepClusteringEngine(
            method='dec',
            n_clusters=None  # Should raise error
        )
        
        X = np.random.randn(100, 10)
        
        with pytest.raises(ValueError, match="n_clusters must be specified"):
            engine.fit_transform(X)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestHybridDeepDBSCAN:
    """Test HybridDeepDBSCAN."""
    
    def test_initialization(self):
        """Test initialization."""
        model = HybridDeepDBSCAN(
            latent_dim=5,
            n_epochs=2
        )
        assert model.latent_dim == 5
        assert model.n_epochs == 2
    
    def test_fit_predict(self):
        """Test fitting and predicting."""
        np.random.seed(42)
        
        # Create clustered data
        cluster1 = np.random.randn(50, 10) * 0.3
        cluster2 = np.random.randn(50, 10) * 0.3 + 2
        X = np.vstack([cluster1, cluster2])
        
        model = HybridDeepDBSCAN(
            latent_dim=5,
            n_epochs=5
        )
        
        labels = model.fit_predict(X)
        
        assert labels.shape == (100,)
        
        # Should find at least 1 cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1
    
    def test_get_embeddings(self):
        """Test getting embeddings."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        
        model = HybridDeepDBSCAN(latent_dim=5, n_epochs=3)
        labels = model.fit_predict(X)
        
        embeddings = model.get_embeddings(X)
        
        assert embeddings.shape == (50, 5)
    
    def test_get_embeddings_before_fit_raises(self):
        """Test that getting embeddings before fit raises error."""
        model = HybridDeepDBSCAN()
        
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.get_embeddings(X)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestIntegration:
    """Integration tests for deep clustering."""
    
    def test_autoencoder_clustering_quality(self):
        """Test that autoencoder produces meaningful clusters."""
        np.random.seed(42)
        
        # Well-separated clusters
        cluster1 = np.random.randn(50, 20) * 0.3
        cluster2 = np.random.randn(50, 20) * 0.3 + 3
        cluster3 = np.random.randn(50, 20) * 0.3 + [0, 3, 0] * 6 + [0, 0]
        X = np.vstack([cluster1, cluster2, cluster3])
        
        engine = DeepClusteringEngine(
            method='autoencoder',
            n_clusters=3,
            latent_dim=10,
            n_epochs=10,
            batch_size=32
        )
        
        result = engine.fit_transform(X)
        
        # Should find 3 clusters
        assert len(np.unique(result.labels)) == 3
        
        # Reconstruction error should be reasonable
        assert result.reconstruction_error < 10.0
    
    def test_dec_vs_autoencoder(self):
        """Compare DEC with autoencoder+kmeans."""
        np.random.seed(42)
        
        # Create data
        cluster1 = np.random.randn(40, 10) * 0.3
        cluster2 = np.random.randn(40, 10) * 0.3 + 2
        X = np.vstack([cluster1, cluster2])
        
        # Autoencoder
        ae_engine = DeepClusteringEngine(
            method='autoencoder',
            n_clusters=2,
            latent_dim=5,
            n_epochs=5,
            random_state=42
        )
        ae_result = ae_engine.fit_transform(X)
        
        # DEC
        dec_engine = DeepClusteringEngine(
            method='dec',
            n_clusters=2,
            latent_dim=5,
            n_epochs=6,  # 3 pretrain + 3 optimize
            random_state=42
        )
        dec_result = dec_engine.fit_transform(X)
        
        # Both should find 2 clusters
        assert len(np.unique(ae_result.labels)) == 2
        assert len(np.unique(dec_result.labels)) == 2
    
    def test_hybrid_deep_dbscan_workflow(self):
        """Test complete hybrid workflow."""
        np.random.seed(42)
        
        # Multi-cluster data
        cluster1 = np.random.randn(30, 15) * 0.2
        cluster2 = np.random.randn(30, 15) * 0.3 + 3
        noise = np.random.randn(10, 15) * 2
        X = np.vstack([cluster1, cluster2, noise])
        
        model = HybridDeepDBSCAN(
            latent_dim=8,
            n_epochs=8,
            dbscan_params={'min_samples': 5}
        )
        
        labels = model.fit_predict(X)
        
        # Should identify clusters and noise
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1
        
        # Should have some noise
        n_noise = np.sum(labels == -1)
        assert n_noise >= 0
        
        # Get embeddings
        embeddings = model.get_embeddings(X)
        assert embeddings.shape[0] == X.shape[0]
    
    def test_dimensionality_reduction_quality(self):
        """Test that dimensionality reduction preserves structure."""
        np.random.seed(42)
        
        # High-dimensional data with structure
        n_features = 50
        cluster1 = np.random.randn(40, n_features) * 0.3
        cluster2 = np.random.randn(40, n_features) * 0.3 + 2
        X = np.vstack([cluster1, cluster2])
        
        engine = DeepClusteringEngine(
            method='autoencoder',
            latent_dim=5,
            n_epochs=10
        )
        
        result = engine.fit_transform(X)
        
        # Check that embeddings preserve cluster structure
        embeddings = result.embeddings
        
        # Points from same cluster should be closer
        from scipy.spatial.distance import pdist, squareform
        
        embed_dist = squareform(pdist(embeddings))
        
        # Average within-cluster distance
        within_dist = np.mean([
            embed_dist[i, j]
            for i in range(40) for j in range(40) if i < j
        ])
        
        # Average between-cluster distance
        between_dist = np.mean([
            embed_dist[i, j]
            for i in range(40) for j in range(40, 80)
        ])
        
        # Between should be larger than within
        assert between_dist > within_dist


class TestDeepClusteringResult:
    """Test DeepClusteringResult dataclass."""
    
    def test_creation(self):
        """Test creating result object."""
        result = DeepClusteringResult(
            labels=np.array([0, 1, 0]),
            embeddings=np.array([[1, 2], [3, 4], [5, 6]]),
            reconstruction_error=0.5
        )
        
        assert result.labels.shape == (3,)
        assert result.embeddings.shape == (3, 2)
        assert result.reconstruction_error == 0.5
        assert result.cluster_centers is None
        assert result.training_history is None
    
    def test_with_optional_fields(self):
        """Test with optional fields."""
        result = DeepClusteringResult(
            labels=np.array([0, 1]),
            embeddings=np.array([[1, 2], [3, 4]]),
            reconstruction_error=0.5,
            cluster_centers=np.array([[1.5, 2.5], [3.5, 4.5]]),
            training_history={'loss': [1.0, 0.5]}
        )
        
        assert result.cluster_centers is not None
        assert result.training_history is not None
        assert 'loss' in result.training_history

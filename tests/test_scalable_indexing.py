# tests/test_scalable_indexing.py

"""Tests for scalable indexing module."""

import pytest
import numpy as np
from enhanced_adaptive_dbscan.scalable_indexing import (
    ScalableIndexManager,
    ScalableDBSCAN,
    IndexConfig,
    ChunkedProcessor,
    DistributedClusteringCoordinator,
    KDTreeIndex,
    ANNOY_AVAILABLE,
    FAISS_AVAILABLE
)


class TestIndexConfig:
    """Test IndexConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = IndexConfig()
        assert config.method == 'auto'
        assert config.metric == 'euclidean'
        assert config.n_trees == 10
        assert config.chunk_size == 100000
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = IndexConfig(
            method='annoy',
            metric='cosine',
            n_trees=20,
            chunk_size=50000
        )
        assert config.method == 'annoy'
        assert config.metric == 'cosine'
        assert config.n_trees == 20
        assert config.chunk_size == 50000


class TestKDTreeIndex:
    """Test KDTree index (always available)."""
    
    def test_build_and_query(self):
        """Test building index and querying neighbors."""
        config = IndexConfig(metric='euclidean')
        index = KDTreeIndex(config)
        
        # Create small test dataset
        X = np.random.randn(100, 5)
        
        # Build index
        index.build(X)
        
        # Query neighbors
        query_points = X[:10]
        distances, indices = index.query(query_points, k=5)
        
        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
        
        # First neighbor should be the point itself
        assert np.allclose(distances[:, 0], 0.0, atol=1e-6)
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        config = IndexConfig()
        index = KDTreeIndex(config)
        
        X = np.random.randn(50, 3)
        index.build(X)
        
        # Save
        filepath = str(tmp_path / "kdtree.pkl")
        index.save(filepath)
        
        # Load
        index2 = KDTreeIndex(config)
        index2.load(filepath)
        
        # Query both indexes
        query = X[:5]
        dist1, idx1 = index.query(query, k=3)
        dist2, idx2 = index2.query(query, k=3)
        
        assert np.allclose(dist1, dist2)
        assert np.array_equal(idx1, idx2)


class TestScalableIndexManager:
    """Test ScalableIndexManager."""
    
    def test_auto_selection_small_dataset(self):
        """Test automatic method selection for small dataset."""
        manager = ScalableIndexManager()
        
        # Small dataset should use KDTree
        X = np.random.randn(100, 5)
        manager.build_index(X)
        
        assert manager.index is not None
        assert isinstance(manager.index, KDTreeIndex)
    
    def test_build_and_query(self):
        """Test building and querying index."""
        manager = ScalableIndexManager()
        
        X = np.random.randn(200, 4)
        manager.build_index(X)
        
        # Query neighbors
        query = X[:10]
        distances, indices = manager.query_neighbors(query, k=5)
        
        assert distances.shape == (10, 5)
        assert indices.shape == (10, 5)
    
    def test_query_before_build_raises(self):
        """Test that querying before building raises error."""
        manager = ScalableIndexManager()
        
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="Index not built"):
            manager.query_neighbors(X, k=3)
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading index."""
        manager = ScalableIndexManager()
        
        X = np.random.randn(100, 3)
        manager.build_index(X)
        
        # Save
        filepath = str(tmp_path / "index")
        manager.save_index(filepath)
        
        # Load
        manager2 = ScalableIndexManager()
        manager2.load_index(filepath)
        
        # Query both
        query = X[:5]
        dist1, idx1 = manager.query_neighbors(query, k=3)
        dist2, idx2 = manager2.query_neighbors(query, k=3)
        
        assert np.allclose(dist1, dist2)


class TestChunkedProcessor:
    """Test ChunkedProcessor."""
    
    def test_process_chunked(self):
        """Test chunked processing."""
        processor = ChunkedProcessor(chunk_size=50)
        
        X = np.random.randn(200, 3)
        
        # Process: compute sum of each chunk
        def process_fn(chunk, state):
            return np.sum(chunk)
        
        def combine_fn(state, result):
            if state is None:
                return result
            return state + result
        
        total = processor.process_chunked(
            X,
            process_fn,
            combine_fn,
            initial_state=None
        )
        
        # Should equal total sum
        expected_total = np.sum(X)
        assert np.allclose(total, expected_total)
    
    def test_build_index_chunked(self):
        """Test building index with chunked processor."""
        processor = ChunkedProcessor(chunk_size=100)
        
        X = np.random.randn(150, 4)
        
        # Build index
        manager = processor.build_index_chunked(X)
        
        assert manager is not None
        assert manager.index is not None
        
        # Query should work
        distances, indices = manager.query_neighbors(X[:5], k=3)
        assert distances.shape == (5, 3)


class TestDistributedClusteringCoordinator:
    """Test DistributedClusteringCoordinator."""
    
    def test_initialization(self):
        """Test coordinator initialization."""
        coordinator = DistributedClusteringCoordinator(n_workers=2)
        assert coordinator.n_workers == 2
    
    def test_distributed_density_estimation(self):
        """Test distributed density estimation."""
        coordinator = DistributedClusteringCoordinator(n_workers=2)
        
        X = np.random.randn(200, 3)
        
        # Build index
        manager = ScalableIndexManager()
        manager.build_index(X)
        
        # Compute density
        densities = coordinator.distributed_density_estimation(X, k=5, index=manager)
        
        assert densities.shape == (200,)
        assert np.all(densities > 0)


class TestScalableDBSCAN:
    """Test ScalableDBSCAN."""
    
    def test_fit_predict(self):
        """Test fitting and predicting with ScalableDBSCAN."""
        # Create synthetic clustered data
        np.random.seed(42)
        cluster1 = np.random.randn(100, 2) * 0.3
        cluster2 = np.random.randn(100, 2) * 0.3 + 3
        X = np.vstack([cluster1, cluster2])
        
        # Cluster
        dbscan = ScalableDBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        assert labels.shape == (200,)
        
        # Should find at least 2 clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 2
    
    def test_single_cluster(self):
        """Test with single cluster."""
        np.random.seed(42)
        X = np.random.randn(50, 2) * 0.3
        
        dbscan = ScalableDBSCAN(eps=1.0, min_samples=5)
        labels = dbscan.fit_predict(X)
        
        # Most points should be in one cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters <= 2
    
    def test_all_noise(self):
        """Test with scattered points (all noise)."""
        np.random.seed(42)
        X = np.random.randn(50, 2) * 10  # Very scattered
        
        dbscan = ScalableDBSCAN(eps=0.1, min_samples=10)
        labels = dbscan.fit_predict(X)
        
        # Should have significant noise
        n_noise = np.sum(labels == -1)
        assert n_noise > 0


class TestIntegration:
    """Integration tests for scalable indexing."""
    
    def test_full_workflow(self):
        """Test complete workflow: index -> density -> clustering."""
        np.random.seed(42)
        
        # Create multi-cluster data
        cluster1 = np.random.randn(100, 3) * 0.3
        cluster2 = np.random.randn(100, 3) * 0.3 + 3
        cluster3 = np.random.randn(100, 3) * 0.3 + [0, 3, 0]
        X = np.vstack([cluster1, cluster2, cluster3])
        
        # Build scalable index
        manager = ScalableIndexManager()
        manager.build_index(X)
        
        # Compute densities
        coordinator = DistributedClusteringCoordinator(n_workers=2)
        densities = coordinator.distributed_density_estimation(
            X, k=10, index=manager
        )
        
        assert densities.shape == (300,)
        
        # Perform scalable clustering
        dbscan = ScalableDBSCAN(eps=0.5, min_samples=10)
        labels = dbscan.fit_predict(X)
        
        # Should find 3 clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert 2 <= n_clusters <= 4  # Allow some tolerance
    
    @pytest.mark.skipif(not ANNOY_AVAILABLE, reason="Annoy not available")
    def test_with_annoy(self):
        """Test with Annoy index if available."""
        config = IndexConfig(method='annoy', n_trees=5)
        manager = ScalableIndexManager(config)
        
        X = np.random.randn(1000, 5)
        manager.build_index(X)
        
        distances, indices = manager.query_neighbors(X[:10], k=5)
        assert distances.shape == (10, 5)
    
    @pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not available")
    def test_with_faiss(self):
        """Test with FAISS index if available."""
        config = IndexConfig(method='faiss')
        manager = ScalableIndexManager(config)
        
        X = np.random.randn(1000, 5).astype(np.float32)
        manager.build_index(X)
        
        distances, indices = manager.query_neighbors(X[:10], k=5)
        assert distances.shape == (10, 5)

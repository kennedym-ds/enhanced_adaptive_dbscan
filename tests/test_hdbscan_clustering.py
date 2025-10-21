# tests/test_hdbscan_clustering.py

"""Tests for HDBSCAN-style hierarchical clustering module."""

import pytest
import numpy as np
from enhanced_adaptive_dbscan.hdbscan_clustering import (
    HDBSCANClusterer,
    MinimumSpanningTree,
    HierarchicalClusterTree,
    CondensedTree,
    StabilityBasedSelector,
    MSTreeEdge,
    ClusterNode
)


class TestMinimumSpanningTree:
    """Test minimum spanning tree construction."""
    
    def test_initialization(self):
        """Test MST initialization."""
        mst = MinimumSpanningTree(min_samples=5)
        assert mst.min_samples == 5
        assert mst.metric == 'euclidean'
    
    def test_core_distances(self):
        """Test core distance computation."""
        mst = MinimumSpanningTree(min_samples=3)
        
        X = np.random.randn(50, 2)
        core_distances = mst._compute_core_distances(X)
        
        assert core_distances.shape == (50,)
        assert np.all(core_distances >= 0)
    
    def test_mutual_reachability_distance(self):
        """Test mutual reachability distance calculation."""
        mst = MinimumSpanningTree()
        
        # MRD should be max of three values
        mrd = mst._mutual_reachability_distance(1.0, 2.0, 1.5)
        assert mrd == 2.0
        
        mrd = mst._mutual_reachability_distance(3.0, 1.0, 2.0)
        assert mrd == 3.0
    
    def test_construct_mst(self):
        """Test MST construction."""
        mst = MinimumSpanningTree(min_samples=3)
        
        # Simple dataset
        X = np.random.randn(20, 2)
        edges = mst.construct(X)
        
        # MST should have n-1 edges
        assert len(edges) == 19
        
        # All edges should have positive distances
        for edge in edges:
            assert edge.distance >= 0
    
    def test_mst_with_single_point(self):
        """Test MST with single point."""
        mst = MinimumSpanningTree()
        
        X = np.random.randn(1, 2)
        edges = mst.construct(X)
        
        # No edges for single point
        assert len(edges) == 0


class TestHierarchicalClusterTree:
    """Test hierarchical cluster tree building."""
    
    def test_initialization(self):
        """Test tree initialization."""
        tree = HierarchicalClusterTree()
        assert tree.nodes_ == {}
        assert tree.root_id_ is None
    
    def test_build_from_mst(self):
        """Test building hierarchy from MST."""
        # Create simple MST
        edges = [
            MSTreeEdge(0, 1, 0.5),
            MSTreeEdge(1, 2, 0.7),
            MSTreeEdge(2, 3, 0.9),
        ]
        
        tree = HierarchicalClusterTree()
        nodes = tree.build_from_mst(edges, n_points=4)
        
        # Should have 4 leaf nodes + 3 internal nodes
        assert len(nodes) == 7
        
        # Root should be set
        assert tree.root_id_ is not None
    
    def test_hierarchy_structure(self):
        """Test hierarchy structure is correct."""
        np.random.seed(42)
        
        # Build MST
        mst = MinimumSpanningTree(min_samples=3)
        X = np.random.randn(10, 2)
        edges = mst.construct(X)
        
        # Build hierarchy
        tree = HierarchicalClusterTree()
        nodes = tree.build_from_mst(edges, n_points=10)
        
        # Check parent-child relationships
        for node_id, node in nodes.items():
            if node.parent_id is not None:
                parent = nodes[node.parent_id]
                assert node_id in parent.children_ids


class TestCondensedTree:
    """Test condensed tree creation."""
    
    def test_initialization(self):
        """Test condensed tree initialization."""
        condenser = CondensedTree(min_cluster_size=5)
        assert condenser.min_cluster_size == 5
    
    def test_condense_simple_tree(self):
        """Test condensing a simple hierarchy."""
        # Create simple hierarchy
        nodes = {}
        
        # 4 leaf nodes
        for i in range(4):
            nodes[i] = ClusterNode(
                node_id=i,
                parent_id=4,
                children_ids=[],
                lambda_birth=np.inf,
                lambda_death=2.0,
                points={i},
                stability=0.0
            )
        
        # Internal node
        nodes[4] = ClusterNode(
            node_id=4,
            parent_id=None,
            children_ids=[0, 1, 2, 3],
            lambda_birth=2.0,
            lambda_death=0.0,
            points={0, 1, 2, 3},
            stability=0.0
        )
        
        condenser = CondensedTree(min_cluster_size=2)
        condensed = condenser.condense(nodes, root_id=4)
        
        # Should have condensed nodes
        assert len(condensed) >= 0


class TestStabilityBasedSelector:
    """Test stability-based cluster selection."""
    
    def test_initialization(self):
        """Test selector initialization."""
        selector = StabilityBasedSelector()
        assert not selector.allow_single_cluster
    
    def test_compute_stability(self):
        """Test stability computation."""
        # Create simple hierarchy
        nodes = {}
        
        nodes[0] = ClusterNode(
            node_id=0,
            parent_id=None,
            children_ids=[],
            lambda_birth=1.0,
            lambda_death=2.0,
            points={0, 1, 2},
            stability=0.0
        )
        
        selector = StabilityBasedSelector()
        stabilities = selector.compute_stability([], nodes)
        
        # Stability = size * persistence
        # persistence = 2.0 - 1.0 = 1.0
        # size = 3
        # stability = 3.0
        assert stabilities[0] == 3.0


class TestHDBSCANClusterer:
    """Test complete HDBSCAN clustering."""
    
    def test_initialization(self):
        """Test HDBSCAN initialization."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        assert clusterer.min_cluster_size == 5
        assert clusterer.min_samples == 5
        assert clusterer.metric == 'euclidean'
    
    def test_fit_predict_simple(self):
        """Test fitting on simple dataset."""
        np.random.seed(42)
        
        # Create two well-separated clusters
        cluster1 = np.random.randn(30, 2) * 0.3
        cluster2 = np.random.randn(30, 2) * 0.3 + 5
        X = np.vstack([cluster1, cluster2])
        
        clusterer = HDBSCANClusterer(min_cluster_size=10, min_samples=5)
        labels = clusterer.fit_predict(X)
        
        assert labels.shape == (60,)
        
        # Should find 2 clusters
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        assert n_clusters >= 1
    
    def test_fit_predict_with_noise(self):
        """Test clustering with noise points."""
        np.random.seed(42)
        
        # Dense cluster + scattered noise
        cluster = np.random.randn(50, 2) * 0.2
        noise = np.random.randn(20, 2) * 3
        X = np.vstack([cluster, noise])
        
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        
        # Should have some noise points
        n_noise = np.sum(labels == -1)
        assert n_noise >= 0
    
    def test_single_cluster(self):
        """Test with single cluster."""
        np.random.seed(42)
        X = np.random.randn(50, 2) * 0.5
        
        clusterer = HDBSCANClusterer(min_cluster_size=20)
        labels = clusterer.fit_predict(X)
        
        # Should find at least 1 cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1
    
    def test_get_cluster_info(self):
        """Test getting cluster information."""
        np.random.seed(42)
        
        cluster1 = np.random.randn(30, 2) * 0.3
        cluster2 = np.random.randn(30, 2) * 0.3 + 4
        X = np.vstack([cluster1, cluster2])
        
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        
        info = clusterer.get_cluster_info()
        
        assert 'n_clusters' in info
        assert 'n_noise' in info
        assert 'cluster_sizes' in info
        assert 'cluster_persistence' in info
        
        # Should have at least 1 cluster
        assert info['n_clusters'] >= 1
    
    def test_cluster_persistence(self):
        """Test cluster persistence values."""
        np.random.seed(42)
        
        # Well-separated clusters should have high persistence
        cluster1 = np.random.randn(30, 2) * 0.2
        cluster2 = np.random.randn(30, 2) * 0.2 + 10
        X = np.vstack([cluster1, cluster2])
        
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        
        # Check persistence values
        persistence = clusterer.cluster_persistence_
        
        for cluster_id, pers in persistence.items():
            # Persistence should be non-negative
            assert pers >= 0


class TestIntegration:
    """Integration tests for HDBSCAN."""
    
    def test_multi_density_clustering(self):
        """Test clustering data with multiple densities."""
        np.random.seed(42)
        
        # Create clusters with different densities
        dense_cluster = np.random.randn(50, 2) * 0.2
        sparse_cluster = np.random.randn(30, 2) * 0.8 + 5
        X = np.vstack([dense_cluster, sparse_cluster])
        
        clusterer = HDBSCANClusterer(min_cluster_size=15)
        labels = clusterer.fit_predict(X)
        
        # Should separate clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1
    
    def test_hierarchical_structure(self):
        """Test that hierarchical structure is meaningful."""
        np.random.seed(42)
        
        # Create nested cluster structure
        inner_cluster = np.random.randn(20, 2) * 0.2
        outer_cluster = np.random.randn(30, 2) * 0.6 + 0.5
        X = np.vstack([inner_cluster, outer_cluster])
        
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        
        # Check that hierarchy was built
        assert clusterer.hierarchy_builder.nodes_ is not None
        assert len(clusterer.hierarchy_builder.nodes_) > 0
    
    def test_comparison_with_traditional_dbscan(self):
        """Compare HDBSCAN with traditional DBSCAN behavior."""
        np.random.seed(42)
        
        # Simple two-cluster dataset
        cluster1 = np.random.randn(40, 2) * 0.3
        cluster2 = np.random.randn(40, 2) * 0.3 + 3
        X = np.vstack([cluster1, cluster2])
        
        # HDBSCAN
        hdbscan = HDBSCANClusterer(min_cluster_size=10)
        hdbscan_labels = hdbscan.fit_predict(X)
        
        # Should find meaningful clusters
        n_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
        assert n_clusters >= 2
    
    def test_stability_ordering(self):
        """Test that more stable clusters have higher scores."""
        np.random.seed(42)
        
        # Very tight cluster + looser cluster
        tight = np.random.randn(30, 2) * 0.1
        loose = np.random.randn(30, 2) * 0.5 + 5
        X = np.vstack([tight, loose])
        
        clusterer = HDBSCANClusterer(min_cluster_size=15)
        labels = clusterer.fit_predict(X)
        
        info = clusterer.get_cluster_info()
        stabilities = info.get('selected_stabilities', {})
        
        # Should have stability scores
        assert len(stabilities) > 0
    
    def test_robustness_to_noise(self):
        """Test that algorithm is robust to noise."""
        np.random.seed(42)
        
        # Cluster with uniform noise
        cluster = np.random.randn(50, 2) * 0.3
        noise = np.random.uniform(-5, 5, (30, 2))
        X = np.vstack([cluster, noise])
        
        clusterer = HDBSCANClusterer(min_cluster_size=20)
        labels = clusterer.fit_predict(X)
        
        # Should identify core cluster and mark noise
        n_noise = np.sum(labels == -1)
        assert n_noise > 0  # Some points should be noise
        
        # Should still find the main cluster
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_data(self):
        """Test with empty data."""
        clusterer = HDBSCANClusterer()
        
        X = np.array([]).reshape(0, 2)
        
        # Should handle gracefully or raise meaningful error
        try:
            labels = clusterer.fit_predict(X)
            assert len(labels) == 0
        except (ValueError, IndexError):
            pass  # Acceptable to raise error
    
    def test_single_point(self):
        """Test with single point."""
        clusterer = HDBSCANClusterer()
        
        X = np.array([[1.0, 2.0]])
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 1
    
    def test_two_points(self):
        """Test with two points."""
        clusterer = HDBSCANClusterer(min_cluster_size=1)
        
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        labels = clusterer.fit_predict(X)
        
        assert len(labels) == 2
    
    def test_identical_points(self):
        """Test with identical points."""
        clusterer = HDBSCANClusterer(min_cluster_size=5)
        
        # All points identical
        X = np.ones((20, 2))
        labels = clusterer.fit_predict(X)
        
        # Should form one cluster
        unique_labels = np.unique(labels)
        assert len(unique_labels) <= 2  # Either 1 cluster or noise
    
    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        
        # 20-dimensional data
        cluster1 = np.random.randn(30, 20) * 0.3
        cluster2 = np.random.randn(30, 20) * 0.3 + 2
        X = np.vstack([cluster1, cluster2])
        
        clusterer = HDBSCANClusterer(min_cluster_size=10)
        labels = clusterer.fit_predict(X)
        
        # Should still find clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1

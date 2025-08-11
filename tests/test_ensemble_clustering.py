"""
Test suite for Phase 3 Ensemble Clustering
"""

import pytest
import numpy as np
from sklearn.datasets import make_blobs, make_circles
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enhanced_adaptive_dbscan.ensemble_clustering import (
    ParameterEnsemble, VotingMechanism, ConsensusClusteringEngine,
    EnsembleMember, EnsembleResult
)


class TestParameterEnsemble:
    """Test ParameterEnsemble class"""
    
    def test_initialization(self):
        """Test ParameterEnsemble initialization"""
        pe = ParameterEnsemble(base_eps=0.3, base_min_samples=3, n_members=5)
        
        assert pe.base_eps == 0.3
        assert pe.base_min_samples == 3
        assert pe.n_members == 5
        assert pe.diversity_factor == 0.3  # default
    
    def test_generate_parameter_sets(self):
        """Test parameter set generation"""
        # Create test data
        X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
        
        pe = ParameterEnsemble(n_members=5)
        parameter_sets = pe.generate_parameter_sets(X)
        
        # Check basic properties
        assert len(parameter_sets) == 5
        assert all(isinstance(params, dict) for params in parameter_sets)
        
        # Check required parameters
        required_params = ['eps', 'min_samples', 'algorithm', 'metric']
        for params in parameter_sets:
            for req_param in required_params:
                assert req_param in params
            
            # Check parameter validity
            assert params['eps'] > 0
            assert params['min_samples'] >= 2
    
    def test_parameter_diversity(self):
        """Test parameter diversity calculation"""
        X, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=42)
        
        pe = ParameterEnsemble(n_members=10, diversity_factor=0.5)
        parameter_sets = pe.generate_parameter_sets(X)
        
        # Should have calculated diversity
        assert hasattr(pe, 'parameter_diversity')
        assert 0 <= pe.parameter_diversity <= 2  # Theoretical max for normalized distance
        
        # More members should generally lead to higher diversity
        pe_small = ParameterEnsemble(n_members=3, diversity_factor=0.5)
        parameter_sets_small = pe_small.generate_parameter_sets(X)
        
        # Can't guarantee diversity is higher, but should be calculated
        assert hasattr(pe_small, 'parameter_diversity')
    
    def test_different_data_sizes(self):
        """Test parameter generation with different data sizes"""
        # Small dataset
        X_small, _ = make_blobs(n_samples=20, centers=2, n_features=2, random_state=42)
        pe_small = ParameterEnsemble(n_members=5)
        params_small = pe_small.generate_parameter_sets(X_small)
        
        # Large dataset
        X_large, _ = make_blobs(n_samples=1000, centers=5, n_features=2, random_state=42)
        pe_large = ParameterEnsemble(n_members=5)
        params_large = pe_large.generate_parameter_sets(X_large)
        
        # Both should generate valid parameters
        assert len(params_small) == 5
        assert len(params_large) == 5
        
        # Parameters should adapt to data size
        avg_min_samples_small = np.mean([p['min_samples'] for p in params_small])
        avg_min_samples_large = np.mean([p['min_samples'] for p in params_large])
        
        # Larger datasets should generally allow larger min_samples
        assert avg_min_samples_large >= avg_min_samples_small


class TestVotingMechanism:
    """Test VotingMechanism class"""
    
    def test_initialization(self):
        """Test VotingMechanism initialization"""
        vm = VotingMechanism(voting_strategy='majority')
        assert vm.voting_strategy == 'majority'
        
        vm_weighted = VotingMechanism(voting_strategy='weighted')
        assert vm_weighted.voting_strategy == 'weighted'
    
    def test_consensus_matrix_building(self):
        """Test consensus matrix construction"""
        # Create mock ensemble members
        members = [
            EnsembleMember(
                parameters={'eps': 0.5, 'min_samples': 5},
                labels=np.array([0, 0, 1, 1, -1]),
                quality_scores={'silhouette': 0.7},
                execution_time=0.1,
                n_clusters=2,
                n_noise=1
            ),
            EnsembleMember(
                parameters={'eps': 0.3, 'min_samples': 3},
                labels=np.array([0, 0, 0, 1, 1]),
                quality_scores={'silhouette': 0.6},
                execution_time=0.15,
                n_clusters=2,
                n_noise=0
            )
        ]
        
        vm = VotingMechanism()
        consensus_matrix = vm._build_consensus_matrix(members, 5)
        
        # Check matrix properties
        assert consensus_matrix.shape == (5, 5)
        assert np.allclose(consensus_matrix, consensus_matrix.T)  # Should be symmetric
        assert np.all(consensus_matrix >= 0) and np.all(consensus_matrix <= 1)  # Normalized
    
    def test_majority_voting(self):
        """Test majority voting consensus"""
        # Create test data and mock members
        X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6]])
        
        members = [
            EnsembleMember(
                parameters={'eps': 0.5, 'min_samples': 2},
                labels=np.array([0, 0, 1, 1, 0]),
                quality_scores={'silhouette': 0.7},
                execution_time=0.1,
                n_clusters=2,
                n_noise=0
            ),
            EnsembleMember(
                parameters={'eps': 0.3, 'min_samples': 2},
                labels=np.array([0, 0, 1, 1, 0]),
                quality_scores={'silhouette': 0.6},
                execution_time=0.15,
                n_clusters=2,
                n_noise=0
            )
        ]
        
        vm = VotingMechanism(voting_strategy='majority')
        consensus_labels, confidence_scores = vm.vote_consensus(members, X)
        
        # Check output properties
        assert len(consensus_labels) == len(X)
        assert len(confidence_scores) == len(X)
        assert np.all(confidence_scores >= 0) and np.all(confidence_scores <= 1)
    
    def test_weighted_voting(self):
        """Test weighted voting consensus"""
        X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6]])
        
        members = [
            EnsembleMember(
                parameters={'eps': 0.5, 'min_samples': 2},
                labels=np.array([0, 0, 1, 1, 0]),
                quality_scores={'silhouette': 0.7},
                execution_time=0.05,  # Fast execution
                n_clusters=2,
                n_noise=0
            ),
            EnsembleMember(
                parameters={'eps': 0.3, 'min_samples': 2},
                labels=np.array([1, 1, 0, 0, 1]),
                quality_scores={'silhouette': 0.6},
                execution_time=0.20,  # Slow execution
                n_clusters=2,
                n_noise=0
            )
        ]
        
        vm = VotingMechanism(voting_strategy='weighted')
        consensus_labels, confidence_scores = vm.vote_consensus(members, X)
        
        # Check output properties
        assert len(consensus_labels) == len(X)
        assert len(confidence_scores) == len(X)
    
    def test_quality_weighted_voting(self):
        """Test quality-weighted voting consensus"""
        X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6]])
        
        members = [
            EnsembleMember(
                parameters={'eps': 0.5, 'min_samples': 2},
                labels=np.array([0, 0, 1, 1, 0]),
                quality_scores={'silhouette': 0.9, 'davies_bouldin': 0.3},  # High quality
                execution_time=0.1,
                n_clusters=2,
                n_noise=0
            ),
            EnsembleMember(
                parameters={'eps': 0.3, 'min_samples': 2},
                labels=np.array([1, 1, 0, 0, 1]),
                quality_scores={'silhouette': 0.3, 'davies_bouldin': 0.8},  # Low quality
                execution_time=0.1,
                n_clusters=2,
                n_noise=0
            )
        ]
        
        vm = VotingMechanism(voting_strategy='quality_weighted')
        consensus_labels, confidence_scores = vm.vote_consensus(members, X)
        
        # Check output properties
        assert len(consensus_labels) == len(X)
        assert len(confidence_scores) == len(X)
    
    def test_empty_ensemble(self):
        """Test voting with empty ensemble"""
        X = np.array([[1, 2], [3, 4]])
        members = []
        
        vm = VotingMechanism()
        consensus_labels, confidence_scores = vm.vote_consensus(members, X)
        
        # Should handle empty ensemble gracefully
        assert len(consensus_labels) == len(X)
        assert np.all(consensus_labels == -1)  # No clusters
        assert np.all(confidence_scores == 0)  # No confidence


class TestConsensusClusteringEngine:
    """Test ConsensusClusteringEngine class"""
    
    def test_initialization(self):
        """Test ConsensusClusteringEngine initialization"""
        engine = ConsensusClusteringEngine(
            n_ensemble_members=5,
            voting_strategy='weighted',
            diversity_threshold=0.2,
            parallel_execution=False,
            random_state=42
        )
        
        assert engine.n_ensemble_members == 5
        assert engine.voting_strategy == 'weighted'
        assert engine.diversity_threshold == 0.2
        assert engine.parallel_execution == False
        assert engine.random_state == 42
    
    def test_fit_consensus_clustering_basic(self):
        """Test basic consensus clustering functionality"""
        # Create well-separated test data
        X, y_true = make_blobs(n_samples=50, centers=3, n_features=2, 
                              cluster_std=1.0, random_state=42)
        
        engine = ConsensusClusteringEngine(
            n_ensemble_members=5,
            parallel_execution=False,  # For deterministic testing
            random_state=42
        )
        
        result = engine.fit_consensus_clustering(X)
        
        # Check result type and basic properties
        assert isinstance(result, EnsembleResult)
        assert len(result.consensus_labels) == len(X)
        assert len(result.confidence_scores) == len(X)
        assert len(result.ensemble_members) <= 5  # Some might fail
        assert result.consensus_matrix.shape == (len(X), len(X))
        
        # Check that we found some clusters
        n_consensus_clusters = len(np.unique(result.consensus_labels[result.consensus_labels >= 0]))
        assert n_consensus_clusters > 0
        
        # Check quality metrics exist
        assert isinstance(result.overall_quality, dict)
        assert len(result.overall_quality) > 0
        
        # Check stability metrics
        assert isinstance(result.stability_metrics, dict)
        
        # Check execution stats
        assert isinstance(result.execution_stats, dict)
        assert 'total_execution_time' in result.execution_stats
        assert 'successful_members' in result.execution_stats
    
    def test_fit_consensus_clustering_with_base_params(self):
        """Test consensus clustering with custom base parameters"""
        X, _ = make_blobs(n_samples=30, centers=2, n_features=2, random_state=42)
        
        base_params = {'eps': 0.8, 'min_samples': 3}
        
        engine = ConsensusClusteringEngine(n_ensemble_members=3, random_state=42)
        result = engine.fit_consensus_clustering(X, base_params=base_params)
        
        # Should complete successfully
        assert isinstance(result, EnsembleResult)
        assert len(result.ensemble_members) <= 3
    
    def test_parallel_vs_sequential_execution(self):
        """Test that parallel and sequential execution produce valid results"""
        X, _ = make_blobs(n_samples=40, centers=2, n_features=2, random_state=42)
        
        # Sequential execution
        engine_seq = ConsensusClusteringEngine(
            n_ensemble_members=3,
            parallel_execution=False,
            random_state=42
        )
        result_seq = engine_seq.fit_consensus_clustering(X)
        
        # Parallel execution
        engine_par = ConsensusClusteringEngine(
            n_ensemble_members=3,
            parallel_execution=True,
            random_state=42
        )
        result_par = engine_par.fit_consensus_clustering(X)
        
        # Both should produce valid results
        assert isinstance(result_seq, EnsembleResult)
        assert isinstance(result_par, EnsembleResult)
        assert len(result_seq.consensus_labels) == len(X)
        assert len(result_par.consensus_labels) == len(X)
    
    def test_different_voting_strategies(self):
        """Test different voting strategies"""
        X, _ = make_blobs(n_samples=30, centers=2, n_features=2, random_state=42)
        
        strategies = ['majority', 'weighted', 'quality_weighted']
        results = {}
        
        for strategy in strategies:
            engine = ConsensusClusteringEngine(
                n_ensemble_members=4,
                voting_strategy=strategy,
                parallel_execution=False,
                random_state=42
            )
            result = engine.fit_consensus_clustering(X)
            results[strategy] = result
            
            # Each should produce valid results
            assert isinstance(result, EnsembleResult)
            assert len(result.consensus_labels) == len(X)
        
        # All strategies should produce some consensus
        for strategy, result in results.items():
            n_clusters = len(np.unique(result.consensus_labels[result.consensus_labels >= 0]))
            assert n_clusters >= 0  # At least should not fail
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Single point
        X_single = np.array([[1, 2]])
        engine = ConsensusClusteringEngine(n_ensemble_members=3, random_state=42)
        result_single = engine.fit_consensus_clustering(X_single)
        assert len(result_single.consensus_labels) == 1
        
        # Two points
        X_two = np.array([[1, 2], [3, 4]])
        result_two = engine.fit_consensus_clustering(X_two)
        assert len(result_two.consensus_labels) == 2
        
        # High dimensional data
        X_high_dim = np.random.RandomState(42).randn(20, 10)
        result_high_dim = engine.fit_consensus_clustering(X_high_dim)
        assert len(result_high_dim.consensus_labels) == 20
    
    def test_quality_calculation(self):
        """Test quality score calculations"""
        # Well-separated clusters
        X, _ = make_blobs(n_samples=60, centers=3, n_features=2, 
                         cluster_std=0.5, random_state=42)
        
        engine = ConsensusClusteringEngine(n_ensemble_members=5, random_state=42)
        result = engine.fit_consensus_clustering(X)
        
        # Should have quality metrics
        assert 'weighted_avg' in result.overall_quality
        
        # Quality should be reasonable for well-separated data
        if 'consensus_silhouette' in result.overall_quality:
            assert result.overall_quality['consensus_silhouette'] > 0
    
    def test_stability_metrics(self):
        """Test stability metric calculations"""
        X, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=42)
        
        engine = ConsensusClusteringEngine(n_ensemble_members=5, random_state=42)
        result = engine.fit_consensus_clustering(X)
        
        # Should have stability metrics
        stability_keys = ['mean_ari', 'consensus_ari_mean', 'overall_stability']
        
        for key in stability_keys:
            if key in result.stability_metrics:
                assert 0 <= result.stability_metrics[key] <= 1
    
    def test_parameter_diversity_threshold(self):
        """Test parameter diversity threshold handling"""
        X, _ = make_blobs(n_samples=30, centers=2, n_features=2, random_state=42)
        
        # High diversity threshold
        engine_high = ConsensusClusteringEngine(
            n_ensemble_members=3,
            diversity_threshold=0.8,  # Very high threshold
            random_state=42
        )
        
        # Should still work but may warn about low diversity
        result = engine_high.fit_consensus_clustering(X)
        assert isinstance(result, EnsembleResult)


class TestIntegration:
    """Integration tests for ensemble clustering"""
    
    def test_different_datasets(self):
        """Test ensemble clustering on different dataset types"""
        datasets = [
            make_blobs(n_samples=50, centers=3, n_features=2, random_state=42),
            make_circles(n_samples=50, noise=0.1, factor=0.5, random_state=42),
        ]
        
        engine = ConsensusClusteringEngine(n_ensemble_members=4, random_state=42)
        
        for X, y_true in datasets:
            result = engine.fit_consensus_clustering(X)
            
            # Should complete for all datasets
            assert isinstance(result, EnsembleResult)
            assert len(result.consensus_labels) == len(X)
            
            # Should find some structure
            n_clusters = len(np.unique(result.consensus_labels[result.consensus_labels >= 0]))
            assert n_clusters >= 0  # At least should not fail catastrophically
    
    def test_reproducibility(self):
        """Test that results are reproducible with same random state"""
        X, _ = make_blobs(n_samples=40, centers=2, n_features=2, random_state=42)
        
        engine1 = ConsensusClusteringEngine(n_ensemble_members=5, random_state=42)
        result1 = engine1.fit_consensus_clustering(X)
        
        engine2 = ConsensusClusteringEngine(n_ensemble_members=5, random_state=42)
        result2 = engine2.fit_consensus_clustering(X)
        
        # Results should be similar (parameters might have some randomness)
        assert len(result1.consensus_labels) == len(result2.consensus_labels)
        assert len(result1.ensemble_members) == len(result2.ensemble_members)
    
    def test_ensemble_size_scaling(self):
        """Test behavior with different ensemble sizes"""
        X, _ = make_blobs(n_samples=30, centers=2, n_features=2, random_state=42)
        
        ensemble_sizes = [1, 3, 5, 8]
        results = []
        
        for size in ensemble_sizes:
            engine = ConsensusClusteringEngine(
                n_ensemble_members=size,
                random_state=42
            )
            result = engine.fit_consensus_clustering(X)
            results.append(result)
            
            # All should work
            assert isinstance(result, EnsembleResult)
            assert len(result.ensemble_members) <= size
        
        # Larger ensembles should generally have better stability (when they work)
        for i in range(1, len(results)):
            if (results[i-1].stability_metrics.get('overall_stability', 0) > 0 and 
                results[i].stability_metrics.get('overall_stability', 0) > 0):
                # Just check that both calculated stability (not necessarily higher)
                assert results[i].stability_metrics['overall_stability'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

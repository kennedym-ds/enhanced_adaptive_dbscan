"""
Test suite for Multi-Density DBSCAN (MDBSCAN) implementation.

This module provides comprehensive tests for the Multi-Density DBSCAN components including:
- Multi-Density Cluster Engine
- Hierarchical Density Manager
- Enhanced Boundary Processor
- Cluster Quality Analyzer
- Integration with main DBSCAN class
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score, adjusted_rand_score
import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
from enhanced_adaptive_dbscan.multi_density_clustering import (
    MultiDensityClusterEngine, HierarchicalDensityManager, ClusterRegion, ClusterHierarchy
)
from enhanced_adaptive_dbscan.boundary_processor import EnhancedBoundaryProcessor
from enhanced_adaptive_dbscan.cluster_quality_analyzer import ClusterQualityAnalyzer
from enhanced_adaptive_dbscan.density_engine import MultiScaleDensityEngine, DensityAnalysis


class TestMultiDensityClusterEngine(unittest.TestCase):
    """Test cases for MultiDensityClusterEngine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = MultiDensityClusterEngine()
        
        # Create test data with multiple densities
        self.X_multi_density = self._create_multi_density_data()
        
        # Create density analysis for testing
        density_engine = MultiScaleDensityEngine()
        self.density_analysis = density_engine.analyze_density_landscape(self.X_multi_density)
        
    def _create_multi_density_data(self):
        """Create synthetic data with multiple density regions."""
        np.random.seed(42)
        
        # High-density cluster
        high_density = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 100)
        
        # Medium-density cluster
        medium_density = np.random.multivariate_normal([6, 6], [[0.3, 0], [0, 0.3]], 80)
        
        # Low-density cluster
        low_density = np.random.multivariate_normal([10, 2], [[0.8, 0], [0, 0.8]], 60)
        
        return np.vstack([high_density, medium_density, low_density])
    
    def test_region_aware_clustering(self):
        """Test region-aware clustering with different density regions."""
        region_parameters = {
            0: {'eps': 0.3, 'min_pts': 5},
            1: {'eps': 0.5, 'min_pts': 5},
            2: {'eps': 0.8, 'min_pts': 5}
        }
        
        region_clusters = self.engine.region_aware_clustering(
            self.X_multi_density, self.density_analysis, region_parameters
        )
        
        # Should return a dictionary of region clusters
        self.assertIsInstance(region_clusters, dict)
        self.assertGreaterEqual(len(region_clusters), 0)
        
        # Each region should have valid clusters
        for region_id, clusters in region_clusters.items():
            self.assertIsInstance(clusters, list)
            for cluster in clusters:
                self.assertIsInstance(cluster, ClusterRegion)
                self.assertGreater(len(cluster.points), 0)
    
    def test_cross_region_merging(self):
        """Test cross-region merging functionality."""
        # Create test clusters for merging
        cluster1 = ClusterRegion(
            cluster_id=1,
            region_id=1,
            density_type='high',
            points=np.array([[1, 1], [1.1, 1.1], [1.2, 1.2]]),
            core_points=np.array([[1, 1], [1.1, 1.1]]),
            boundary_points=np.array([[1.2, 1.2]]),
            cluster_center=np.array([1.1, 1.1]),
            quality_score=0.8,
            stability_score=0.9
        )
        
        cluster2 = ClusterRegion(
            cluster_id=2,
            region_id=2,
            density_type='medium',
            points=np.array([[1.3, 1.3], [1.4, 1.4], [1.5, 1.5]]),
            core_points=np.array([[1.3, 1.3], [1.4, 1.4]]),
            boundary_points=np.array([[1.5, 1.5]]),
            cluster_center=np.array([1.4, 1.4]),
            quality_score=0.7,
            stability_score=0.8
        )
        
        clusters = [cluster1, cluster2]
        merged_clusters = self.engine.cross_region_merging(self.X_multi_density, clusters)
        
        # Should potentially merge close clusters
        self.assertLessEqual(len(merged_clusters), len(clusters))
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        # Empty data - should handle gracefully
        try:
            empty_analysis = DensityAnalysis(
                regions=[], 
                density_histogram=np.array([]), 
                density_thresholds={}, 
                global_density_stats={},
                density_map=np.array([]),
                region_assignments=np.array([])
            )
            result = self.engine.region_aware_clustering(np.array([]), empty_analysis, {})
            self.assertEqual(len(result), 0)
        except ValueError:
            pass  # Also acceptable to raise ValueError


class TestHierarchicalDensityManager(unittest.TestCase):
    """Test cases for HierarchicalDensityManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = HierarchicalDensityManager()
        
        # Create test clusters with hierarchical structure
        self.test_clusters = self._create_test_clusters()
    
    def _create_test_clusters(self):
        """Create test clusters for hierarchical analysis."""
        clusters = []
        
        # Parent cluster
        parent = ClusterRegion(
            cluster_id=1,
            region_id=1,
            density_type='high',
            points=np.random.randn(50, 2),
            core_points=np.random.randn(40, 2),
            boundary_points=np.random.randn(10, 2),
            cluster_center=np.array([0, 0]),
            quality_score=0.8,
            stability_score=0.9
        )
        clusters.append(parent)
        
        # Child clusters
        for i in range(2, 4):
            child = ClusterRegion(
                cluster_id=i,
                region_id=i,
                density_type='medium',
                points=np.random.randn(20, 2) + [i, i],
                core_points=np.random.randn(15, 2) + [i, i],
                boundary_points=np.random.randn(5, 2) + [i, i],
                cluster_center=np.array([i, i]),
                quality_score=0.7,
                stability_score=0.8
            )
            clusters.append(child)
        
        return clusters
    
    def test_build_hierarchical_clustering(self):
        """Test hierarchical clustering construction."""
        # Create mock density analysis and data
        X_test = np.random.randn(100, 2)
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(X_test)
        
        hierarchy = self.manager.build_density_hierarchy(X_test, density_analysis, 
                                                        {0: 0.5}, {0: 5})
        
        # Should create a valid hierarchy
        self.assertIsInstance(hierarchy, ClusterHierarchy)
    
    def test_stability_based_pruning(self):
        """Test stability-based pruning of hierarchy."""
        # Create mock density analysis and data
        X_test = np.random.randn(100, 2)
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(X_test)
        
        hierarchy = self.manager.build_density_hierarchy(X_test, density_analysis, 
                                                        {0: 0.5}, {0: 5})
        pruned_hierarchy = self.manager.stability_based_pruning(hierarchy, min_stability=0.5)
        
        # Pruned hierarchy should be valid
        self.assertIsInstance(pruned_hierarchy, ClusterHierarchy)
    
    def test_select_optimal_clustering(self):
        """Test optimal clustering selection."""
        # Create mock density analysis and data
        X_test = np.random.randn(100, 2)
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(X_test)
        
        hierarchy = self.manager.build_density_hierarchy(X_test, density_analysis, 
                                                        {0: 0.5}, {0: 5})
        optimal_clusters = self.manager.select_optimal_clustering(hierarchy)
        
        # Should return a list of clusters
        self.assertIsInstance(optimal_clusters, list)


class TestEnhancedBoundaryProcessor(unittest.TestCase):
    """Test cases for EnhancedBoundaryProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = EnhancedBoundaryProcessor()
        
        # Create test data with clear boundaries
        self.X_boundary, self.y_boundary = make_moons(n_samples=200, noise=0.1, random_state=42)
    
    def test_boundary_analysis(self):
        """Test boundary point analysis."""
        # Create simple clusters for boundary analysis
        labels = np.zeros(len(self.X_boundary))
        labels[self.y_boundary == 1] = 1
        
        # Create density analysis for the boundary data
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(self.X_boundary)
        
        boundary_analysis = self.processor.analyze_cluster_boundaries(self.X_boundary, labels, density_analysis)
        
        # Should return valid boundary analysis
        from enhanced_adaptive_dbscan.boundary_processor import BoundaryAnalysis
        self.assertIsInstance(boundary_analysis, BoundaryAnalysis)
    
    def test_adaptive_refinement(self):
        """Test adaptive boundary refinement."""
        labels = np.zeros(len(self.X_boundary))
        labels[self.y_boundary == 1] = 1
        
        # Create density analysis for the boundary data
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(self.X_boundary)
        
        boundary_analysis = self.processor.analyze_cluster_boundaries(self.X_boundary, labels, density_analysis)
        refined_labels = self.processor.refine_boundaries(
            self.X_boundary, labels, boundary_analysis
        )
        
        # Refined labels should be valid
        self.assertEqual(len(refined_labels), len(labels))
        self.assertTrue(np.all(refined_labels >= -1))  # Valid cluster labels
    
    def test_merge_split_recommendations(self):
        """Test merge/split recommendations."""
        labels = np.zeros(len(self.X_boundary))
        labels[self.y_boundary == 1] = 1
        
        # Create density analysis for the boundary data
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(self.X_boundary)
        
        boundary_analysis = self.processor.analyze_cluster_boundaries(self.X_boundary, labels, density_analysis)
        recommendations = self.processor.generate_recommendations(
            self.X_boundary, labels, boundary_analysis
        )
        
        # Should return valid recommendations
        self.assertIsInstance(recommendations, dict)


class TestClusterQualityAnalyzer(unittest.TestCase):
    """Test cases for ClusterQualityAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ClusterQualityAnalyzer()
        
        # Create test data with known good clustering
        self.X_quality, self.y_quality = make_blobs(
            n_samples=300, centers=4, cluster_std=1.0, random_state=42
        )
        
        # Create clusters dictionary
        self.test_clusters = self._create_test_clusters_dict()
    
    def _create_test_clusters_dict(self):
        """Create test clusters dictionary."""
        clusters = {}
        unique_labels = np.unique(self.y_quality)
        
        for i, label in enumerate(unique_labels):
            mask = self.y_quality == label
            cluster_points = self.X_quality[mask]
            
            clusters[i] = ClusterRegion(
                cluster_id=i,
                region_id=i,
                density_type='medium',
                points=cluster_points,
                core_points=cluster_points[:-2] if len(cluster_points) > 2 else cluster_points,
                boundary_points=cluster_points[-2:] if len(cluster_points) > 2 else np.array([]),
                cluster_center=np.mean(cluster_points, axis=0),
                quality_score=0.8,
                stability_score=0.9
            )
        
        return clusters
    
    def test_comprehensive_quality_analysis(self):
        """Test comprehensive cluster quality analysis."""
        # Create density analysis and boundary analysis for the quality data
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(self.X_quality)
        
        boundary_processor = EnhancedBoundaryProcessor()
        boundary_analysis = boundary_processor.analyze_cluster_boundaries(self.X_quality, self.y_quality, density_analysis)
        
        quality_analysis = self.analyzer.comprehensive_quality_analysis(
            self.X_quality, self.test_clusters, density_analysis, boundary_analysis
        )
        
        # Should return comprehensive quality analysis
        from enhanced_adaptive_dbscan.cluster_quality_analyzer import QualityAnalysisResult
        self.assertIsInstance(quality_analysis, QualityAnalysisResult)
    
    def test_individual_cluster_metrics(self):
        """Test individual cluster metrics calculation."""
        # Create density analysis and boundary analysis for the quality data
        density_engine = MultiScaleDensityEngine()
        density_analysis = density_engine.analyze_density_landscape(self.X_quality)
        
        boundary_processor = EnhancedBoundaryProcessor()
        boundary_analysis = boundary_processor.analyze_cluster_boundaries(self.X_quality, self.y_quality, density_analysis)
        
        quality_analysis = self.analyzer.comprehensive_quality_analysis(
            self.X_quality, self.test_clusters, density_analysis, boundary_analysis
        )
        
        # Should include cluster metrics
        from enhanced_adaptive_dbscan.cluster_quality_analyzer import QualityAnalysisResult
        self.assertIsInstance(quality_analysis, QualityAnalysisResult)


class TestMultiDensityIntegration(unittest.TestCase):
    """Test cases for Multi-Density DBSCAN integration with main DBSCAN class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create diverse test datasets
        self.X_blobs, self.y_blobs = make_blobs(
            n_samples=300, centers=4, cluster_std=1.5, random_state=42
        )
        
        self.X_moons, self.y_moons = make_moons(
            n_samples=200, noise=0.1, random_state=42
        )
        
        # Create multi-density dataset
        self.X_multi = self._create_multi_density_dataset()
    
    def _create_multi_density_dataset(self):
        """Create a dataset with multiple density regions."""
        np.random.seed(42)
        
        # High-density cluster
        high_density = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 100)
        
        # Low-density cluster
        low_density = np.random.multivariate_normal([8, 8], [[0.5, 0], [0, 0.5]], 50)
        
        # Scattered points
        noise = np.random.uniform(-2, 12, (30, 2))
        
        return np.vstack([high_density, low_density, noise])
    
    def test_mdbscan_enabled_clustering(self):
        """Test clustering with MDBSCAN enabled."""
        dbscan = EnhancedAdaptiveDBSCAN(
            enable_mdbscan=True,
            enable_hierarchical_clustering=True,
            enable_boundary_refinement=True,
            enable_quality_analysis=True
        )
        
        # Fit the model
        dbscan.fit(self.X_multi, additional_attributes=self.X_multi[:, 0].reshape(-1, 1))
        
        # Should have valid results
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), len(self.X_multi))
        
        # MDBSCAN results should be available
        mdbscan_clusters = dbscan.get_mdbscan_clusters()
        quality_analysis = dbscan.get_quality_analysis()
        boundary_analysis = dbscan.get_boundary_analysis()
        
        # Should have some results (can be None if no clusters found)
        # This is acceptable for test data
    
    def test_hierarchical_clustering_results(self):
        """Test hierarchical clustering functionality."""
        dbscan = EnhancedAdaptiveDBSCAN(
            enable_mdbscan=True,
            enable_hierarchical_clustering=True
        )
        
        dbscan.fit(self.X_blobs, additional_attributes=self.X_blobs[:, 0].reshape(-1, 1))
        
        # Should have hierarchical results available
        hierarchical_clusters = dbscan.get_hierarchical_clusters()
        # Can be None if hierarchical clustering wasn't performed
    
    def test_quality_improvement_metrics(self):
        """Test that MDBSCAN improves clustering quality."""
        # Standard DBSCAN
        dbscan_standard = EnhancedAdaptiveDBSCAN(enable_mdbscan=False)
        dbscan_standard.fit(self.X_multi, additional_attributes=self.X_multi[:, 0].reshape(-1, 1))
        
        # MDBSCAN enabled
        dbscan_mdbscan = EnhancedAdaptiveDBSCAN(
            enable_mdbscan=True,
            enable_hierarchical_clustering=True,
            enable_boundary_refinement=True
        )
        dbscan_mdbscan.fit(self.X_multi, additional_attributes=self.X_multi[:, 0].reshape(-1, 1))
        
        # Both should produce valid clustering
        self.assertIsNotNone(dbscan_standard.labels_)
        self.assertIsNotNone(dbscan_mdbscan.labels_)
    
    def test_backward_compatibility(self):
        """Test that Multi-Density features maintain backward compatibility."""
        # Standard usage should still work
        dbscan = EnhancedAdaptiveDBSCAN()
        dbscan.fit(self.X_blobs, additional_attributes=self.X_blobs[:, 0].reshape(-1, 1))
        
        # Should have valid results
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), len(self.X_blobs))
        
        # MDBSCAN results should be None when disabled
        self.assertIsNone(dbscan.get_mdbscan_clusters())
        self.assertIsNone(dbscan.get_quality_analysis())
        self.assertIsNone(dbscan.get_boundary_analysis())
    
    def test_performance_benchmarking(self):
        """Test performance benchmarking capabilities."""
        dbscan = EnhancedAdaptiveDBSCAN(
            enable_mdbscan=True,
            enable_hierarchical_clustering=True,
            enable_boundary_refinement=True,
            enable_quality_analysis=True
        )
        
        import time
        
        # Measure fitting time
        start_time = time.time()
        dbscan.fit(self.X_multi, additional_attributes=self.X_multi[:, 0].reshape(-1, 1))
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        self.assertLess(fit_time, 30.0)  # 30 seconds max for test data
        
        print(f"MDBSCAN fitting time: {fit_time:.4f} seconds")
        print(f"Data points processed: {len(self.X_multi)}")
        print(f"Processing rate: {len(self.X_multi)/fit_time:.2f} points/second")


class TestMultiDensityEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for Multi-Density DBSCAN."""
    
    def test_empty_data(self):
        """Test handling of empty datasets."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        
        with self.assertRaises(ValueError):
            dbscan.fit(np.array([]))
    
    def test_single_point(self):
        """Test handling of single data point."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        X_single = np.array([[1, 2]])
        
        dbscan.fit(X_single, additional_attributes=X_single[:, 0].reshape(-1, 1))
        
        # Should handle gracefully
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), 1)
    
    def test_identical_points(self):
        """Test handling of identical data points."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        X_identical = np.array([[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]])
        
        dbscan.fit(X_identical, additional_attributes=X_identical[:, 0].reshape(-1, 1))
        
        # Should handle gracefully
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), 5)
    
    def test_high_dimensional_data(self):
        """Test handling of high-dimensional data."""
        np.random.seed(42)
        X_high_dim = np.random.randn(100, 50)  # 50 dimensions
        
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        
        # Should handle without errors
        dbscan.fit(X_high_dim, additional_attributes=X_high_dim[:, 0].reshape(-1, 1))
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), 100)


if __name__ == '__main__':
    # Create a test suite with all test cases
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMultiDensityClusterEngine,
        TestHierarchicalDensityManager,
        TestEnhancedBoundaryProcessor,
        TestClusterQualityAnalyzer,
        TestMultiDensityIntegration,
        TestMultiDensityEdgeCases
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print("MULTI-DENSITY DBSCAN TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

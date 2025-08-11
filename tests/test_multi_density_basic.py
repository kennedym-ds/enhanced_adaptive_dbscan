"""
Basic Multi-Density DBSCAN Integration Test

This module provides focused tests to validate the core Multi-Density DBSCAN 
(MDBSCAN) integration works correctly with the Enhanced Adaptive DBSCAN framework.
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs
import sys
import os

# Add the package to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN


class TestMultiDensityBasicIntegration(unittest.TestCase):
    """Basic integration test for Multi-Density DBSCAN functionality."""
    
    def setUp(self):
        """Set up test data."""
        # Create simple test data
        self.X_simple, self.y_simple = make_blobs(
            n_samples=100, centers=3, cluster_std=1.0, random_state=42
        )
        
        # Create multi-density test data
        np.random.seed(42)
        high_density = np.random.multivariate_normal([2, 2], [[0.1, 0], [0, 0.1]], 50)
        low_density = np.random.multivariate_normal([8, 8], [[0.5, 0], [0, 0.5]], 30)
        self.X_multi = np.vstack([high_density, low_density])
    
    def test_basic_mdbscan_enabled(self):
        """Test basic MDBSCAN functionality."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        
        # Should fit without errors
        dbscan.fit(self.X_simple)
        
        # Should have valid results
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), len(self.X_simple))
        
        # Labels should be integers
        self.assertTrue(all(isinstance(label, (int, np.integer)) for label in dbscan.labels_))
    
    def test_mdbscan_disabled_backward_compatibility(self):
        """Test that MDBSCAN disabled mode works (backward compatibility)."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=False)
        
        # Should fit without errors
        dbscan.fit(self.X_simple)
        
        # Should have valid results
        self.assertIsNotNone(dbscan.labels_)
        self.assertEqual(len(dbscan.labels_), len(self.X_simple))
        
        # MDBSCAN results should be None
        self.assertIsNone(dbscan.get_mdbscan_clusters())
        self.assertIsNone(dbscan.get_quality_analysis())
        self.assertIsNone(dbscan.get_boundary_analysis())
        self.assertIsNone(dbscan.get_hierarchical_clusters())
    
    def test_multi_density_components_access(self):
        """Test access to Multi-Density component results."""
        dbscan = EnhancedAdaptiveDBSCAN(
            enable_mdbscan=True,
            enable_hierarchical_clustering=True,
            enable_boundary_refinement=True,
            enable_quality_analysis=True
        )
        
        # Fit the model
        dbscan.fit(self.X_multi)
        
        # Access methods should not raise errors
        mdbscan_clusters = dbscan.get_mdbscan_clusters()
        quality_analysis = dbscan.get_quality_analysis()
        boundary_analysis = dbscan.get_boundary_analysis()
        hierarchical_clusters = dbscan.get_hierarchical_clusters()
        
        # Results can be None or valid data structures
        if mdbscan_clusters is not None:
            self.assertIsInstance(mdbscan_clusters, (list, dict))
        
        if quality_analysis is not None:
            self.assertIsInstance(quality_analysis, dict)
        
        if boundary_analysis is not None:
            self.assertIsInstance(boundary_analysis, dict)
        
        if hierarchical_clusters is not None:
            self.assertIsInstance(hierarchical_clusters, (list, dict))
    
    def test_single_parameter_combinations(self):
        """Test individual Multi-Density parameter combinations."""
        test_configs = [
            {'enable_mdbscan': True, 'enable_hierarchical_clustering': False},
            {'enable_mdbscan': True, 'enable_boundary_refinement': False},
            {'enable_mdbscan': True, 'enable_quality_analysis': False},
            {'enable_mdbscan': True, 'enable_hierarchical_clustering': True, 'enable_boundary_refinement': False, 'enable_quality_analysis': False}
        ]
        
        for config in test_configs:
            with self.subTest(config=config):
                dbscan = EnhancedAdaptiveDBSCAN(**config)
                
                # Should fit without errors
                dbscan.fit(self.X_simple)
                
                # Should have valid results
                self.assertIsNotNone(dbscan.labels_)
                self.assertEqual(len(dbscan.labels_), len(self.X_simple))
    
    def test_edge_cases(self):
        """Test edge case handling."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        
        # Single point
        X_single = np.array([[1, 2]])
        dbscan.fit(X_single)
        self.assertEqual(len(dbscan.labels_), 1)
        
        # Few points
        X_few = np.array([[1, 2], [3, 4]])
        dbscan.fit(X_few)
        self.assertEqual(len(dbscan.labels_), 2)
        
        # Identical points
        X_identical = np.array([[1, 2], [1, 2], [1, 2]])
        dbscan.fit(X_identical)
        self.assertEqual(len(dbscan.labels_), 3)
    
    def test_fit_predict_method(self):
        """Test fit_predict method with Multi-Density enabled."""
        dbscan = EnhancedAdaptiveDBSCAN(enable_mdbscan=True)
        
        # fit_predict should work
        labels = dbscan.fit_predict(self.X_simple)
        
        # Should return valid labels
        self.assertIsNotNone(labels)
        self.assertEqual(len(labels), len(self.X_simple))
        
        # Should be same as dbscan.labels_
        np.testing.assert_array_equal(labels, dbscan.labels_)
    
    def test_performance_timing(self):
        """Test that Multi-Density DBSCAN completes in reasonable time."""
        dbscan = EnhancedAdaptiveDBSCAN(
            enable_mdbscan=True,
            enable_hierarchical_clustering=True,
            enable_boundary_refinement=True,
            enable_quality_analysis=True
        )
        
        import time
        start_time = time.time()
        dbscan.fit(self.X_multi)
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time for small dataset
        self.assertLess(fit_time, 10.0)  # 10 seconds max
        
        print(f"Multi-Density DBSCAN fit time: {fit_time:.4f} seconds for {len(self.X_multi)} points")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

# tests/test_density_engine.py

import unittest
import numpy as np
from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
from enhanced_adaptive_dbscan.density_engine import (
    MultiScaleDensityEngine, 
    RelativeDensityComputer, 
    DynamicBoundaryManager,
    DensityRegion,
    DensityAnalysis
)

class TestDensityEngine(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with multiple density regions."""
        np.random.seed(42)
        
        # Create multi-density dataset
        # High density cluster
        high_density = np.random.normal([2, 2], 0.3, (100, 2))
        
        # Medium density cluster  
        medium_density = np.random.normal([-2, 2], 0.6, (50, 2))
        
        # Low density cluster (more spread out)
        low_density = np.random.normal([0, -3], 1.2, (30, 2))
        
        # Combine all points
        self.X_multi_density = np.vstack([high_density, medium_density, low_density])
        
        # Simple dataset for basic testing
        self.X_simple = np.random.randn(100, 2)
        
    def test_relative_density_computer_initialization(self):
        """Test RelativeDensityComputer initialization."""
        computer = RelativeDensityComputer(k=10, density_bins=30)
        self.assertEqual(computer.k, 10)
        self.assertEqual(computer.density_bins, 30)
        self.assertEqual(computer.low_density_threshold, 0.3)
        self.assertEqual(computer.high_density_threshold, 0.7)
        
    def test_point_density_computation(self):
        """Test basic point density computation."""
        computer = RelativeDensityComputer(k=5)
        densities = computer.compute_point_densities(self.X_simple)
        
        self.assertEqual(len(densities), len(self.X_simple))
        self.assertTrue(np.all(densities > 0))
        self.assertTrue(np.all(np.isfinite(densities)))
        
    def test_relative_density_computation(self):
        """Test relative density computation and thresholds."""
        computer = RelativeDensityComputer(k=5)
        raw_densities = computer.compute_point_densities(self.X_simple)
        relative_densities, thresholds = computer.compute_relative_densities(raw_densities)
        
        # Check normalization
        self.assertAlmostEqual(np.min(relative_densities), 0.0, places=6)
        self.assertAlmostEqual(np.max(relative_densities), 1.0, places=6)
        
        # Check thresholds
        self.assertIn('low', thresholds)
        self.assertIn('high', thresholds)
        self.assertIn('mean', thresholds)
        self.assertIn('std', thresholds)
        
        # Verify threshold ordering
        self.assertLessEqual(thresholds['low'], thresholds['high'])
        
    def test_density_histogram_creation(self):
        """Test density histogram creation."""
        computer = RelativeDensityComputer(k=5, density_bins=20)
        raw_densities = computer.compute_point_densities(self.X_simple)
        relative_densities, _ = computer.compute_relative_densities(raw_densities)
        histogram = computer.create_density_histogram(relative_densities)
        
        self.assertEqual(len(histogram), 20)
        self.assertEqual(np.sum(histogram), len(self.X_simple))
        
    def test_density_partitioning(self):
        """Test partitioning data into density regions."""
        computer = RelativeDensityComputer(k=10)
        raw_densities = computer.compute_point_densities(self.X_multi_density)
        relative_densities, thresholds = computer.compute_relative_densities(raw_densities)
        regions = computer.partition_by_density(self.X_multi_density, relative_densities, thresholds)
        
        # Should have exactly 3 regions (low, medium, high)
        self.assertLessEqual(len(regions), 3)
        self.assertGreaterEqual(len(regions), 1)
        
        # Check region properties
        total_points = 0
        for region in regions:
            self.assertIsInstance(region, DensityRegion)
            self.assertIn(region.density_type, ['low', 'medium', 'high'])
            self.assertGreater(len(region.points), 0)
            self.assertGreater(len(region.indices), 0)
            self.assertEqual(len(region.points), len(region.indices))
            total_points += len(region.points)
            
        # All points should be assigned to some region
        self.assertEqual(total_points, len(self.X_multi_density))
        
    def test_boundary_manager_initialization(self):
        """Test DynamicBoundaryManager initialization."""
        manager = DynamicBoundaryManager(boundary_tolerance=0.2)
        self.assertEqual(manager.boundary_tolerance, 0.2)
        
    def test_multi_scale_density_engine_initialization(self):
        """Test MultiScaleDensityEngine initialization."""
        engine = MultiScaleDensityEngine(
            k=15,
            density_bins=40,
            low_density_threshold=0.2,
            high_density_threshold=0.8
        )
        
        self.assertEqual(engine.k, 15)
        self.assertEqual(engine.density_bins, 40)
        self.assertEqual(engine.low_density_threshold, 0.2)
        self.assertEqual(engine.high_density_threshold, 0.8)
        self.assertIsNotNone(engine.density_computer)
        self.assertIsNotNone(engine.boundary_manager)
        
    def test_full_density_landscape_analysis(self):
        """Test complete density landscape analysis."""
        engine = MultiScaleDensityEngine(k=10)
        analysis = engine.analyze_density_landscape(self.X_multi_density)
        
        # Check analysis structure
        self.assertIsInstance(analysis, DensityAnalysis)
        self.assertIsInstance(analysis.regions, list)
        self.assertIsInstance(analysis.density_histogram, np.ndarray)
        self.assertIsInstance(analysis.density_thresholds, dict)
        self.assertIsInstance(analysis.global_density_stats, dict)
        self.assertIsInstance(analysis.density_map, np.ndarray)
        self.assertIsInstance(analysis.region_assignments, np.ndarray)
        
        # Check array shapes
        self.assertEqual(len(analysis.density_map), len(self.X_multi_density))
        self.assertEqual(len(analysis.region_assignments), len(self.X_multi_density))
        
        # Check that we have multiple regions for multi-density data
        self.assertGreater(len(analysis.regions), 0)
        self.assertLessEqual(len(analysis.regions), 3)
        
        # Check global statistics
        stats = analysis.global_density_stats
        self.assertIn('mean_density', stats)
        self.assertIn('std_density', stats)
        self.assertIn('num_regions', stats)
        self.assertEqual(stats['num_regions'], len(analysis.regions))
        
    def test_region_specific_parameters(self):
        """Test generation of region-specific clustering parameters."""
        engine = MultiScaleDensityEngine(k=10)
        analysis = engine.analyze_density_landscape(self.X_multi_density)
        
        base_eps = 1.0
        base_min_pts = 5
        parameters = engine.get_region_specific_parameters(analysis, base_eps, base_min_pts)
        
        # Check parameter structure
        self.assertIsInstance(parameters, dict)
        self.assertEqual(len(parameters), len(analysis.regions))
        
        # Check parameter values for each region
        for region_id, params in parameters.items():
            self.assertIn('eps', params)
            self.assertIn('min_pts', params)
            self.assertIn('density_type', params)
            self.assertIn('relative_density', params)
            
            # Parameters should be positive
            self.assertGreater(params['eps'], 0)
            self.assertGreater(params['min_pts'], 0)
            self.assertIn(params['density_type'], ['low', 'medium', 'high'])
            
    def test_stability_metrics(self):
        """Test computation of stability metrics."""
        engine = MultiScaleDensityEngine(k=10, enable_stability_analysis=True)
        analysis = engine.analyze_density_landscape(self.X_multi_density)
        stability_scores = engine.compute_stability_metrics(analysis, self.X_multi_density)
        
        # Check stability scores
        self.assertIsInstance(stability_scores, dict)
        self.assertEqual(len(stability_scores), len(analysis.regions))
        
        for region_id, score in stability_scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
            
    def test_analysis_summary(self):
        """Test generation of analysis summary."""
        engine = MultiScaleDensityEngine(k=10)
        analysis = engine.analyze_density_landscape(self.X_multi_density)
        summary = engine.get_analysis_summary(analysis)
        
        self.assertIsInstance(summary, str)
        self.assertIn("Multi-Scale Density Analysis Summary", summary)
        self.assertIn("Total Points:", summary)
        self.assertIn("Regions Identified:", summary)
        
    def test_caching_functionality(self):
        """Test that analysis results are properly cached."""
        engine = MultiScaleDensityEngine(k=10)
        
        # First analysis
        analysis1 = engine.analyze_density_landscape(self.X_simple)
        
        # Second analysis with same data (should use cache)
        analysis2 = engine.analyze_density_landscape(self.X_simple)
        
        # Should return the same object (cached)
        self.assertIs(analysis1, analysis2)
        
        # Force recompute
        analysis3 = engine.analyze_density_landscape(self.X_simple, force_recompute=True)
        
        # Should be different object but same content
        self.assertIsNot(analysis1, analysis3)
        self.assertEqual(len(analysis1.regions), len(analysis3.regions))


class TestEnhancedDBSCANWithMultiScale(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and models."""
        np.random.seed(42)
        
        # Create multi-density test data
        high_density = np.random.normal([3, 3], 0.4, (80, 2))
        medium_density = np.random.normal([-3, 3], 0.8, (40, 2))
        low_density = np.random.normal([0, -4], 1.5, (20, 2))
        
        self.X_multi = np.vstack([high_density, medium_density, low_density])
        
        # Add severity feature for testing
        severity = np.random.randint(1, 11, size=len(self.X_multi)).reshape(-1, 1)
        self.X_full = np.hstack((self.X_multi, severity))
        
    def test_initialization_with_multi_scale(self):
        """Test initialization with multi-scale density enabled."""
        model = EnhancedAdaptiveDBSCAN(
            enable_multi_scale_density=True,
            low_density_threshold=0.2,
            high_density_threshold=0.8,
            k=10
        )
        
        self.assertTrue(model.enable_multi_scale_density)
        self.assertEqual(model.low_density_threshold, 0.2)
        self.assertEqual(model.high_density_threshold, 0.8)
        self.assertIsNotNone(model.density_engine_)
        
    def test_initialization_without_multi_scale(self):
        """Test initialization with multi-scale density disabled."""
        model = EnhancedAdaptiveDBSCAN(enable_multi_scale_density=False)
        
        self.assertFalse(model.enable_multi_scale_density)
        self.assertIsNone(model.density_engine_)
        self.assertIsNone(model.density_analysis_)
        
    def test_multi_scale_parameter_computation(self):
        """Test multi-scale parameter computation."""
        model = EnhancedAdaptiveDBSCAN(
            enable_multi_scale_density=True,
            k=10,
            max_points=1000  # Avoid subsampling for this test
        )
        
        # Fit with multi-scale enabled
        model.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        
        # Check that density analysis was performed
        self.assertIsNotNone(model.density_analysis_)
        self.assertIsNotNone(model.region_parameters_)
        
        # Check that clustering was successful
        self.assertTrue(hasattr(model, 'labels_'))
        self.assertGreater(len(set(model.labels_)), 1)  # Should find multiple clusters
        
    def test_get_density_analysis(self):
        """Test retrieval of density analysis results."""
        model = EnhancedAdaptiveDBSCAN(
            enable_multi_scale_density=True,
            k=10,
            max_points=1000
        )
        
        # Should raise error before fitting
        with self.assertRaises(ValueError):
            model.get_density_analysis()
            
        # Fit model
        model.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        
        # Should return analysis after fitting
        analysis = model.get_density_analysis()
        self.assertIsInstance(analysis, DensityAnalysis)
        self.assertGreater(len(analysis.regions), 0)
        
    def test_get_region_parameters(self):
        """Test retrieval of region-specific parameters."""
        model = EnhancedAdaptiveDBSCAN(
            enable_multi_scale_density=True,
            k=10,
            max_points=1000
        )
        
        # Should raise error before fitting
        with self.assertRaises(ValueError):
            model.get_region_parameters()
            
        # Fit model
        model.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        
        # Should return parameters after fitting
        parameters = model.get_region_parameters()
        self.assertIsInstance(parameters, dict)
        self.assertGreater(len(parameters), 0)
        
        # Check parameter structure
        for region_id, params in parameters.items():
            self.assertIn('eps', params)
            self.assertIn('min_pts', params)
            self.assertIn('density_type', params)
            
    def test_error_handling_without_multi_scale(self):
        """Test error handling when multi-scale is disabled."""
        model = EnhancedAdaptiveDBSCAN(enable_multi_scale_density=False)
        
        # Fit model
        model.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        
        # Should raise errors when trying to access multi-scale features
        with self.assertRaises(ValueError):
            model.get_density_analysis()
            
        with self.assertRaises(ValueError):
            model.get_region_parameters()
            
    def test_backward_compatibility(self):
        """Test that existing functionality still works without multi-scale."""
        # Test with multi-scale disabled (default behavior)
        model_traditional = EnhancedAdaptiveDBSCAN(
            enable_multi_scale_density=False,
            k=10,
            max_points=1000
        )
        
        model_traditional.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        labels_traditional = model_traditional.labels_
        
        # Should still work as before
        self.assertTrue(hasattr(model_traditional, 'labels_'))
        self.assertEqual(len(labels_traditional), len(self.X_full))
        
        # Test with multi-scale enabled
        model_enhanced = EnhancedAdaptiveDBSCAN(
            enable_multi_scale_density=True,
            k=10,
            max_points=1000
        )
        
        model_enhanced.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        labels_enhanced = model_enhanced.labels_
        
        # Should also work
        self.assertTrue(hasattr(model_enhanced, 'labels_'))
        self.assertEqual(len(labels_enhanced), len(self.X_full))
        
        # Both should find clusters (exact results may differ due to different algorithms)
        self.assertGreater(len(set(labels_traditional)) - 1, 0)  # Exclude noise label -1
        self.assertGreater(len(set(labels_enhanced)) - 1, 0)    # Exclude noise label -1


if __name__ == '__main__':
    unittest.main()

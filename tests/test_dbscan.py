# tests/test_dbscan.py

import unittest
import numpy as np
from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN

class TestEnhancedAdaptiveDBSCAN(unittest.TestCase):
    def setUp(self):
        # Generate a small synthetic dataset for testing
        np.random.seed(0)
        X = np.vstack([
            np.random.randn(50, 2) + np.array([5, 5]),
            np.random.randn(50, 2) + np.array([-5, -5]),
            np.random.randn(50, 2) + np.array([5, -5]),
            np.random.randn(50, 2) + np.array([-5, 5])
        ])
        severity = np.random.randint(1, 11, size=200).reshape(-1, 1)
        self.X_full = np.hstack((X, severity))

        self.model = EnhancedAdaptiveDBSCAN(
            wafer_shape='circular',
            wafer_size=20,
            k=5,
            density_scaling=1.0,
            buffer_ratio=0.1,
            min_scaling=3,
            max_scaling=10,
            n_jobs=2,
            max_points=1000,
            subsample_ratio=0.5,
            random_state=42,
            additional_features=[2],
            feature_weights=[1.0],
            stability_threshold=0.6
        )

    def test_initial_fit(self):
        # Test initial fitting
        self.model.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        labels = self.model.labels_
        self.assertEqual(len(labels), len(self.X_full), "Labels length mismatch.")
        self.assertTrue(len(set(labels)) > 1, "Should detect multiple clusters.")

    def test_incremental_fit(self):
        # Fit initial data
        self.model.fit(self.X_full, additional_attributes=self.X_full[:, 2].reshape(-1, 1))
        initial_labels = self.model.labels_.copy()

        # Generate new data points
        X_new = np.vstack([
            np.random.randn(10, 2) + np.array([5, 5]),  # Should belong to existing cluster
            np.random.randn(10, 2) + np.array([15, 15])  # New cluster
        ])
        severity_new = np.random.randint(1, 11, size=20).reshape(-1, 1)
        X_new_full = np.hstack((X_new, severity_new))

        # Incrementally fit new data
        self.model.fit_incremental(X_new_full, additional_attributes_new=severity_new)
        updated_labels = self.model.labels_

        # Check that labels have been updated
        self.assertEqual(len(updated_labels), len(self.X_full) + 20, "Labels length mismatch after incremental fit.")

    def test_empty_dataset(self):
        # Test handling of empty dataset
        with self.assertRaises(ValueError):
            self.model.fit([], additional_attributes=None)

    def test_single_point_cluster(self):
        # Test with a single point which should be noise
        X_single = np.array([[0, 0, 5]])
        self.model.fit(X_single, additional_attributes=np.array([[5]]))
        labels = self.model.labels_
        self.assertEqual(labels[0], -1, "Single point should be labeled as noise.")

if __name__ == '__main__':
    unittest.main()

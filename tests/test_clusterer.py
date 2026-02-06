"""Tests for WaferClusterer (end-to-end clustering)."""

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from wafer_defect_clustering.clusterer import WaferClusterer
from wafer_defect_clustering.features import DefectFeatureEncoder
from wafer_defect_clustering.wafer import WaferMap


def _make_test_wafer(n_clusters=3, points_per_cluster=30, noise=20, seed=42):
    """Generate a wafer with well-separated clusters for testing."""
    rng = np.random.RandomState(seed)
    wafer = WaferMap(diameter_mm=300)

    all_x, all_y, all_sizes = [], [], []
    cluster_centres = [(-60, 0), (60, 0), (0, 80)]

    for i in range(min(n_clusters, len(cluster_centres))):
        cx, cy = cluster_centres[i]
        x = rng.normal(cx, 5, points_per_cluster)
        y = rng.normal(cy, 5, points_per_cluster)
        all_x.extend(x)
        all_y.extend(y)
        all_sizes.extend(rng.exponential(2.0, points_per_cluster))

    # Add noise
    x_noise = rng.uniform(-130, 130, noise)
    y_noise = rng.uniform(-130, 130, noise)
    all_x.extend(x_noise)
    all_y.extend(y_noise)
    all_sizes.extend(rng.exponential(1.0, noise))

    wafer.add_defects(
        x=np.array(all_x),
        y=np.array(all_y),
        size=np.array(all_sizes),
    )
    return wafer


@pytest.fixture()
def test_wafer():
    return _make_test_wafer()


class TestBasicClustering:
    def test_fit_returns_self(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        result = clusterer.fit(test_wafer)
        assert result is clusterer

    def test_fit_predict_returns_labels(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        labels = clusterer.fit_predict(test_wafer)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == test_wafer.n_defects

    def test_finds_clusters(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        # Should find at least 2 clusters (3 planted + noise)
        assert clusterer.n_clusters >= 2

    def test_noise_fraction(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        assert 0.0 <= clusterer.noise_fraction <= 1.0

    def test_probabilities_exist(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        assert hasattr(clusterer, 'probabilities_')
        assert len(clusterer.probabilities_) == test_wafer.n_defects

    def test_outlier_scores_exist(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        assert hasattr(clusterer, 'outlier_scores_')
        assert len(clusterer.outlier_scores_) == test_wafer.n_defects


class TestEdgeCompensation:
    def test_with_edge_compensation(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=True)
        labels = clusterer.fit_predict(test_wafer)
        assert len(labels) == test_wafer.n_defects

    def test_without_edge_compensation(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=False)
        labels = clusterer.fit_predict(test_wafer)
        assert len(labels) == test_wafer.n_defects


class TestPatternClassification:
    def test_pattern_results_populated(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5, classify_patterns=True)
        clusterer.fit(test_wafer)
        assert isinstance(clusterer.pattern_results_, dict)
        if clusterer.n_clusters > 0:
            assert len(clusterer.pattern_results_) > 0

    def test_pattern_results_skipped(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5, classify_patterns=False)
        clusterer.fit(test_wafer)
        assert clusterer.pattern_results_ == {}


class TestFeatureEncoder:
    def test_with_encoder(self):
        wafer = _make_test_wafer()
        encoder = DefectFeatureEncoder(size_weight=0.5)
        clusterer = WaferClusterer(
            min_cluster_size=5,
            feature_encoder=encoder,
            edge_compensation=False,
        )
        labels = clusterer.fit_predict(wafer)
        assert len(labels) == wafer.n_defects


class TestInputFlexibility:
    def test_wafer_as_x(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        labels = clusterer.fit_predict(test_wafer)
        assert len(labels) == test_wafer.n_defects

    def test_ndarray_as_x(self):
        X = np.random.RandomState(42).normal(0, 30, (100, 2))
        clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=False)
        labels = clusterer.fit_predict(X)
        assert len(labels) == 100

    def test_wafer_kwarg(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        labels = clusterer.fit_predict(wafer=test_wafer)
        assert len(labels) == test_wafer.n_defects

    def test_invalid_x_raises(self):
        clusterer = WaferClusterer()
        with pytest.raises(TypeError):
            clusterer.fit('not_a_wafer')


class TestSummary:
    def test_summary_dataframe(self, test_wafer):
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        df = clusterer.summary()
        assert 'cluster_id' in df.columns
        assert 'pattern_type' in df.columns
        assert 'n_defects' in df.columns
        assert 'zone' in df.columns
        assert len(df) == clusterer.n_clusters

    def test_summary_has_silhouette_column(self, test_wafer):
        """summary() should include a silhouette_score column."""
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        df = clusterer.summary()
        assert 'silhouette_score' in df.columns

    def test_silhouette_values_range(self, test_wafer):
        """Per-cluster silhouette scores should be in [-1, 1]."""
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        df = clusterer.summary()
        if len(df) > 0:
            assert all(df['silhouette_score'].between(-1.0, 1.0))

    def test_silhouette_single_cluster(self):
        """When only 1 cluster exists, silhouette should be 0.0 (undefined)."""
        # Create a tight single cluster with no noise
        wafer = WaferMap(diameter_mm=300)
        rng = np.random.RandomState(99)
        x = rng.normal(0, 3, 50)
        y = rng.normal(0, 3, 50)
        wafer.add_defects(x=x, y=y)
        clusterer = WaferClusterer(
            min_cluster_size=5,
            allow_single_cluster=True,
            edge_compensation=False,
        )
        clusterer.fit(wafer)
        df = clusterer.summary()
        if clusterer.n_clusters == 1:
            assert df['silhouette_score'].iloc[0] == 0.0


class TestPrediction:
    def test_predict_without_edge_compensation_raises(self, test_wafer):
        """predict() raises NotImplementedError with sklearn HDBSCAN."""
        clusterer = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
        )
        clusterer.fit(test_wafer)
        with pytest.raises(NotImplementedError, match='not supported'):
            clusterer.predict(np.array([[-60, 0], [60, 0]]))

    def test_predict_with_edge_compensation_raises(self, test_wafer):
        clusterer = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=True,
        )
        clusterer.fit(test_wafer)
        with pytest.raises(ValueError, match='not supported'):
            clusterer.predict(np.array([[0, 0]]))


class TestSmallWafer:
    def test_too_few_defects(self):
        wafer = WaferMap()
        wafer.add_defects(x=[1, 2], y=[3, 4])
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(wafer)
        assert np.all(clusterer.labels_ == -1)

    def test_empty_wafer(self):
        wafer = WaferMap()
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(wafer)
        assert len(clusterer.labels_) == 0


class TestSklearnCompat:
    def test_get_params(self):
        clusterer = WaferClusterer(min_cluster_size=10)
        params = clusterer.get_params()
        assert params['min_cluster_size'] == 10

    def test_set_params(self):
        clusterer = WaferClusterer()
        clusterer.set_params(min_cluster_size=20)
        assert clusterer.min_cluster_size == 20


class TestSklearnHDBSCAN:
    """Tests for sklearn.cluster.HDBSCAN backend (Q5)."""

    def test_uses_sklearn_hdbscan(self, test_wafer):
        """Internal hdbscan_ attribute should be sklearn.cluster.HDBSCAN."""
        clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=False)
        clusterer.fit(test_wafer)
        assert isinstance(clusterer.hdbscan_, SklearnHDBSCAN)

    def test_outlier_scores_populated(self, test_wafer):
        """outlier_scores_ must still be populated (computed as 1 - probabilities_)."""
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        assert hasattr(clusterer, 'outlier_scores_')
        assert len(clusterer.outlier_scores_) == test_wafer.n_defects
        # Values should be in [0, 1]
        assert np.all(clusterer.outlier_scores_ >= 0)
        assert np.all(clusterer.outlier_scores_ <= 1)

    def test_cluster_persistence_populated(self, test_wafer):
        """cluster_persistence_ should still be available (may be empty array)."""
        clusterer = WaferClusterer(min_cluster_size=5)
        clusterer.fit(test_wafer)
        assert hasattr(clusterer, 'cluster_persistence_')
        assert isinstance(clusterer.cluster_persistence_, np.ndarray)

    def test_precomputed_metric_works(self, test_wafer):
        """Edge compensation (precomputed metric) should work with sklearn HDBSCAN."""
        clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=True)
        labels = clusterer.fit_predict(test_wafer)
        assert len(labels) == test_wafer.n_defects

    def test_predict_raises_not_supported(self, test_wafer):
        """predict() should raise informative error with sklearn HDBSCAN (no approximate_predict)."""
        clusterer = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
        )
        clusterer.fit(test_wafer)
        with pytest.raises((ValueError, NotImplementedError)):
            clusterer.predict(np.array([[0, 0]]))


# ===================================================================
# Iterative Harvesting (M5)
# ===================================================================


def _make_two_blob_wafer(seed=42):
    """Two well-separated blobs for harvesting tests."""
    rng = np.random.default_rng(seed)
    wafer = WaferMap(diameter_mm=300)
    # Blob A: 30 pts at (-60, 0)
    a = rng.normal(loc=[-60, 0], scale=3.0, size=(30, 2))
    # Blob B: 30 pts at (60, 0)
    b = rng.normal(loc=[60, 0], scale=3.0, size=(30, 2))
    # 10 scattered noise points
    noise_ang = rng.uniform(0, 2 * np.pi, 10)
    noise_r = rng.uniform(80, 140, 10)
    noise = np.column_stack([noise_r * np.cos(noise_ang), noise_r * np.sin(noise_ang)])
    pts = np.vstack([a, b, noise])
    wafer.add_defects(x=pts[:, 0], y=pts[:, 1])
    return wafer


class TestHarvesting:
    """Iterative cluster harvesting (M5)."""

    def test_harvesting_off_by_default(self):
        """Default `harvesting=False` produces standard HDBSCAN results."""
        c = WaferClusterer(min_cluster_size=5)
        assert c.harvesting is False

    def test_harvesting_finds_separated_clusters(self):
        """Two separated blobs → both harvested as distinct clusters."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
            harvesting=True,
            classify_patterns=False,
        )
        labels = c.fit_predict(wafer)

        assert labels.shape == (wafer.n_defects,)
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 2, f'Expected ≥2 harvested clusters, got {n_clusters}'

    def test_harvesting_labels_sequential(self):
        """Harvested labels should be 0, 1, 2, ... with no gaps."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
            harvesting=True,
            classify_patterns=False,
        )
        labels = c.fit_predict(wafer)

        cluster_ids = sorted(set(labels) - {-1})
        assert cluster_ids == list(range(len(cluster_ids)))

    def test_harvesting_respects_max_iterations(self):
        """Setting max_iterations=1 should harvest at most 1 cluster."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
            harvesting=True,
            harvesting_max_iterations=1,
            classify_patterns=False,
        )
        labels = c.fit_predict(wafer)

        n_clusters = len(set(labels) - {-1})
        assert n_clusters <= 1, f'Expected ≤1 cluster with max_iter=1, got {n_clusters}'

    def test_harvesting_respects_min_silhouette(self):
        """Very high min_silhouette threshold should reject poor clusters."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
            harvesting=True,
            harvesting_min_silhouette=0.99,  # impossibly high
            classify_patterns=False,
        )
        labels = c.fit_predict(wafer)

        # With such a high threshold, no cluster should be harvested
        n_clusters = len(set(labels) - {-1})
        assert n_clusters == 0, f'Expected 0 clusters at min_sil=0.99, got {n_clusters}'

    def test_harvesting_pattern_classification_runs(self):
        """Pattern classification runs correctly on harvested labels."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
            harvesting=True,
            classify_patterns=True,
        )
        c.fit(wafer)

        # pattern_results_ should have entries for harvested clusters
        n_clusters = len(set(c.labels_) - {-1})
        if n_clusters > 0:
            assert len(c.pattern_results_) > 0

    def test_harvesting_summary_has_harvest_iteration(self):
        """summary() should include harvest_iteration column when harvesting."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=False,
            harvesting=True,
            classify_patterns=False,
        )
        c.fit(wafer)
        df = c.summary()

        if len(df) > 0:
            assert 'harvest_iteration' in df.columns
            # Iterations should start at 1
            assert df['harvest_iteration'].min() >= 1

    def test_harvesting_backward_compat(self, test_wafer):
        """harvesting=False should produce identical results to pre-harvesting."""
        c_off = WaferClusterer(
            min_cluster_size=5,
            harvesting=False,
            classify_patterns=False,
        )
        c_off.fit(test_wafer)

        c_default = WaferClusterer(
            min_cluster_size=5,
            classify_patterns=False,
        )
        c_default.fit(test_wafer)

        np.testing.assert_array_equal(c_off.labels_, c_default.labels_)

    def test_harvesting_with_edge_compensation(self):
        """Harvesting works in combination with edge compensation."""
        wafer = _make_two_blob_wafer()
        c = WaferClusterer(
            min_cluster_size=5,
            edge_compensation=True,
            harvesting=True,
            classify_patterns=False,
        )
        labels = c.fit_predict(wafer)
        assert labels.shape == (wafer.n_defects,)

    def test_harvesting_empty_wafer(self):
        """Empty wafer handled gracefully with harvesting on."""
        wafer = WaferMap(diameter_mm=300)
        c = WaferClusterer(
            min_cluster_size=5,
            harvesting=True,
            classify_patterns=False,
        )
        labels = c.fit_predict(wafer)
        assert len(labels) == 0

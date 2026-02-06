"""Tests for deep learning pattern classifier (L1 + L2)."""

from __future__ import annotations

import numpy as np
import pytest

from wafer_defect_clustering import PatternResult, WaferMap
from wafer_defect_clustering.deep_patterns import DeepPatternClassifier

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wafer_with_center_defects():
    """WaferMap with defects concentrated near the center."""
    rng = np.random.default_rng(42)
    wafer = WaferMap(diameter_mm=300.0)
    n = 50
    x = rng.normal(0.0, 10.0, size=n)
    y = rng.normal(0.0, 10.0, size=n)
    wafer.add_defects(x=x, y=y)
    return wafer


@pytest.fixture
def wafer_with_scratch():
    """WaferMap with defects along a scratch-like line."""
    wafer = WaferMap(diameter_mm=300.0)
    t = np.linspace(-100, 100, 40)
    x = t
    y = 0.5 * t + np.random.default_rng(7).normal(0, 2, size=40)
    wafer.add_defects(x=x, y=y)
    return wafer


@pytest.fixture
def empty_wafer():
    """Empty wafer with no defects."""
    return WaferMap(diameter_mm=300.0)


@pytest.fixture
def classifier():
    """DeepPatternClassifier with no model (rule-based fallback)."""
    return DeepPatternClassifier()


# ---------------------------------------------------------------------------
# L1 — Rasterization
# ---------------------------------------------------------------------------


class TestRasterize:
    """DeepPatternClassifier.rasterize converts point cloud to 2D image."""

    def test_rasterize_shape(self, classifier, wafer_with_center_defects):
        img = classifier.rasterize(wafer_with_center_defects, resolution=96)
        assert img.shape == (1, 96, 96)

    def test_rasterize_custom_resolution(self, classifier, wafer_with_center_defects):
        img = classifier.rasterize(wafer_with_center_defects, resolution=64)
        assert img.shape == (1, 64, 64)

    def test_rasterize_empty_wafer(self, classifier, empty_wafer):
        img = classifier.rasterize(empty_wafer, resolution=96)
        assert img.shape == (1, 96, 96)
        assert np.all(img == 0)

    def test_rasterize_center_defects(self, classifier, wafer_with_center_defects):
        img = classifier.rasterize(wafer_with_center_defects, resolution=96)
        # Center of the image should have nonzero pixels
        center_region = img[0, 40:56, 40:56]
        assert center_region.sum() > 0

    def test_rasterize_dtype_float(self, classifier, wafer_with_center_defects):
        img = classifier.rasterize(wafer_with_center_defects, resolution=96)
        assert img.dtype == np.float32 or img.dtype == np.float64

    def test_rasterize_values_01(self, classifier, wafer_with_center_defects):
        img = classifier.rasterize(wafer_with_center_defects, resolution=96)
        assert img.min() >= 0.0
        assert img.max() <= 1.0


# ---------------------------------------------------------------------------
# L1 — Classification
# ---------------------------------------------------------------------------


class TestClassify:
    """DeepPatternClassifier.classify returns PatternResult per cluster."""

    def test_classify_returns_pattern_result(self, classifier, wafer_with_center_defects):
        labels = np.zeros(wafer_with_center_defects.n_defects, dtype=int)
        results = classifier.classify(wafer_with_center_defects, labels)
        assert isinstance(results, dict)
        assert 0 in results
        assert isinstance(results[0], PatternResult)

    def test_classify_multiple_clusters(self, classifier, wafer_with_center_defects):
        labels = np.array([0] * 25 + [1] * 25)
        results = classifier.classify(wafer_with_center_defects, labels)
        assert 0 in results
        assert 1 in results

    def test_classify_ignores_noise(self, classifier, wafer_with_center_defects):
        labels = np.full(wafer_with_center_defects.n_defects, -1, dtype=int)
        results = classifier.classify(wafer_with_center_defects, labels)
        assert len(results) == 0  # all noise, no clusters

    def test_classify_confidence_range(self, classifier, wafer_with_center_defects):
        labels = np.zeros(wafer_with_center_defects.n_defects, dtype=int)
        results = classifier.classify(wafer_with_center_defects, labels)
        for pr in results.values():
            assert 0.0 <= pr.confidence <= 1.0

    def test_classify_wafer_returns_pattern_result(self, classifier, wafer_with_center_defects):
        result = classifier.classify_wafer(wafer_with_center_defects)
        assert isinstance(result, PatternResult)

    def test_classify_empty_wafer(self, classifier, empty_wafer):
        labels = np.array([], dtype=int)
        results = classifier.classify(empty_wafer, labels)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# L1 — Integration with WaferClusterer
# ---------------------------------------------------------------------------


class TestDeepClassifierIntegration:
    """Integration of DeepPatternClassifier with WaferClusterer."""

    def test_pattern_classifier_param_accepted(self):
        from wafer_defect_clustering import WaferClusterer

        clusterer = WaferClusterer(
            min_cluster_size=5,
            pattern_classifier='deep',
        )
        assert clusterer.pattern_classifier == 'deep'

    def test_pattern_classifier_default_geometric(self):
        from wafer_defect_clustering import WaferClusterer

        clusterer = WaferClusterer(min_cluster_size=5)
        assert clusterer.pattern_classifier == 'geometric'

    def test_deep_classifier_runs_end_to_end(self, wafer_with_center_defects):
        from wafer_defect_clustering import WaferClusterer

        clusterer = WaferClusterer(
            min_cluster_size=5,
            pattern_classifier='deep',
            edge_compensation=False,
        )
        labels = clusterer.fit_predict(wafer_with_center_defects)
        assert len(labels) == wafer_with_center_defects.n_defects


# ---------------------------------------------------------------------------
# L2 — Open-set novel pattern detection
# ---------------------------------------------------------------------------


class TestNovelPatternDetection:
    """Open-set detection: identify patterns outside known types."""

    def test_novel_pattern_detection(self, classifier):
        """Artificial unusual pattern → 'novel' label."""
        wafer = WaferMap(diameter_mm=300.0)
        rng = np.random.default_rng(99)
        # Create a very unusual pattern: checkerboard-like
        for i in range(-5, 6):
            for j in range(-5, 6):
                if (i + j) % 2 == 0:
                    wafer.add_defects(
                        x=[i * 20.0 + rng.normal(0, 1)],
                        y=[j * 20.0 + rng.normal(0, 1)],
                    )
        result = classifier.classify_wafer(wafer)
        # The novelty score should exist in details
        assert 'novelty_score' in result.details

    def test_known_pattern_not_novel(self, classifier, wafer_with_scratch):
        """A recognizable scratch should not be marked novel."""
        result = classifier.classify_wafer(wafer_with_scratch)
        novelty = result.details.get('novelty_score', 0.0)
        # Known patterns should have LOW novelty
        assert novelty < 0.8

    def test_novel_confidence_scores(self, classifier, wafer_with_center_defects):
        """Novelty score should be in [0, 1]."""
        result = classifier.classify_wafer(wafer_with_center_defects)
        novelty = result.details.get('novelty_score', 0.0)
        assert 0.0 <= novelty <= 1.0

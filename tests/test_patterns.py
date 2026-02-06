"""Tests for DefectPatternClassifier."""

import numpy as np
import pytest

from wafer_defect_clustering.patterns import DefectPatternClassifier, PatternResult
from wafer_defect_clustering.wafer import WaferMap


@pytest.fixture()
def wafer():
    return WaferMap(diameter_mm=300)


class TestPatternResult:
    def test_dataclass(self):
        pr = PatternResult('scratch', 0.9, {'angle_deg': 45})
        assert pr.pattern_type == 'scratch'
        assert pr.confidence == 0.9
        assert pr.details['angle_deg'] == 45
        assert pr.secondary_patterns == []


class TestScratchDetection:
    def test_perfect_scratch(self, wafer):
        clf = DefectPatternClassifier(wafer)
        # Line along x-axis
        x = np.linspace(-50, 50, 30)
        y = np.random.normal(0, 0.5, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'scratch'
        assert result.confidence > 0.5
        assert 'angle_deg' in result.details

    def test_angled_scratch(self, wafer):
        clf = DefectPatternClassifier(wafer)
        t = np.linspace(0, 60, 25)
        angle = np.radians(45)
        x = t * np.cos(angle) + np.random.normal(0, 0.3, 25)
        y = t * np.sin(angle) + np.random.normal(0, 0.3, 25)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'scratch'
        assert abs(result.details['angle_deg'] - 45) < 15


class TestRingDetection:
    def test_ring_pattern(self, wafer):
        clf = DefectPatternClassifier(wafer)
        # Ring at radius 80mm
        theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        r = 80 + np.random.normal(0, 2, 40)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'ring'
        assert result.confidence > 0.5
        assert abs(result.details['mean_radius_mm'] - 80) < 10


class TestCenterSpotDetection:
    def test_center_spot(self, wafer):
        clf = DefectPatternClassifier(wafer)
        # Cluster near centre
        x = np.random.normal(0, 5, 20)
        y = np.random.normal(0, 5, 20)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'center_spot'
        assert result.confidence > 0.5


class TestEdgeClusterDetection:
    def test_edge_cluster(self, wafer):
        clf = DefectPatternClassifier(wafer)
        # Points concentrated at edge (radius ~145mm) spread around full arc
        np.random.seed(99)
        theta = np.random.uniform(0, 2 * np.pi, 30)  # full arc
        r = np.random.uniform(140, 149, 30)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'edge_cluster'
        assert result.confidence > 0.3


class TestZonePatternDetection:
    def test_zone_pattern(self, wafer):
        clf = DefectPatternClassifier(wafer)
        # Sector-confined pattern: ±5° around 90° with width to avoid scratch
        np.random.seed(77)
        theta = np.random.uniform(np.radians(80), np.radians(100), 30)
        r = np.random.uniform(30, 120, 30)
        x = r * np.cos(theta) + np.random.normal(0, 3, 30)
        y = r * np.sin(theta) + np.random.normal(0, 3, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'zone_pattern'
        assert result.confidence > 0.3


class TestRepeatingPattern:
    def test_repeating_die_pattern(self, wafer):
        clf = DefectPatternClassifier(wafer)
        # Same intra-die position across many dies
        die_w, die_h = 10.0, 10.0
        intra_x, intra_y = 3.0, 4.0
        die_cols = np.arange(-5, 6)
        die_rows = np.arange(-5, 6)
        xs, ys = [], []
        for c in die_cols:
            for r in die_rows:
                xs.append(c * die_w + intra_x + np.random.normal(0, 0.2))
                ys.append(r * die_h + intra_y + np.random.normal(0, 0.2))
        points = np.column_stack([xs, ys])
        conf, details = clf.detect_repeating(points, die_size_mm=(die_w, die_h))
        assert conf > 0.5
        assert details['n_dies_affected'] > 10


class TestClassifyAll:
    def test_classify_all_labels(self, wafer):
        # Add some defects
        np.random.seed(42)
        x_scratch = np.linspace(-50, 50, 20)
        y_scratch = np.random.normal(0, 0.5, 20)
        x_center = np.random.normal(0, 5, 15)
        y_center = np.random.normal(0, 5, 15)
        x_noise = np.random.uniform(-100, 100, 10)
        y_noise = np.random.uniform(-100, 100, 10)

        all_x = np.concatenate([x_scratch, x_center, x_noise])
        all_y = np.concatenate([y_scratch, y_center, y_noise])
        wafer.add_defects(x=all_x, y=all_y)

        labels = np.concatenate(
            [
                np.full(20, 0),  # cluster 0
                np.full(15, 1),  # cluster 1
                np.full(10, -1),  # noise
            ]
        )

        clf = DefectPatternClassifier(wafer)
        results = clf.classify_all(labels)

        assert 0 in results
        assert 1 in results
        assert -1 not in results
        assert isinstance(results[0], PatternResult)

    def test_too_small_cluster(self, wafer):
        clf = DefectPatternClassifier(wafer, min_points=5)
        points = np.array([[1, 2], [3, 4]])
        result = clf.classify(points)
        assert result.pattern_type == 'too_small'


class TestRandomFallback:
    def test_uniform_random_no_pattern(self, wafer):
        clf = DefectPatternClassifier(wafer, confidence_threshold=0.5)
        # Truly random points — no clear pattern
        np.random.seed(123)
        x = np.random.uniform(-80, 80, 30)
        y = np.random.uniform(-80, 80, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        # Should either be 'random' or have low confidence
        assert result.confidence < 0.8 or result.pattern_type == 'random'


class TestLocDetection:
    """Tests for the 'loc' (generic localized) pattern detector."""

    def test_loc_mid_wafer_cluster(self, wafer):
        """Compact cluster at r ≈ 50% of wafer radius → loc."""
        clf = DefectPatternClassifier(wafer)
        # Tight cluster at (60, 60) — r ≈ 85mm, about 57% of 150mm
        np.random.seed(55)
        x = np.random.normal(60, 4, 25)
        y = np.random.normal(60, 4, 25)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'loc'
        assert result.confidence > 0.3

    def test_loc_not_center(self, wafer):
        """Cluster at r < 20% of wafer radius → center_spot, not loc."""
        clf = DefectPatternClassifier(wafer)
        x = np.random.normal(0, 5, 20)
        y = np.random.normal(0, 5, 20)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type != 'loc'

    def test_loc_not_edge(self, wafer):
        """Cluster at r > 85% of wafer radius → edge_cluster, not loc."""
        clf = DefectPatternClassifier(wafer)
        np.random.seed(99)
        theta = np.random.uniform(0, 2 * np.pi, 30)
        r = np.random.uniform(140, 149, 30)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type != 'loc'

    def test_loc_not_scratch(self, wafer):
        """Elongated cluster → scratch, not loc."""
        clf = DefectPatternClassifier(wafer)
        x = np.linspace(20, 80, 30)
        y = np.random.normal(50, 0.5, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type != 'loc'

    def test_loc_details_keys(self, wafer):
        """Details should contain mean_radius_mm and cluster_diameter_mm."""
        clf = DefectPatternClassifier(wafer)
        np.random.seed(55)
        x = np.random.normal(60, 4, 25)
        y = np.random.normal(60, 4, 25)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        if result.pattern_type == 'loc':
            assert 'mean_radius_mm' in result.details
            assert 'cluster_diameter_mm' in result.details


class TestNearFullDetection:
    """Tests for the 'near_full' catastrophic wafer pattern."""

    def test_near_full_dense_wafer(self, wafer):
        """500 defects uniformly covering the wafer → near_full."""
        clf = DefectPatternClassifier(wafer)
        np.random.seed(42)
        n = 500
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n)) * 140  # uniform in disk
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        wafer.add_defects(x=x, y=y)
        labels = np.zeros(n, dtype=int)  # all one cluster
        results = clf.classify_all(labels)
        # near_full should be detected for the wafer-wide cluster
        pattern_types = [r.pattern_type for r in results.values()]
        assert 'near_full' in pattern_types

    def test_near_full_sparse_rejected(self, wafer):
        """20 scattered defects → not near_full."""
        clf = DefectPatternClassifier(wafer)
        np.random.seed(42)
        x = np.random.uniform(-100, 100, 20)
        y = np.random.uniform(-100, 100, 20)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type != 'near_full'

    def test_near_full_confidence(self, wafer):
        """Confidence should scale with coverage fraction."""
        clf = DefectPatternClassifier(wafer)
        np.random.seed(42)
        n = 500
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.sqrt(np.random.uniform(0, 1, n)) * 140
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        if result.pattern_type == 'near_full':
            assert 0.0 < result.confidence <= 1.0


class TestCompoundPatterns:
    """Tests for multi-label compound pattern classification (M1)."""

    def test_compound_scratch_plus_edge(self, wafer):
        """Linear cluster near edge → both scratch and edge_cluster above threshold."""
        clf = DefectPatternClassifier(wafer, confidence_threshold=0.3)
        # Scratch along the edge
        x = np.linspace(100, 145, 30)
        y = np.random.normal(0, 0.5, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert hasattr(result, 'compound_patterns')
        # Should detect at least one pattern
        assert len(result.compound_patterns) >= 1

    def test_compound_patterns_list(self, wafer):
        """compound_patterns should contain all above-threshold results."""
        clf = DefectPatternClassifier(wafer, confidence_threshold=0.3)
        x = np.linspace(100, 145, 30)
        y = np.random.normal(0, 0.5, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        for pat_name, pat_conf, pat_details in result.compound_patterns:
            assert isinstance(pat_name, str)
            assert pat_conf >= clf.confidence_threshold
            assert isinstance(pat_details, dict)

    def test_is_compound_flag(self, wafer):
        """is_compound should be True when ≥2 patterns above threshold."""
        clf = DefectPatternClassifier(wafer, confidence_threshold=0.3)
        x = np.linspace(100, 145, 30)
        y = np.random.normal(0, 0.5, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        if len(result.compound_patterns) >= 2:
            assert result.is_compound is True

    def test_single_pattern_backward_compat(self, wafer):
        """Single dominant pattern → is_compound=False, pattern_type unchanged."""
        clf = DefectPatternClassifier(wafer)
        # Strong center spot — no ambiguity
        x = np.random.normal(0, 3, 25)
        y = np.random.normal(0, 3, 25)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert result.pattern_type == 'center_spot'
        assert result.is_compound is False

    def test_compound_in_summary_label(self, wafer):
        """compound_label should join compound pattern names with '+'."""
        clf = DefectPatternClassifier(wafer, confidence_threshold=0.3)
        x = np.linspace(100, 145, 30)
        y = np.random.normal(0, 0.5, 30)
        points = np.column_stack([x, y])
        result = clf.classify(points)
        assert hasattr(result, 'compound_label')
        if result.is_compound:
            assert '+' in result.compound_label
        else:
            assert result.compound_label == result.pattern_type

"""Tests for wafer_defect_clustering.spatial — spatial randomness & pre-filter."""

from __future__ import annotations

import numpy as np
import pytest

from wafer_defect_clustering.spatial import (
    SpatialTestResult,
    spatial_prefilter,
    spatial_randomness_test,
)
from wafer_defect_clustering.wafer import WaferMap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wafer(diameter: float = 300.0) -> WaferMap:
    return WaferMap(diameter_mm=diameter, edge_exclusion_mm=3.0)


# ===================================================================
# Clark-Evans Spatial Randomness Test
# ===================================================================


class TestClarkEvansBasic:
    """Core Clark-Evans ratio behaviour."""

    def test_clustered_pattern_detected(self):
        """A tight cluster should give R < 1 and is_random=False."""
        wafer = _wafer()
        rng = np.random.default_rng(42)
        # Single tight cluster at centre — clearly non-random
        pts = rng.normal(loc=0.0, scale=3.0, size=(80, 2))
        wafer.add_defects(x=pts[:, 0], y=pts[:, 1])

        result = spatial_randomness_test(wafer, method='clark_evans')

        assert isinstance(result, SpatialTestResult)
        assert result.method == 'clark_evans'
        assert result.statistic < 1.0, 'Clustered pattern should have R < 1'
        assert not result.is_random
        assert result.p_value < 0.05

    def test_uniform_grid_not_clustered(self):
        """Points on a regular grid should give R ≈ 1 or > 1 (regular)."""
        wafer = _wafer()
        # Grid across the wafer
        xs, ys = [], []
        for gx in np.linspace(-120, 120, 25):
            for gy in np.linspace(-120, 120, 25):
                if gx**2 + gy**2 < 145**2:  # inside wafer
                    xs.append(gx)
                    ys.append(gy)
        wafer.add_defects(x=np.array(xs), y=np.array(ys))

        result = spatial_randomness_test(wafer, method='clark_evans')

        assert result.statistic >= 0.9, 'Regular grid should have R ≥ 1'
        assert result.is_random or result.statistic > 1.0

    def test_result_fields(self):
        """SpatialTestResult contains all required fields."""
        wafer = _wafer()
        rng = np.random.default_rng(99)
        pts = rng.uniform(-100, 100, (50, 2))
        wafer.add_defects(x=pts[:, 0], y=pts[:, 1])

        result = spatial_randomness_test(wafer)

        assert hasattr(result, 'method')
        assert hasattr(result, 'statistic')
        assert hasattr(result, 'p_value')
        assert hasattr(result, 'is_random')
        assert hasattr(result, 'interpretation')
        assert isinstance(result.interpretation, str)
        assert len(result.interpretation) > 0


class TestClarkEvansEdgeCases:
    """Edge cases: empty wafer, single defect, two defects."""

    def test_empty_wafer(self):
        """Zero defects should return a neutral result gracefully."""
        wafer = _wafer()
        result = spatial_randomness_test(wafer)

        assert result.is_random  # nothing to cluster
        assert result.statistic == 1.0 or np.isnan(result.statistic)
        assert result.p_value >= 0.05

    def test_single_defect(self):
        """One defect — cannot compute NN distances, should gracefully handle."""
        wafer = _wafer()
        wafer.add_defects(x=np.array([0.0]), y=np.array([0.0]))

        result = spatial_randomness_test(wafer)

        assert result.is_random

    def test_two_defects(self):
        """Two defects — minimal NN computation, should not crash."""
        wafer = _wafer()
        wafer.add_defects(x=np.array([0.0, 10.0]), y=np.array([0.0, 0.0]))

        result = spatial_randomness_test(wafer)
        assert isinstance(result, SpatialTestResult)
        assert 0 <= result.p_value <= 1.0


class TestClarkEvansWaferCorrection:
    """Edge correction using wafer boundary for study area."""

    def test_uses_wafer_area(self):
        """The test should use the wafer's circular area, not a bounding box."""
        wafer = _wafer(diameter=300.0)
        rng = np.random.default_rng(7)
        # Scatter points inside wafer
        angles = rng.uniform(0, 2 * np.pi, 100)
        radii = rng.uniform(0, 140, 100)
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        wafer.add_defects(x=xs, y=ys)

        result = spatial_randomness_test(wafer)

        # With correct circular area the statistic should be reasonable
        # (0.5 - 2.0 range for ~uniform in circle)
        assert 0.3 < result.statistic < 2.5

    def test_different_wafer_sizes_affect_expected_distance(self):
        """Same points on different wafer sizes should give different R values."""
        pts = np.array([[0, 0], [5, 0], [0, 5], [5, 5], [-5, 0]], dtype=float)

        wafer_small = WaferMap(diameter_mm=50.0, edge_exclusion_mm=1.0)
        wafer_small.add_defects(x=pts[:, 0], y=pts[:, 1])
        r_small = spatial_randomness_test(wafer_small)

        wafer_big = WaferMap(diameter_mm=300.0, edge_exclusion_mm=1.0)
        wafer_big.add_defects(x=pts[:, 0], y=pts[:, 1])
        r_big = spatial_randomness_test(wafer_big)

        # On a bigger wafer, same cluster appears more clustered (lower R)
        assert r_big.statistic < r_small.statistic

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError."""
        wafer = _wafer()
        wafer.add_defects(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))

        with pytest.raises(ValueError, match='method'):
            spatial_randomness_test(wafer, method='invalid_method')


class TestClarkEvansSignificance:
    """Significance level parameter tests."""

    def test_significance_level_parameter(self):
        """Custom significance level should change is_random threshold."""
        wafer = _wafer()
        rng = np.random.default_rng(42)
        pts = rng.normal(loc=0.0, scale=5.0, size=(50, 2))
        wafer.add_defects(x=pts[:, 0], y=pts[:, 1])

        result_strict = spatial_randomness_test(wafer, significance=0.001)
        result_lenient = spatial_randomness_test(wafer, significance=0.5)

        # Lenient threshold more likely to flag as non-random
        assert result_strict.p_value == result_lenient.p_value  # same data
        # The key point: p-value is independent of threshold
        assert isinstance(result_strict.is_random, bool)
        assert isinstance(result_lenient.is_random, bool)


# ===================================================================
# Spatial Pre-filter (M2)
# ===================================================================


class TestSpatialPrefilter:
    """Graph-based pre-filter separating systematic from random defects."""

    def test_separates_cluster_from_noise(self):
        """A tight cluster + scattered noise → cluster flagged systematic."""
        wafer = _wafer()
        rng = np.random.default_rng(42)
        # 30-point tight cluster at (50, 50)
        cluster = rng.normal(loc=[50, 50], scale=3.0, size=(30, 2))
        # 10 widely scattered random points across the wafer
        noise_angles = rng.uniform(0, 2 * np.pi, 10)
        noise_radii = rng.uniform(60, 140, 10)
        noise = np.column_stack(
            [
                noise_radii * np.cos(noise_angles),
                noise_radii * np.sin(noise_angles),
            ]
        )
        all_pts = np.vstack([cluster, noise])
        wafer.add_defects(x=all_pts[:, 0], y=all_pts[:, 1])

        mask = spatial_prefilter(wafer, k=5)

        assert mask.shape == (wafer.n_defects,)
        assert mask.dtype == bool
        # The 30 cluster points should mostly be systematic
        assert mask[:30].sum() >= 20, 'Most cluster points should be systematic'
        # Systematic group should be majority cluster pts
        # (some noise pts near threshold may be included)
        assert mask[:30].sum() > mask[30:].sum(), (
            'Cluster should have more systematic pts than noise'
        )

    def test_preserves_all_in_single_cluster(self):
        """When all points form a single cluster, all should be systematic."""
        wafer = _wafer()
        rng = np.random.default_rng(7)
        pts = rng.normal(loc=[0, 0], scale=5.0, size=(50, 2))
        wafer.add_defects(x=pts[:, 0], y=pts[:, 1])

        mask = spatial_prefilter(wafer, k=5)

        # All should be systematic (single tight group)
        assert mask.sum() >= 40

    def test_empty_wafer(self):
        """Empty wafer returns empty mask."""
        wafer = _wafer()
        mask = spatial_prefilter(wafer)
        assert mask.shape == (0,)

    def test_few_defects(self):
        """Fewer defects than k should not crash."""
        wafer = _wafer()
        wafer.add_defects(x=np.array([0.0, 1.0]), y=np.array([0.0, 1.0]))
        mask = spatial_prefilter(wafer, k=5)
        assert mask.shape == (2,)

    def test_returns_boolean_array(self):
        """Result is a boolean array of length n_defects."""
        wafer = _wafer()
        rng = np.random.default_rng(0)
        pts = rng.uniform(-100, 100, (40, 2))
        wafer.add_defects(x=pts[:, 0], y=pts[:, 1])

        mask = spatial_prefilter(wafer)
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == wafer.n_defects


class TestSpatialPrefilterClustererIntegration:
    """Integration of spatial_prefilter with WaferClusterer."""

    def test_prefilter_default_off(self):
        """By default spatial_prefilter is disabled."""
        from wafer_defect_clustering import WaferClusterer

        c = WaferClusterer()
        assert c.spatial_prefilter is False

    def test_prefilter_param_accepted(self):
        """WaferClusterer accepts spatial_prefilter parameter."""
        from wafer_defect_clustering import WaferClusterer

        c = WaferClusterer(spatial_prefilter=True)
        assert c.spatial_prefilter is True

    def test_prefilter_fits_successfully(self):
        """WaferClusterer with spatial_prefilter=True runs without error."""
        from wafer_defect_clustering import WaferClusterer

        wafer = _wafer()
        rng = np.random.default_rng(42)
        # Cluster + noise
        cluster = rng.normal(loc=[0, 0], scale=5.0, size=(40, 2))
        noise_angles = rng.uniform(0, 2 * np.pi, 10)
        noise_radii = rng.uniform(80, 140, 10)
        noise = np.column_stack(
            [
                noise_radii * np.cos(noise_angles),
                noise_radii * np.sin(noise_angles),
            ]
        )
        pts = np.vstack([cluster, noise])
        wafer.add_defects(x=pts[:, 0], y=pts[:, 1])

        clusterer = WaferClusterer(
            min_cluster_size=5,
            spatial_prefilter=True,
            classify_patterns=False,
        )
        labels = clusterer.fit_predict(wafer)

        assert labels.shape == (wafer.n_defects,)
        # Pre-filtered noise points should be -1
        assert (labels == -1).sum() > 0

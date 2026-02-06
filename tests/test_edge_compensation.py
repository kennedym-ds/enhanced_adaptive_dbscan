"""Tests for edge density compensation."""

import numpy as np

from wafer_defect_clustering.edge_compensation import (
    apply_edge_compensation,
    compute_adaptive_bandwidth,
    compute_area_coverage_fraction,
    compute_edge_density_weights,
)
from wafer_defect_clustering.wafer import WaferMap


class TestAreaCoverageFraction:
    def test_fully_inside(self):
        # Point at centre, small neighbourhood → full coverage
        frac = compute_area_coverage_fraction(
            point_radius=0.0, neighbourhood_radius=10.0, wafer_radius=150.0
        )
        assert frac == 1.0

    def test_fully_inside_nonzero_radius(self):
        # Point at 100mm, neighbourhood 40mm, wafer 150mm → 100+40=140 < 150
        frac = compute_area_coverage_fraction(100.0, 40.0, 150.0)
        assert frac == 1.0

    def test_partial_overlap(self):
        # Point at 145mm, neighbourhood 10mm, wafer 150mm → 145+10=155 > 150
        frac = compute_area_coverage_fraction(145.0, 10.0, 150.0)
        assert 0.0 < frac < 1.0

    def test_outside_wafer(self):
        # Point at 200mm, wafer 150mm, neighbourhood 10mm → fully outside
        frac = compute_area_coverage_fraction(200.0, 10.0, 150.0)
        assert frac == 0.0

    def test_edge_point_half_coverage(self):
        # Point exactly on edge → ~50% coverage for small neighbourhood
        frac = compute_area_coverage_fraction(150.0, 10.0, 150.0)
        # Should be roughly 0.5 (not exact due to circle-circle geometry)
        assert 0.3 < frac < 0.7

    def test_monotonicity(self):
        # As point moves closer to edge, coverage decreases
        fracs = [
            compute_area_coverage_fraction(r, 10.0, 150.0) for r in [0, 50, 100, 130, 140, 145, 148]
        ]
        # Should be monotonically non-increasing
        for i in range(len(fracs) - 1):
            assert fracs[i] >= fracs[i + 1]


class TestEdgeDensityWeights:
    def test_empty_wafer(self):
        w = WaferMap()
        weights = compute_edge_density_weights(w)
        assert len(weights) == 0

    def test_center_weight_near_one(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0], y=[0])  # wafer centre
        weights = compute_edge_density_weights(w, bandwidth_mm=5.0)
        assert np.isclose(weights[0], 1.0)

    def test_edge_weight_greater_than_one(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[148], y=[0])  # 2mm from edge
        weights = compute_edge_density_weights(w, bandwidth_mm=5.0)
        assert weights[0] > 1.0

    def test_edge_weight_higher_than_center(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0, 148], y=[0, 0])
        weights = compute_edge_density_weights(w, bandwidth_mm=5.0)
        assert weights[1] > weights[0]


class TestApplyEdgeCompensation:
    def test_empty_wafer(self):
        w = WaferMap()
        D = apply_edge_compensation(w)
        assert D.shape == (0, 0)

    def test_distance_matrix_shape(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0, 50, 100], y=[0, 0, 0])
        D = apply_edge_compensation(w)
        assert D.shape == (3, 3)

    def test_symmetric(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0, 50, 100, 148], y=[0, 0, 0, 0])
        D = apply_edge_compensation(w)
        np.testing.assert_allclose(D, D.T, atol=1e-10)

    def test_diagonal_zero(self):
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0, 50], y=[0, 0])
        D = apply_edge_compensation(w)
        np.testing.assert_allclose(np.diag(D), 0.0)

    def test_edge_distances_shrink(self):
        """Edge compensation should make edge-point distances smaller
        relative to raw Euclidean, because edge weights > 1."""
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[146, 148], y=[0, 0])  # both near edge
        D_comp = apply_edge_compensation(w, bandwidth_mm=5.0)
        raw_dist = np.sqrt((148 - 146) ** 2)
        # Compensated distance should be less than raw (scaled down by weights > 1)
        assert D_comp[0, 1] < raw_dist

    def test_center_distances_unchanged(self):
        """At wafer centre, weights ≈ 1 → compensated ≈ raw."""
        w = WaferMap(diameter_mm=300)
        w.add_defects(x=[0, 5], y=[0, 0])  # both at centre
        D_comp = apply_edge_compensation(w, bandwidth_mm=5.0)
        raw_dist = 5.0
        # Should be very close to raw
        assert abs(D_comp[0, 1] - raw_dist) < 0.5


class TestVectorizedCoverage:
    """Tests that compute_area_coverage_fraction accepts array inputs."""

    def test_array_input(self):
        """Array of point_radius values should return array of fractions."""
        radii = np.array([0.0, 50.0, 100.0, 145.0, 160.0])
        fracs = compute_area_coverage_fraction(radii, 10.0, 150.0)
        assert isinstance(fracs, np.ndarray)
        assert fracs.shape == (5,)

    def test_scalar_still_works(self):
        """Scalar input should still return a float."""
        frac = compute_area_coverage_fraction(0.0, 10.0, 150.0)
        assert isinstance(frac, (float, np.floating))

    def test_vectorized_matches_loop(self):
        """Vectorized result must match the scalar loop exactly."""
        radii = np.linspace(0, 160, 200)
        # Vectorized call
        vec_result = compute_area_coverage_fraction(radii, 10.0, 150.0)
        # Scalar loop
        loop_result = np.array(
            [compute_area_coverage_fraction(float(r), 10.0, 150.0) for r in radii]
        )
        np.testing.assert_allclose(vec_result, loop_result, atol=1e-12)

    def test_boundary_cases_array(self):
        """Check edge cases: exactly on boundary, exactly at centre, fully outside."""
        radii = np.array([0.0, 140.0, 150.0, 200.0])
        fracs = compute_area_coverage_fraction(radii, 10.0, 150.0)
        assert fracs[0] == 1.0  # fully inside
        assert fracs[1] == 1.0  # 140 + 10 = 150, exactly touching
        assert 0.0 < fracs[2] < 1.0  # on edge, partial
        assert fracs[3] == 0.0  # fully outside

    def test_empty_array(self):
        """Empty array input should return empty array."""
        fracs = compute_area_coverage_fraction(np.array([]), 10.0, 150.0)
        assert isinstance(fracs, np.ndarray)
        assert len(fracs) == 0

    def test_weights_no_loop(self):
        """compute_edge_density_weights should use vectorized path (perf check).

        After vectorization, this function should handle 10k+ points quickly.
        """
        w = WaferMap(diameter_mm=300)
        rng = np.random.RandomState(42)
        n = 10_000
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0, 148, n)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        w.add_defects(x=x, y=y)

        import time

        start = time.perf_counter()
        weights = compute_edge_density_weights(w, bandwidth_mm=5.0)
        elapsed = time.perf_counter() - start

        assert weights.shape == (n,)
        # Should complete in well under 1 second with vectorization
        # (loop version takes ~2-5 seconds for 10k)
        assert elapsed < 1.0, f'Took {elapsed:.2f}s — vectorization may not be working'


# ===================================================================
# Adaptive Bandwidth (M3)
# ===================================================================


class TestAdaptiveBandwidth:
    """compute_adaptive_bandwidth and 'auto' mode."""

    def _make_wafer(self, n=100, seed=42):
        w = WaferMap(diameter_mm=300.0, edge_exclusion_mm=3.0)
        rng = np.random.default_rng(seed)
        angles = rng.uniform(0, 2 * np.pi, n)
        radii = rng.uniform(0, 140, n)
        w.add_defects(x=radii * np.cos(angles), y=radii * np.sin(angles))
        return w

    def test_returns_per_point_bandwidth(self):
        """compute_adaptive_bandwidth returns an array of length n_defects."""
        w = self._make_wafer()
        bw = compute_adaptive_bandwidth(w, k=5)
        assert bw.shape == (w.n_defects,)
        assert np.all(bw > 0), 'All bandwidths must be positive'

    def test_dense_region_smaller_bandwidth(self):
        """Points in a dense cluster should get smaller bandwidths."""
        w = WaferMap(diameter_mm=300.0, edge_exclusion_mm=3.0)
        rng = np.random.default_rng(42)
        # Dense cluster at centre
        dense = rng.normal(loc=[0, 0], scale=3.0, size=(50, 2))
        # Sparse points at edge
        sparse_angles = rng.uniform(0, 2 * np.pi, 20)
        sparse_r = rng.uniform(120, 140, 20)
        sparse = np.column_stack(
            [
                sparse_r * np.cos(sparse_angles),
                sparse_r * np.sin(sparse_angles),
            ]
        )
        pts = np.vstack([dense, sparse])
        w.add_defects(x=pts[:, 0], y=pts[:, 1])

        bw = compute_adaptive_bandwidth(w, k=5)

        mean_dense_bw = bw[:50].mean()
        mean_sparse_bw = bw[50:].mean()
        assert mean_dense_bw < mean_sparse_bw, (
            f'Dense bw ({mean_dense_bw:.2f}) should be < sparse bw ({mean_sparse_bw:.2f})'
        )

    def test_bandwidth_floor_applied(self):
        """Bandwidths should be floored at min_bandwidth."""
        w = WaferMap(diameter_mm=300.0, edge_exclusion_mm=3.0)
        # Very tight cluster — extreme small NN distances
        pts = np.array(
            [[0, 0], [0.01, 0], [0, 0.01], [0.01, 0.01], [-0.01, 0], [0, -0.01]], dtype=float
        )
        w.add_defects(x=pts[:, 0], y=pts[:, 1])

        bw = compute_adaptive_bandwidth(w, k=3, min_bandwidth=1.0)
        assert np.all(bw >= 1.0), 'Floor should be respected'

    def test_empty_wafer(self):
        """Empty wafer returns empty array."""
        w = WaferMap(diameter_mm=300.0)
        bw = compute_adaptive_bandwidth(w)
        assert bw.shape == (0,)

    def test_single_defect(self):
        """Single defect uses default bandwidth."""
        w = WaferMap(diameter_mm=300.0)
        w.add_defects(x=np.array([0.0]), y=np.array([0.0]))
        bw = compute_adaptive_bandwidth(w, k=5)
        assert bw.shape == (1,)
        assert bw[0] > 0


class TestAutoModeWeights:
    """compute_edge_density_weights with bandwidth_mm='auto'."""

    def test_auto_produces_valid_weights(self):
        """'auto' bandwidth produces weights ≥ 1.0."""
        w = WaferMap(diameter_mm=300.0, edge_exclusion_mm=3.0)
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * np.pi, 80)
        radii = rng.uniform(0, 145, 80)
        w.add_defects(x=radii * np.cos(angles), y=radii * np.sin(angles))

        weights = compute_edge_density_weights(w, bandwidth_mm='auto')
        assert weights.shape == (w.n_defects,)
        assert np.all(weights >= 1.0)

    def test_backward_compat_float(self):
        """Float bandwidth still works as before."""
        w = WaferMap(diameter_mm=300.0, edge_exclusion_mm=3.0)
        rng = np.random.default_rng(7)
        pts = rng.uniform(-100, 100, (30, 2))
        w.add_defects(x=pts[:, 0], y=pts[:, 1])

        weights = compute_edge_density_weights(w, bandwidth_mm=5.0)
        assert weights.shape == (w.n_defects,)
        assert np.all(weights >= 1.0)

    def test_auto_compensation_integration(self):
        """apply_edge_compensation with 'auto' produces valid distance matrix."""
        w = WaferMap(diameter_mm=300.0, edge_exclusion_mm=3.0)
        rng = np.random.default_rng(42)
        pts = rng.uniform(-100, 100, (30, 2))
        w.add_defects(x=pts[:, 0], y=pts[:, 1])

        D = apply_edge_compensation(w, bandwidth_mm='auto')
        assert D.shape == (30, 30)
        assert np.allclose(D, D.T), 'Distance matrix should be symmetric'
        assert np.all(D >= 0), 'All distances should be non-negative'

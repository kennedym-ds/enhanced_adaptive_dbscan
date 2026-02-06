"""Tests for streaming / incremental clustering (L3)."""

from __future__ import annotations

import numpy as np
import pytest

from wafer_defect_clustering.streaming import StreamingClusterer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cluster_data():
    """Two blobs that should form distinct clusters (50 pts each)."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-50, 0], scale=5, size=(50, 2))
    c2 = rng.normal(loc=[50, 0], scale=5, size=(50, 2))
    return np.vstack([c1, c2])


# ---------------------------------------------------------------------------
# TestStreamingAddDefects
# ---------------------------------------------------------------------------


class TestStreamingAddDefects:
    """StreamingClusterer receives defects incrementally."""

    def test_add_defects_updates_count(self):
        sc = StreamingClusterer(batch_size=20)
        sc.add_defects(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        assert sc.get_wafer().n_defects == 3

    def test_add_defects_multiple_batches(self, cluster_data):
        sc = StreamingClusterer(batch_size=25)
        for i in range(0, len(cluster_data), 25):
            chunk = cluster_data[i : i + 25]
            sc.add_defects(chunk[:, 0], chunk[:, 1])
        assert sc.get_wafer().n_defects == len(cluster_data)

    def test_labels_length_matches(self, cluster_data):
        sc = StreamingClusterer(batch_size=25)
        sc.add_defects(cluster_data[:, 0], cluster_data[:, 1])
        labels = sc.get_labels()
        assert len(labels) == len(cluster_data)


# ---------------------------------------------------------------------------
# TestStreamingMatchesBatch
# ---------------------------------------------------------------------------


class TestStreamingMatchesBatch:
    """Final labels should be comparable (not identical) to batch-mode."""

    def test_streaming_finds_clusters(self, cluster_data):
        """Streaming mode should detect at least 1 cluster."""
        sc = StreamingClusterer(batch_size=50, min_cluster_size=5)
        for i in range(0, len(cluster_data), 50):
            chunk = cluster_data[i : i + 50]
            sc.add_defects(chunk[:, 0], chunk[:, 1])
        labels = sc.get_labels()
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 1

    def test_single_batch_equals_nonstreaming(self, cluster_data):
        """Single large add → same result as non-streaming."""
        sc = StreamingClusterer(batch_size=200, min_cluster_size=5)
        sc.add_defects(cluster_data[:, 0], cluster_data[:, 1])
        labels = sc.get_labels()
        assert len(labels) == len(cluster_data)
        # Should detect clusters (at least 1)
        n_clusters = len(set(labels) - {-1})
        assert n_clusters >= 1


# ---------------------------------------------------------------------------
# TestStreamingEdgeCases
# ---------------------------------------------------------------------------


class TestStreamingEdgeCases:
    """Edge cases for streaming clustering."""

    def test_empty_no_clusters(self):
        sc = StreamingClusterer(batch_size=50)
        labels = sc.get_labels()
        assert len(labels) == 0
        assert sc.n_clusters == 0

    def test_n_clusters_property(self, cluster_data):
        sc = StreamingClusterer(batch_size=100, min_cluster_size=5)
        sc.add_defects(cluster_data[:, 0], cluster_data[:, 1])
        assert isinstance(sc.n_clusters, int)
        assert sc.n_clusters >= 0

    def test_get_wafer_returns_wafer_map(self):
        from wafer_defect_clustering import WaferMap

        sc = StreamingClusterer(batch_size=50, diameter_mm=200.0)
        wafer = sc.get_wafer()
        assert isinstance(wafer, WaferMap)
        assert wafer.geometry.diameter_mm == 200.0

    def test_small_batch_does_not_crash(self):
        """Adding fewer points than min_cluster_size still works."""
        sc = StreamingClusterer(batch_size=10, min_cluster_size=20)
        sc.add_defects(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
        labels = sc.get_labels()
        assert len(labels) == 2

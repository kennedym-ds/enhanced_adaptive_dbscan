"""Tests for visualization module (structural, not rendering)."""

import numpy as np
import pytest

from wafer_defect_clustering.patterns import PatternResult
from wafer_defect_clustering.visualization import (
    plot_cluster_details,
    plot_pattern_summary,
    plot_radial_distribution,
    plot_wafer_grid,
    plot_wafer_map,
)
from wafer_defect_clustering.wafer import WaferMap


@pytest.fixture()
def wafer_with_data():
    w = WaferMap(diameter_mm=300)
    np.random.seed(42)
    w.add_defects(x=np.random.uniform(-100, 100, 50), y=np.random.uniform(-100, 100, 50))
    return w


@pytest.fixture()
def labels_for_wafer():
    rng = np.random.RandomState(42)
    return rng.choice([0, 1, 2, -1], 50, p=[0.3, 0.3, 0.2, 0.2])


class TestPlotWaferMap:
    def test_no_labels(self, wafer_with_data):
        fig = plot_wafer_map(wafer_with_data)
        assert fig is not None
        assert len(fig.data) > 0  # at least wafer boundary + defects

    def test_with_labels(self, wafer_with_data, labels_for_wafer):
        fig = plot_wafer_map(wafer_with_data, labels_for_wafer)
        assert fig is not None

    def test_empty_wafer(self):
        w = WaferMap()
        fig = plot_wafer_map(w)
        assert fig is not None

    def test_show_zones(self, wafer_with_data):
        fig = plot_wafer_map(wafer_with_data, show_zones=True)
        assert fig is not None

    def test_show_dies(self, wafer_with_data):
        fig = plot_wafer_map(wafer_with_data, show_dies=True)
        assert fig is not None


class TestPlotWaferGrid:
    def test_grid(self, wafer_with_data, labels_for_wafer):
        wafers = [wafer_with_data] * 3
        labels = [labels_for_wafer] * 3
        fig = plot_wafer_grid(wafers, labels, ncols=2)
        assert fig is not None


class TestPlotPatternSummary:
    def test_with_results(self):
        results = {
            0: PatternResult('scratch', 0.85, {'angle_deg': 30}),
            1: PatternResult('ring', 0.72, {'mean_radius_mm': 80}),
        }
        fig = plot_pattern_summary(results)
        assert fig is not None
        assert len(fig.data) > 0

    def test_empty_results(self):
        fig = plot_pattern_summary({})
        assert fig is not None


class TestPlotRadialDistribution:
    def test_no_labels(self, wafer_with_data):
        fig = plot_radial_distribution(wafer_with_data)
        assert fig is not None

    def test_with_labels(self, wafer_with_data, labels_for_wafer):
        fig = plot_radial_distribution(wafer_with_data, labels_for_wafer)
        assert fig is not None


class TestPlotClusterDetails:
    def test_cluster_detail(self, wafer_with_data, labels_for_wafer):
        pr = PatternResult('scratch', 0.8, {'angle_deg': 45, 'length_mm': 50})
        fig = plot_cluster_details(
            wafer_with_data, labels_for_wafer, cluster_id=0, pattern_result=pr
        )
        assert fig is not None

    def test_cluster_detail_no_pattern(self, wafer_with_data, labels_for_wafer):
        fig = plot_cluster_details(wafer_with_data, labels_for_wafer, cluster_id=0)
        assert fig is not None

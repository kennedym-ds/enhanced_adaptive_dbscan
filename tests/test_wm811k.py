"""Tests for WM-811K benchmark utilities."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.wm811k_benchmark import (
    WM811K_PATTERN_NAMES,
    pixel_to_coordinates,
    run_benchmark,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_pixel_map():
    """A 26×26 binary pixel map with 4 defect pixels."""
    m = np.zeros((26, 26), dtype=np.uint8)
    m[5, 5] = 1
    m[10, 10] = 1
    m[20, 20] = 1
    m[13, 13] = 1
    return m


@pytest.fixture
def sample_wafer_maps():
    """Small synthetic WM-811K-style dataset: list of (pixel_map, label)."""
    rng = np.random.default_rng(42)
    maps = []
    labels = []
    for i in range(10):
        m = np.zeros((26, 26), dtype=np.uint8)
        # Sprinkle 5-15 defects randomly
        n_def = rng.integers(5, 16)
        rows = rng.integers(0, 26, size=n_def)
        cols = rng.integers(0, 26, size=n_def)
        m[rows, cols] = 1
        maps.append(m)
        labels.append(i % len(WM811K_PATTERN_NAMES))
    return maps, labels


# ---------------------------------------------------------------------------
# TestPixelToCoordinates
# ---------------------------------------------------------------------------


class TestPixelToCoordinates:
    """pixel_to_coordinates converts a pixel map to mm coordinates."""

    def test_output_shape(self, simple_pixel_map):
        coords = pixel_to_coordinates(simple_pixel_map, diameter_mm=300.0)
        assert coords.ndim == 2
        assert coords.shape[1] == 2
        assert coords.shape[0] == 4  # 4 nonzero pixels

    def test_coordinates_within_wafer(self, simple_pixel_map):
        coords = pixel_to_coordinates(simple_pixel_map, diameter_mm=300.0)
        radius = 150.0
        distances = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        assert np.all(distances <= radius * 1.1)  # small tolerance

    def test_centered_map_produces_centered_coords(self):
        """Single pixel at center → coords near (0, 0)."""
        m = np.zeros((26, 26), dtype=np.uint8)
        m[13, 13] = 1
        coords = pixel_to_coordinates(m, diameter_mm=300.0)
        assert abs(coords[0, 0]) < 10.0
        assert abs(coords[0, 1]) < 10.0

    def test_empty_map_returns_empty(self):
        m = np.zeros((26, 26), dtype=np.uint8)
        coords = pixel_to_coordinates(m, diameter_mm=300.0)
        assert coords.shape == (0, 2)

    def test_custom_diameter(self, simple_pixel_map):
        coords_300 = pixel_to_coordinates(simple_pixel_map, diameter_mm=300.0)
        coords_200 = pixel_to_coordinates(simple_pixel_map, diameter_mm=200.0)
        # Smaller diameter → tighter coordinates
        extent_300 = np.ptp(coords_300, axis=0).max()
        extent_200 = np.ptp(coords_200, axis=0).max()
        assert extent_200 < extent_300


# ---------------------------------------------------------------------------
# TestRunBenchmark
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    """run_benchmark executes clustering + classification on sample data."""

    def test_benchmark_runs_on_sample(self, sample_wafer_maps):
        maps, labels = sample_wafer_maps
        result = run_benchmark(maps, labels, diameter_mm=300.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_benchmark_metrics_format(self, sample_wafer_maps):
        maps, labels = sample_wafer_maps
        result = run_benchmark(maps, labels, diameter_mm=300.0)
        # Must have standard classification metric columns
        for col in ['pattern', 'precision', 'recall', 'f1', 'support']:
            assert col in result.columns, f'Missing column: {col}'

    def test_benchmark_empty_input(self):
        result = run_benchmark([], [], diameter_mm=300.0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_benchmark_skips_without_data(self, tmp_path):
        """When pointed at a nonexistent data path, returns empty + warning."""
        result = run_benchmark(
            wafer_maps=None,
            labels=None,
            diameter_mm=300.0,
            data_path=str(tmp_path / 'nonexistent'),
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# TestWM811KPatternNames
# ---------------------------------------------------------------------------


class TestWM811KPatternNames:
    def test_pattern_names_count(self):
        # WM-811K has 9 pattern types
        assert len(WM811K_PATTERN_NAMES) == 9

    def test_pattern_names_include_standard(self):
        # Key WM-811K pattern types
        expected = {'Center', 'Edge-Loc', 'Edge-Ring', 'Scratch', 'Random', 'Near-full', 'none'}
        assert expected.issubset(set(WM811K_PATTERN_NAMES))

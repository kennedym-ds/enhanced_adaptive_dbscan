"""Tests for multi-wafer lot analysis (L4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wafer_defect_clustering import WaferMap
from wafer_defect_clustering.lot_analysis import (
    ExcursionResult,
    LotAnalyzer,
    RecurringPattern,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_scratch_wafer(seed: int):
    """Create a wafer with a scratch-like pattern."""
    rng = np.random.default_rng(seed)
    wafer = WaferMap(diameter_mm=300.0)
    t = np.linspace(-80, 80, 30)
    x = t + rng.normal(0, 2, size=30)
    y = 0.5 * t + rng.normal(0, 2, size=30)
    wafer.add_defects(x=x, y=y)
    labels = np.zeros(30, dtype=int)  # all one cluster
    return wafer, labels


def _make_random_wafer(seed: int):
    """Wafer with random scattered defects (no clear pattern)."""
    rng = np.random.default_rng(seed)
    wafer = WaferMap(diameter_mm=300.0)
    n = rng.integers(10, 30)
    r = rng.uniform(0, 140, size=n)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    wafer.add_defects(x=x, y=y)
    labels = np.full(n, -1, dtype=int)  # all noise
    return wafer, labels


@pytest.fixture
def scratch_lot():
    """Lot of 5 wafers where all have a scratch pattern."""
    return [_make_scratch_wafer(i) for i in range(5)]


@pytest.fixture
def random_lot():
    """Lot of 5 wafers with random noise only."""
    return [_make_random_wafer(100 + i) for i in range(5)]


@pytest.fixture
def mixed_lot():
    """Lot with 3 scratch wafers and 2 random wafers."""
    return [_make_scratch_wafer(i) for i in range(3)] + [
        _make_random_wafer(100 + i) for i in range(2)
    ]


# ---------------------------------------------------------------------------
# TestLotAnalyzerPatternSummary
# ---------------------------------------------------------------------------


class TestLotAnalyzerPatternSummary:
    """pattern_summary() returns a DataFrame with expected columns."""

    def test_summary_dataframe_shape(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        df = analyzer.pattern_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # one row per wafer

    def test_summary_columns(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        df = analyzer.pattern_summary()
        for col in ['wafer_index', 'n_defects', 'n_clusters', 'dominant_pattern']:
            assert col in df.columns, f'Missing column: {col}'

    def test_empty_lot(self):
        analyzer = LotAnalyzer([], [])
        df = analyzer.pattern_summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# TestRecurringPatterns
# ---------------------------------------------------------------------------


class TestRecurringPatterns:
    """recurring_patterns() finds patterns shared across multiple wafers."""

    def test_scratch_lot_has_recurring(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        recurring = analyzer.recurring_patterns(min_wafers=3)
        assert len(recurring) >= 1
        assert all(isinstance(r, RecurringPattern) for r in recurring)

    def test_recurring_pattern_fields(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        recurring = analyzer.recurring_patterns(min_wafers=3)
        if recurring:
            r = recurring[0]
            assert hasattr(r, 'pattern_type')
            assert hasattr(r, 'wafer_indices')
            assert hasattr(r, 'consistency_score')
            assert 0.0 <= r.consistency_score <= 1.0

    def test_random_lot_no_recurring(self, random_lot):
        wafers, labels = zip(*random_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        recurring = analyzer.recurring_patterns(min_wafers=3)
        # All noise → no cluster patterns → no recurring
        assert len(recurring) == 0


# ---------------------------------------------------------------------------
# TestExcursionCheck
# ---------------------------------------------------------------------------


class TestExcursionCheck:
    """excursion_check() flags abnormal defect counts or patterns."""

    def test_excursion_returns_result(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        result = analyzer.excursion_check()
        assert isinstance(result, ExcursionResult)

    def test_excursion_above_baseline(self):
        """High defect count wafer triggers excursion."""
        wafers = []
        labels_list = []
        # Create 10 normal wafers with ~15 defects each
        for i in range(10):
            w, lb = _make_random_wafer(200 + i)
            wafers.append(w)
            labels_list.append(lb)
        # Add one outlier wafer with many defects
        rng = np.random.default_rng(999)
        big_wafer = WaferMap(diameter_mm=300.0)
        big_wafer.add_defects(
            x=rng.uniform(-100, 100, size=500),
            y=rng.uniform(-100, 100, size=500),
        )
        wafers.append(big_wafer)
        labels_list.append(np.full(500, -1, dtype=int))

        analyzer = LotAnalyzer(wafers, labels_list)
        result = analyzer.excursion_check()
        assert result.is_excursion

    def test_normal_lot_no_excursion(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        result = analyzer.excursion_check()
        # Similar wafers → no excursion
        assert isinstance(result.is_excursion, bool)


# ---------------------------------------------------------------------------
# TestLotOverviewPlot
# ---------------------------------------------------------------------------


class TestLotOverviewPlot:
    """plot_lot_overview() returns a Plotly figure."""

    def test_lot_overview_figure(self, scratch_lot):
        wafers, labels = zip(*scratch_lot)
        analyzer = LotAnalyzer(list(wafers), list(labels))
        fig = analyzer.plot_lot_overview()
        import plotly.graph_objects as go

        assert isinstance(fig, go.Figure)

"""Multi-wafer lot analysis (L4).

Provides :class:`LotAnalyzer` which takes a collection of clustered wafer
maps from a single lot and performs cross-wafer analysis:

1. **Pattern summary** — per-wafer defect / cluster / pattern overview.
2. **Recurring patterns** — patterns appearing on multiple wafers in the lot.
3. **Excursion detection** — flag wafers with abnormal defect counts or
   unusual pattern distributions compared to the lot baseline.
4. **Lot overview plot** — grid of wafer maps for visual inspection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .patterns import DefectPatternClassifier, PatternResult
from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = [
    'LotAnalyzer',
    'RecurringPattern',
    'ExcursionResult',
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RecurringPattern:
    """A defect pattern that recurs across multiple wafers in a lot.

    Attributes
    ----------
    pattern_type : str
        Pattern name (e.g. ``'scratch'``, ``'edge_cluster'``).
    wafer_indices : list of int
        Indices (0-based) of wafers exhibiting this pattern.
    consistency_score : float
        How consistent the pattern is across wafers (0–1).
    mean_location : tuple of float
        Mean (x, y) centroid of the pattern across wafers.
    """

    pattern_type: str
    wafer_indices: list[int] = field(default_factory=list)
    consistency_score: float = 0.0
    mean_location: tuple[float, float] = (0.0, 0.0)


@dataclass
class ExcursionResult:
    """Result of a lot-level excursion check.

    Attributes
    ----------
    is_excursion : bool
        ``True`` when one or more wafers deviate significantly from the
        lot baseline.
    flagged_wafers : list of int
        Indices of wafers that triggered the excursion flag.
    defect_counts : list of int
        Per-wafer defect counts.
    mean_defects : float
        Lot-level mean defect count.
    std_defects : float
        Lot-level standard deviation of defect counts.
    threshold : float
        Z-score threshold used for excursion detection.
    details : dict
        Additional metadata.
    """

    is_excursion: bool = False
    flagged_wafers: list[int] = field(default_factory=list)
    defect_counts: list[int] = field(default_factory=list)
    mean_defects: float = 0.0
    std_defects: float = 0.0
    threshold: float = 2.5
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LotAnalyzer
# ---------------------------------------------------------------------------


class LotAnalyzer:
    """Multi-wafer lot-level pattern analysis.

    Parameters
    ----------
    wafers : list of WaferMap
        Wafer maps in lot order (slot 1 … 25).
    labels : list of np.ndarray
        Cluster labels for each wafer (same order as *wafers*).
    """

    def __init__(
        self,
        wafers: list[WaferMap],
        labels: list[np.ndarray],
    ) -> None:
        if len(wafers) != len(labels):
            raise ValueError(
                f'wafers and labels must have same length (got {len(wafers)} vs {len(labels)})'
            )
        self.wafers = wafers
        self.labels = labels
        self._pattern_cache: list[dict[int, PatternResult]] | None = None

    # ------------------------------------------------------------------
    # Pattern classification (lazy, cached)
    # ------------------------------------------------------------------

    def _classify_all(self) -> list[dict[int, PatternResult]]:
        """Run geometric pattern classification on every wafer."""
        if self._pattern_cache is not None:
            return self._pattern_cache

        results: list[dict[int, PatternResult]] = []
        for wafer, lbls in zip(self.wafers, self.labels):
            if wafer.n_defects == 0 or len(set(lbls) - {-1}) == 0:
                results.append({})
                continue
            classifier = DefectPatternClassifier(wafer)
            results.append(classifier.classify_all(lbls))

        self._pattern_cache = results
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pattern_summary(self) -> pd.DataFrame:
        """Per-wafer summary with defect counts, clusters, and patterns.

        Returns
        -------
        pd.DataFrame
            Columns: ``wafer_index``, ``n_defects``, ``n_clusters``,
            ``dominant_pattern``.
        """
        if len(self.wafers) == 0:
            return pd.DataFrame(
                columns=['wafer_index', 'n_defects', 'n_clusters', 'dominant_pattern']
            )

        all_patterns = self._classify_all()
        rows = []
        for i, (wafer, lbls, pat_dict) in enumerate(zip(self.wafers, self.labels, all_patterns)):
            n_clusters = len(set(lbls) - {-1})
            dominant = 'none'
            if pat_dict:
                best = max(pat_dict.values(), key=lambda r: r.confidence)
                dominant = best.pattern_type
            rows.append(
                {
                    'wafer_index': i,
                    'n_defects': wafer.n_defects,
                    'n_clusters': n_clusters,
                    'dominant_pattern': dominant,
                }
            )
        return pd.DataFrame(rows)

    def recurring_patterns(self, min_wafers: int = 3) -> list[RecurringPattern]:
        """Identify patterns that appear on at least *min_wafers* wafers.

        Parameters
        ----------
        min_wafers : int
            Minimum number of wafers a pattern must appear on to be
            considered recurring.

        Returns
        -------
        list of RecurringPattern
        """
        all_patterns = self._classify_all()

        # Collect (pattern_type → list of wafer indices)
        pattern_wafers: dict[str, list[int]] = {}
        pattern_locations: dict[str, list[tuple[float, float]]] = {}

        for i, (wafer, pat_dict) in enumerate(zip(self.wafers, all_patterns)):
            for cid, pr in pat_dict.items():
                pt = pr.pattern_type
                if pt in ('none', 'too_small', 'random'):
                    continue
                pattern_wafers.setdefault(pt, []).append(i)

                # Compute cluster centroid
                coords = wafer.coordinates
                mask = self.labels[i] == cid
                if mask.any():
                    cx = float(coords[mask, 0].mean())
                    cy = float(coords[mask, 1].mean())
                    pattern_locations.setdefault(pt, []).append((cx, cy))

        result = []
        for pt, wafer_idxs in pattern_wafers.items():
            unique_wafers = sorted(set(wafer_idxs))
            if len(unique_wafers) < min_wafers:
                continue

            # Consistency: fraction of lot wafers exhibiting this pattern
            consistency = len(unique_wafers) / max(len(self.wafers), 1)

            # Mean location across wafers
            locs = pattern_locations.get(pt, [(0.0, 0.0)])
            mean_x = float(np.mean([l[0] for l in locs]))
            mean_y = float(np.mean([l[1] for l in locs]))

            result.append(
                RecurringPattern(
                    pattern_type=pt,
                    wafer_indices=unique_wafers,
                    consistency_score=round(consistency, 4),
                    mean_location=(round(mean_x, 2), round(mean_y, 2)),
                )
            )

        return result

    def excursion_check(
        self,
        threshold: float = 2.5,
    ) -> ExcursionResult:
        """Check for lot-level excursions based on defect count z-scores.

        Parameters
        ----------
        threshold : float
            Z-score threshold above which a wafer is flagged.

        Returns
        -------
        ExcursionResult
        """
        if len(self.wafers) == 0:
            return ExcursionResult()

        counts = [w.n_defects for w in self.wafers]
        mean_d = float(np.mean(counts))
        std_d = float(np.std(counts)) if len(counts) > 1 else 0.0

        flagged: list[int] = []
        if std_d > 0:
            for i, c in enumerate(counts):
                z = abs(c - mean_d) / std_d
                if z > threshold:
                    flagged.append(i)

        return ExcursionResult(
            is_excursion=len(flagged) > 0,
            flagged_wafers=flagged,
            defect_counts=counts,
            mean_defects=round(mean_d, 2),
            std_defects=round(std_d, 2),
            threshold=threshold,
        )

    def plot_lot_overview(self) -> go.Figure:
        """Render a grid of wafer maps for the lot.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        n = len(self.wafers)
        if n == 0:
            return go.Figure()

        # Layout: up to 5 columns
        ncols = min(n, 5)
        nrows = (n + ncols - 1) // ncols

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[f'Wafer {i}' for i in range(n)],
        )

        for idx, (wafer, lbls) in enumerate(zip(self.wafers, self.labels)):
            row = idx // ncols + 1
            col = idx % ncols + 1

            coords = wafer.coordinates
            if len(coords) == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=lbls,
                        colorscale='Viridis',
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        fig.update_layout(
            title='Lot Overview',
            height=250 * nrows,
            width=250 * ncols,
        )
        return fig

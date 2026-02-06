"""Streaming / incremental defect clustering (L3).

Provides :class:`StreamingClusterer` which accepts defects incrementally
(e.g. as they arrive from a wafer inspection tool scan) and maintains
evolving cluster assignments via micro-batch HDBSCAN.

Design
------
This is NOT a full dynamic HDBSCAN (Bubble-tree / insertion-based).
Instead it uses a practical micro-batch approach:

1. Accumulate incoming defects.
2. When the batch size is reached (or on explicit ``get_labels()``),
   re-cluster *all* accumulated points with HDBSCAN.
3. Return updated labels.

The main trade-off is simplicity vs. latency — each ``get_labels()``
call re-clusters the entire dataset.  For wafer-scale data (typically
< 50 000 defects) this is fast enough.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.cluster import HDBSCAN

from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = ['StreamingClusterer']


class StreamingClusterer:
    """Incremental micro-batch defect clustering.

    Parameters
    ----------
    batch_size : int
        Number of defects to accumulate before triggering an automatic
        re-cluster.  Re-clustering also happens on any ``get_labels()``
        call.
    diameter_mm : float
        Wafer diameter used for the internal :class:`WaferMap`.
    min_cluster_size : int
        Passed to ``sklearn.cluster.HDBSCAN``.
    **hdbscan_kwargs
        Additional keyword arguments forwarded to HDBSCAN.

    Examples
    --------
    >>> sc = StreamingClusterer(batch_size=100, min_cluster_size=5)
    >>> sc.add_defects(x_batch_1, y_batch_1)
    >>> sc.add_defects(x_batch_2, y_batch_2)
    >>> labels = sc.get_labels()
    """

    def __init__(
        self,
        batch_size: int = 100,
        diameter_mm: float = 300.0,
        min_cluster_size: int = 5,
        **hdbscan_kwargs: Any,
    ) -> None:
        self.batch_size = batch_size
        self.diameter_mm = diameter_mm
        self.min_cluster_size = min_cluster_size
        self._hdbscan_kwargs = hdbscan_kwargs

        self._wafer = WaferMap(diameter_mm=diameter_mm)
        self._labels: np.ndarray = np.array([], dtype=int)
        self._dirty = True  # labels need recomputation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_defects(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add a batch of defect coordinates.

        Parameters
        ----------
        x, y : array-like
            1-D arrays of defect coordinates in mm.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if len(x) != len(y):
            raise ValueError(f'x and y must have same length (got {len(x)} vs {len(y)})')
        if len(x) == 0:
            return

        self._wafer.add_defects(x=x, y=y)
        self._dirty = True

        # Auto-cluster when enough new points arrive
        if self._wafer.n_defects >= self.batch_size and self._dirty:
            self._recluster()

    def get_labels(self) -> np.ndarray:
        """Return current cluster labels for all accumulated defects.

        Returns
        -------
        np.ndarray
            Integer labels (``-1`` = noise).
        """
        if self._wafer.n_defects == 0:
            return np.array([], dtype=int)

        if self._dirty:
            self._recluster()

        return self._labels.copy()

    def get_wafer(self) -> WaferMap:
        """Return the internal :class:`WaferMap` with all accumulated defects."""
        return self._wafer

    @property
    def n_clusters(self) -> int:
        """Number of clusters found (excluding noise)."""
        if self._wafer.n_defects == 0:
            return 0
        if self._dirty:
            self._recluster()
        return len(set(self._labels) - {-1})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recluster(self) -> None:
        """Run HDBSCAN on all accumulated defects."""
        coords = self._wafer.coordinates
        n = len(coords)

        if n < self.min_cluster_size:
            # Not enough points for any cluster
            self._labels = np.full(n, -1, dtype=int)
            self._dirty = False
            return

        hdb = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            **self._hdbscan_kwargs,
        )
        self._labels = hdb.fit_predict(coords)
        self._dirty = False
        logger.debug(
            'Re-clustered %d defects → %d clusters',
            n,
            self.n_clusters,
        )

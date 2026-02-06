"""
Spatial statistics for semiconductor wafer defect analysis.

Provides spatial randomness testing via the Clark-Evans nearest-neighbour
ratio and spatial pre-filtering for separating systematic defect groups
from random particle contamination.

The Clark-Evans test determines whether a point pattern is
**clustered** (R < 1), **random** (R ≈ 1), or **regular** (R > 1)
by comparing observed mean nearest-neighbour distance to the expected
value under Complete Spatial Randomness (CSR) within the wafer's
circular domain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import KDTree
from scipy.stats import norm

from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = [
    'SpatialTestResult',
    'spatial_randomness_test',
    'spatial_prefilter',
]


@dataclass
class SpatialTestResult:
    """Result of a spatial randomness test.

    Attributes
    ----------
    method : str
        Test method name (e.g. ``'clark_evans'``).
    statistic : float
        Test statistic.  For Clark-Evans this is the R ratio:
        R < 1 → clustered, R ≈ 1 → random, R > 1 → regular.
    p_value : float
        Two-sided p-value under the null hypothesis of Complete Spatial
        Randomness (CSR).
    is_random : bool
        ``True`` if the pattern is consistent with randomness at the
        specified significance level.
    interpretation : str
        Human-readable description of the result.
    """

    method: str
    statistic: float
    p_value: float
    is_random: bool
    interpretation: str


def spatial_randomness_test(
    wafer: WaferMap,
    *,
    method: Literal['clark_evans'] = 'clark_evans',
    significance: float = 0.05,
) -> SpatialTestResult:
    """Test whether defects on a wafer follow Complete Spatial Randomness.

    Parameters
    ----------
    wafer : WaferMap
        Wafer with defect data.
    method : ``'clark_evans'``
        Statistical test to use.  Currently only Clark-Evans is supported.
    significance : float
        Significance level for the ``is_random`` determination.

    Returns
    -------
    SpatialTestResult
        Contains the test statistic, p-value, and interpretation.

    Raises
    ------
    ValueError
        If *method* is not a recognised test name.
    """
    if method != 'clark_evans':
        raise ValueError(
            f"Unknown spatial randomness method '{method}'. Supported methods: 'clark_evans'."
        )
    return _clark_evans_test(wafer, significance=significance)


# ------------------------------------------------------------------
# Clark-Evans implementation
# ------------------------------------------------------------------


def _clark_evans_test(
    wafer: WaferMap,
    *,
    significance: float = 0.05,
) -> SpatialTestResult:
    """Clark-Evans nearest-neighbour ratio test for CSR.

    The ratio R = r_observed / r_expected compares the observed mean
    nearest-neighbour distance to the expected value under a homogeneous
    Poisson process within the wafer's circular study area.

    Reference
    ---------
    Clark, P.J. & Evans, F.C. (1954). *Distance to nearest neighbor as a
    measure of spatial relationships in populations.* Ecology 35, 445–453.
    """
    n = wafer.n_defects

    # Edge cases -------------------------------------------------------
    if n <= 1:
        interp = 'Insufficient defects for spatial analysis.'
        return SpatialTestResult(
            method='clark_evans',
            statistic=1.0,
            p_value=1.0,
            is_random=True,
            interpretation=interp,
        )

    # Study area — use the wafer's full usable disc area
    wafer_radius = wafer.geometry.usable_radius_mm
    area = np.pi * wafer_radius**2

    # Observed mean nearest-neighbour distance -------------------------
    coords = wafer.coordinates  # (n, 2)
    tree = KDTree(coords)
    # k=2 because query includes the point itself as neighbour 0
    dists, _ = tree.query(coords, k=2)
    nn_dists = dists[:, 1]  # nearest neighbour distances
    r_observed = float(np.mean(nn_dists))

    # Expected mean NN distance under CSR in a circle ------------------
    # E(r) = 0.5 * sqrt(A / n)
    density = n / area
    r_expected = 0.5 * np.sqrt(area / n)

    # Clark-Evans ratio ------------------------------------------------
    R = r_observed / r_expected if r_expected > 0 else 1.0

    # Standard error and z-score (two-sided) ---------------------------
    # SE = 0.26136 / sqrt(n * density)
    se = 0.26136 / np.sqrt(n * density) if density > 0 else 1.0
    z = (r_observed - r_expected) / se if se > 0 else 0.0
    p_value = float(2.0 * norm.sf(abs(z)))  # two-sided

    # Interpretation ---------------------------------------------------
    is_random = p_value >= significance

    if R < 0.8:
        strength = 'strongly'
    elif R < 1.0:
        strength = 'moderately'
    else:
        strength = ''

    if is_random:
        interpretation = (
            f'Clark-Evans R = {R:.3f}, p = {p_value:.4f}. '
            f'Defect distribution is consistent with spatial randomness '
            f'(CSR) at α = {significance}.'
        )
    elif R < 1.0:
        interpretation = (
            f'Clark-Evans R = {R:.3f}, p = {p_value:.4f}. '
            f'Defects are {strength} clustered '
            f'(significantly non-random at α = {significance}).'
        )
    else:
        interpretation = (
            f'Clark-Evans R = {R:.3f}, p = {p_value:.4f}. '
            f'Defects are more regular than expected under CSR '
            f'(significantly non-random at α = {significance}).'
        )

    logger.debug(
        'Clark-Evans test: R=%.3f, z=%.3f, p=%.4f, n=%d, area=%.1f',
        R,
        z,
        p_value,
        n,
        area,
    )

    return SpatialTestResult(
        method='clark_evans',
        statistic=R,
        p_value=p_value,
        is_random=is_random,
        interpretation=interpretation,
    )


# ------------------------------------------------------------------
# Spatial pre-filter (graph-based)
# ------------------------------------------------------------------


def spatial_prefilter(
    wafer: WaferMap,
    *,
    k: int = 5,
    min_component_size: int = 3,
    edge_threshold_sigma: float = 1.5,
) -> np.ndarray:
    """Separate systematic defect groups from random particle noise.

    Builds a k-nearest-neighbour graph over defect coordinates, removes
    edges whose length exceeds ``mean + edge_threshold_sigma * std``, and
    marks connected components with ``≥ min_component_size`` nodes as
    **systematic** (``True``).

    Parameters
    ----------
    wafer : WaferMap
        Wafer with defect data.
    k : int
        Number of nearest neighbours for the graph.
    min_component_size : int
        Minimum component size to be considered systematic.
    edge_threshold_sigma : float
        Number of standard deviations above the mean edge length to
        use as the pruning threshold.

    Returns
    -------
    np.ndarray of shape ``(n_defects,)``
        Boolean mask — ``True`` for defects in systematic groups.
    """
    n = wafer.n_defects

    if n == 0:
        return np.empty(0, dtype=bool)

    if n < 2:
        return np.zeros(n, dtype=bool)

    coords = wafer.coordinates
    # Clamp k to at most n-1
    k_actual = min(k, n - 1)

    tree = KDTree(coords)
    dists, indices = tree.query(coords, k=k_actual + 1)  # +1 for self

    # Collect all edge lengths (excluding self-distance at column 0)
    edge_lengths = dists[:, 1:].ravel()
    mean_len = float(np.mean(edge_lengths))
    std_len = float(np.std(edge_lengths))
    threshold = mean_len + edge_threshold_sigma * std_len

    # Build adjacency via union-find for connected components
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    # Connect edges shorter than threshold
    for i in range(n):
        for j_idx in range(1, k_actual + 1):
            if dists[i, j_idx] <= threshold:
                union(i, indices[i, j_idx])

    # Count component sizes
    roots = np.array([find(i) for i in range(n)])
    unique_roots, counts = np.unique(roots, return_counts=True)
    root_to_size = dict(zip(unique_roots, counts))

    # Mark systematic: component size >= min_component_size
    mask = np.array(
        [root_to_size[find(i)] >= min_component_size for i in range(n)],
        dtype=bool,
    )

    n_systematic = int(mask.sum())
    logger.debug(
        'Spatial pre-filter: %d/%d defects in systematic groups (threshold=%.2f mm, k=%d)',
        n_systematic,
        n,
        threshold,
        k_actual,
    )

    return mask

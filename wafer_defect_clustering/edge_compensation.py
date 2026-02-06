"""
Edge density compensation for semiconductor wafer defect clustering.

Near wafer edges, the physical boundary truncates the local neighbourhood
available for density estimation.  A cluster at the edge has defects on only
one side, making it appear less dense than an equivalent cluster at the wafer
centre.  This module corrects for that geometric bias so that HDBSCAN treats
edge and centre clusters fairly.

The key insight: for a point at distance *d* from the edge, only a fraction
of its neighbourhood circle lies inside the wafer.  We compute that fraction
analytically (circle–circle intersection) and use its inverse as a density
weight.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = [
    'compute_adaptive_bandwidth',
    'compute_area_coverage_fraction',
    'compute_edge_density_weights',
    'apply_edge_compensation',
]


def compute_area_coverage_fraction(
    point_radius: float | np.ndarray,
    neighbourhood_radius: float,
    wafer_radius: float,
) -> float | np.ndarray:
    """Fraction of a circle that lies inside the wafer boundary.

    Given a point at distance *point_radius* from the wafer centre and a
    neighbourhood circle of *neighbourhood_radius*, compute the fraction of
    that neighbourhood circle that intersects the wafer disc.

    Uses the analytic circle–circle intersection area formula.  Accepts both
    scalar and array inputs for *point_radius* — array inputs are computed
    via vectorized NumPy operations without Python loops.

    Parameters
    ----------
    point_radius : float or np.ndarray
        Distance(s) of the point(s) from the wafer centre (mm).
    neighbourhood_radius : float
        Radius of the neighbourhood circle (mm).
    wafer_radius : float
        Radius of the wafer (mm).

    Returns
    -------
    float or np.ndarray
        Fraction(s) in ``(0, 1]``.  Returns 1.0 when the neighbourhood is
        entirely inside the wafer.  Returns the same type as the input:
        float for scalar, ndarray for array.
    """
    r = neighbourhood_radius
    R = wafer_radius
    d = np.asarray(point_radius, dtype=np.float64)
    scalar_input = d.ndim == 0

    if d.size == 0:
        return np.empty(0, dtype=np.float64)

    # Ensure at least 1-d for uniform treatment
    d = np.atleast_1d(d)

    full_area = np.pi * r**2
    result = np.ones_like(d)

    # Neighbourhood fully inside the wafer
    fully_inside = d + r <= R
    # Point is outside the wafer entirely
    fully_outside = d >= R + r
    # Partial overlap — circle–circle intersection
    partial = ~fully_inside & ~fully_outside

    result[fully_outside] = 0.0

    if np.any(partial):
        dp = d[partial]
        # Circle–circle intersection area
        # https://mathworld.wolfram.com/Circle-CircleIntersection.html
        part1 = r**2 * np.arccos((dp**2 + r**2 - R**2) / (2 * dp * r))
        part2 = R**2 * np.arccos((dp**2 + R**2 - r**2) / (2 * dp * R))
        part3 = 0.5 * np.sqrt((-dp + r + R) * (dp + r - R) * (dp - r + R) * (dp + r + R))
        intersection = part1 + part2 - part3
        result[partial] = np.clip(intersection / full_area, 0.0, 1.0)

    if scalar_input:
        return float(result[0])
    return result


def compute_adaptive_bandwidth(
    wafer: WaferMap,
    *,
    k: int = 5,
    min_bandwidth: float = 1.0,
) -> np.ndarray:
    """Compute per-point adaptive bandwidth based on k-th NN distance.

    For each defect point, the bandwidth is set to the distance to its
    k-th nearest neighbour, clamped to ``[min_bandwidth, 2 * wafer_radius]``.
    Dense regions get smaller bandwidths; sparse regions get larger ones.

    Parameters
    ----------
    wafer : WaferMap
        Wafer with defect data.
    k : int
        Neighbour index for bandwidth (1-based: k=5 → 5th nearest).
    min_bandwidth : float
        Floor for bandwidth values (mm).

    Returns
    -------
    np.ndarray of shape ``(n_defects,)``
        Per-point bandwidths (mm).
    """
    n = wafer.n_defects
    if n == 0:
        return np.empty(0, dtype=np.float64)

    coords = wafer.coordinates
    max_bw = 2.0 * wafer.geometry.radius_mm
    k_actual = min(k, n - 1)

    if k_actual < 1:
        # Only one point — use a default bandwidth
        return np.full(1, min(max_bw, max(min_bandwidth, 5.0)), dtype=np.float64)

    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=k_actual + 1)  # +1 for self at col 0
    # k-th NN distance (0-indexed column k_actual)
    kth_dists = dists[:, k_actual]

    bw = np.clip(kth_dists, min_bandwidth, max_bw)

    logger.debug(
        'Adaptive bandwidth: min=%.2f, median=%.2f, max=%.2f (k=%d)',
        float(bw.min()),
        float(np.median(bw)),
        float(bw.max()),
        k_actual,
    )
    return bw


def compute_edge_density_weights(
    wafer: WaferMap,
    bandwidth_mm: float | str = 5.0,
) -> np.ndarray:
    """Compute density compensation weights for each defect.

    Points near the wafer edge get weights > 1.0 (because their local
    neighbourhood is truncated).  Points near the centre get weight ≈ 1.0.

    Parameters
    ----------
    wafer : WaferMap
        Wafer with defect data.
    bandwidth_mm : float or ``'auto'``
        Neighbourhood radius used for density estimation (mm).  Should
        roughly match the scale at which HDBSCAN estimates density — a
        reasonable default is 5 mm for a 300 mm wafer.  When ``'auto'``,
        uses :func:`compute_adaptive_bandwidth` for per-point bandwidths.

    Returns
    -------
    np.ndarray of shape ``(n_defects,)``
        Multiplicative density weights (≥ 1.0).
    """
    if wafer.n_defects == 0:
        return np.empty(0, dtype=float)

    radii = wafer.radii
    wafer_radius = wafer.geometry.radius_mm

    if isinstance(bandwidth_mm, str) and bandwidth_mm == 'auto':
        bandwidths = compute_adaptive_bandwidth(wafer)
        # Per-point adaptive coverage: each point has its own neighbourhood radius
        fractions = np.array(
            [
                compute_area_coverage_fraction(r, bw, wafer_radius)
                for r, bw in zip(radii, bandwidths)
            ],
            dtype=float,
        )
    else:
        fractions = compute_area_coverage_fraction(radii, float(bandwidth_mm), wafer_radius)

    # Avoid division by zero; floor at 10 % coverage
    fractions = np.clip(fractions, 0.1, 1.0)

    # Weight = inverse of coverage fraction (full coverage → weight 1.0)
    weights = 1.0 / fractions
    return weights


def apply_edge_compensation(
    wafer: WaferMap,
    bandwidth_mm: float | str = 5.0,
    feature_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Build an edge-compensated pairwise distance matrix.

    Computes Euclidean distances between defects and scales them by the
    edge density weights.  The result can be passed directly to HDBSCAN
    with ``metric='precomputed'``.

    The scaling is symmetric: ``D_compensated[i, j] = D[i, j] / sqrt(w_i * w_j)``,
    which shrinks distances for edge points (making them appear closer /
    denser) and preserves distances for centre points.

    Parameters
    ----------
    wafer : WaferMap
        Wafer with defect data.
    bandwidth_mm : float or ``'auto'``
        Bandwidth for edge weight computation.  See
        :func:`compute_edge_density_weights`.
    feature_matrix : np.ndarray, optional
        If provided, use this ``(n, d)`` matrix instead of raw ``(x, y)``
        coordinates for distance computation (e.g. after feature encoding).

    Returns
    -------
    np.ndarray of shape ``(n_defects, n_defects)``
        Symmetric distance matrix suitable for ``metric='precomputed'``.
    """
    if wafer.n_defects == 0:
        return np.empty((0, 0), dtype=float)

    X = feature_matrix if feature_matrix is not None else wafer.coordinates
    weights = compute_edge_density_weights(wafer, bandwidth_mm=bandwidth_mm)

    # Pairwise Euclidean distances
    D = squareform(pdist(X, metric='euclidean'))

    # Symmetric scaling: closer distances for edge points
    w_outer = np.sqrt(np.outer(weights, weights))
    D_compensated = D / w_outer

    return D_compensated

"""
Automatic semiconductor defect pattern classification.

Semiconductor engineers manually classify wafer defect patterns by visual
inspection today.  This module automates that by analysing the geometry of
each defect cluster and matching it against known pattern signatures.

Supported pattern types
-----------------------
- **scratch** — Linear cluster (high PCA eccentricity, length/width > 5).
- **ring** — Annular cluster at a consistent radius from wafer centre.
- **center_spot** — Radially symmetric cluster concentrated near wafer centre.
- **edge_cluster** — Defects concentrated at the wafer periphery.
- **zone_pattern** — Defects confined to an angular sector.
- **repeating** — Periodic pattern matching die-grid spacing (reticle defect).
- **random** — No strong geometric signature (typically particle contamination).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = ['DefectPatternClassifier', 'PatternResult']


@dataclass
class PatternResult:
    """Result of a single cluster's pattern classification.

    Attributes
    ----------
    pattern_type : str
        Primary pattern name (``'scratch'``, ``'ring'``, etc.).
    confidence : float
        Confidence score in ``[0, 1]``.
    details : dict
        Pattern-specific metadata (e.g. scratch angle, ring radius).
    secondary_patterns : list of tuple
        Alternative classifications ranked by confidence:
        ``[(pattern_type, confidence, details), ...]``.
    compound_patterns : list of tuple
        All patterns scoring above the confidence threshold:
        ``[(pattern_type, confidence, details), ...]``.
        Populated when multi-label classification is active.
    is_compound : bool
        ``True`` when two or more patterns score above the
        confidence threshold simultaneously.
    compound_label : str
        Joined label of all compound patterns (e.g. ``'scratch+edge_cluster'``).
        Falls back to ``pattern_type`` when not compound.
    """

    pattern_type: str
    confidence: float
    details: dict[str, Any] = field(default_factory=dict)
    secondary_patterns: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)
    compound_patterns: list[tuple[str, float, dict[str, Any]]] = field(default_factory=list)
    is_compound: bool = False
    compound_label: str = ''


class DefectPatternClassifier:
    """Classify defect clusters into known semiconductor wafer defect patterns.

    Parameters
    ----------
    wafer : WaferMap
        The wafer geometry (needed for radius and centre reference).
    min_points : int
        Minimum points in a cluster to attempt classification.  Clusters
        smaller than this are labelled ``'too_small'``.
    confidence_threshold : float
        Minimum confidence to assign a pattern.  Below this, the cluster
        is labelled ``'random'``.

    Examples
    --------
    >>> clf = DefectPatternClassifier(wafer)
    >>> result = clf.classify(cluster_points)
    >>> result.pattern_type
    'scratch'
    """

    def __init__(
        self,
        wafer: WaferMap,
        min_points: int = 4,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.wafer = wafer
        self.min_points = min_points
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, cluster_points: np.ndarray) -> PatternResult:
        """Classify a single cluster.

        Parameters
        ----------
        cluster_points : np.ndarray, shape ``(n, 2)``
            ``[x, y]`` coordinates (mm from wafer centre) of the defects
            belonging to one cluster.

        Returns
        -------
        PatternResult
        """
        if len(cluster_points) < self.min_points:
            return PatternResult('too_small', 0.0, {'n_points': len(cluster_points)})

        # Run all detectors and rank
        detectors: list[tuple[str, Callable]] = [
            ('scratch', self._detect_scratch),
            ('ring', self._detect_ring),
            ('center_spot', self._detect_center_spot),
            ('edge_cluster', self._detect_edge_cluster),
            ('zone_pattern', self._detect_zone_pattern),
            ('loc', self._detect_loc),
            ('near_full', self._detect_near_full),
        ]

        scores: list[tuple[str, float, dict[str, Any]]] = []
        for name, func in detectors:
            try:
                conf, details = func(cluster_points)
                scores.append((name, conf, details))
            except Exception:
                logger.debug('Detector %s failed', name, exc_info=True)

        if not scores:
            return PatternResult('random', 0.0)

        # Sort by confidence descending
        scores.sort(key=lambda x: x[1], reverse=True)

        # Disambiguation: edge_cluster is more specific than ring when both score
        # high.  A ring at the wafer edge IS an edge cluster (and more actionable).
        score_dict = {s[0]: s for s in scores}
        if (
            'edge_cluster' in score_dict
            and 'ring' in score_dict
            and score_dict['edge_cluster'][1] > 0.4
            and score_dict['ring'][1] > 0.4
        ):
            # Promote edge_cluster to first position
            scores = [score_dict['edge_cluster']] + [s for s in scores if s[0] != 'edge_cluster']

        # Disambiguation: loc beats zone_pattern for spatially compact clusters.
        # A tight cluster at a specific angle triggers zone_pattern, but if the
        # cluster is small enough to be 'loc', that's more specific.
        score_dict = {s[0]: s for s in scores}
        if (
            'loc' in score_dict
            and 'zone_pattern' in score_dict
            and score_dict['loc'][1] > 0.3
            and score_dict['zone_pattern'][1] > 0.3
        ):
            loc_details = score_dict['loc'][2]
            if loc_details.get('diameter_fraction', 1.0) < 0.30:
                scores = [score_dict['loc']] + [s for s in scores if s[0] != 'loc']

        best_name, best_conf, best_details = scores[0]

        # Multi-label: collect all patterns above threshold
        above_threshold = [
            (name, conf, det) for name, conf, det in scores if conf >= self.confidence_threshold
        ]
        is_compound = len(above_threshold) >= 2
        if above_threshold:
            compound_label = '+'.join(name for name, _, _ in above_threshold)
        else:
            compound_label = best_name

        if best_conf < self.confidence_threshold:
            return PatternResult(
                'random',
                1.0 - best_conf,
                {},
                secondary_patterns=scores,
                compound_patterns=[],
                is_compound=False,
                compound_label='random',
            )

        return PatternResult(
            pattern_type=best_name,
            confidence=best_conf,
            details=best_details,
            secondary_patterns=scores[1:],
            compound_patterns=above_threshold,
            is_compound=is_compound,
            compound_label=compound_label,
        )

    def classify_all(self, labels: np.ndarray) -> dict[int, PatternResult]:
        """Classify every cluster produced by a clustering run.

        Also detects wafer-wide ``near_full`` patterns (catastrophic defect
        coverage).  A cluster that covers >60 %% of the wafer area with
        defects spanning ≥80 %% of the radial range is labelled ``near_full``.

        Parameters
        ----------
        labels : np.ndarray of shape ``(n_defects,)``
            Cluster labels (``-1`` = noise, ignored).

        Returns
        -------
        dict mapping cluster id → PatternResult
        """
        coords = self.wafer.coordinates
        results: dict[int, PatternResult] = {}

        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask = labels == cid
            results[cid] = self.classify(coords[mask])

        return results

    # ------------------------------------------------------------------
    # Individual pattern detectors
    # ------------------------------------------------------------------

    def _detect_scratch(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect linear scratch patterns using PCA eccentricity.

        A scratch is a thin, elongated cluster.  We fit PCA and check
        the ratio of the first to second explained variance.
        """
        if len(points) < 3:
            return 0.0, {}

        pca = PCA(n_components=2)
        pca.fit(points)
        var_ratio = pca.explained_variance_ratio_

        # Eccentricity = ratio of explained variance along first vs second axis
        eccentricity = var_ratio[0] / max(var_ratio[1], 1e-10)

        # Project onto first component to get length, second for width
        projected = pca.transform(points)
        length = np.ptp(projected[:, 0])
        width = np.ptp(projected[:, 1]) if projected.shape[1] > 1 else 1e-10
        aspect_ratio = length / max(width, 1e-10)

        # Scratch heuristic: high eccentricity AND high aspect ratio
        # eccentricity > 10 and aspect > 5 → very confident
        conf = 0.0
        if eccentricity > 3 and aspect_ratio > 3:
            conf = min(1.0, (eccentricity / 15) * 0.5 + (aspect_ratio / 10) * 0.5)

        angle_deg = float(np.degrees(np.arctan2(pca.components_[0, 1], pca.components_[0, 0])))

        return conf, {
            'eccentricity': float(eccentricity),
            'aspect_ratio': float(aspect_ratio),
            'angle_deg': angle_deg,
            'length_mm': float(length),
            'width_mm': float(width),
        }

    def _detect_ring(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect annular ring patterns using radial distribution.

        A ring has defects at a consistent radius from the wafer centre
        with angular spread.
        """
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        angles = np.arctan2(points[:, 1], points[:, 0])

        mean_r = float(np.mean(radii))
        std_r = float(np.std(radii))
        cv_r = std_r / max(mean_r, 1e-10)  # coefficient of variation

        # Angular spread: compute the circular range
        angular_range = _circular_range(angles)
        angular_span_deg = float(np.degrees(angular_range))

        # Ring heuristic: low radial CV (tight radius) + wide angular span
        conf = 0.0
        if cv_r < 0.20 and angular_span_deg > 120:
            radial_score = max(0, 1.0 - cv_r / 0.20)
            angular_score = min(1.0, angular_span_deg / 270)
            conf = radial_score * 0.6 + angular_score * 0.4

        return conf, {
            'mean_radius_mm': mean_r,
            'std_radius_mm': std_r,
            'cv_radius': cv_r,
            'angular_span_deg': angular_span_deg,
            'ring_width_mm': float(std_r * 2),
        }

    def _detect_center_spot(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect centre concentration using radial density."""
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        wafer_r = self.wafer.geometry.radius_mm

        mean_r = float(np.mean(radii))
        max_r = float(np.max(radii))
        frac_of_radius = mean_r / wafer_r

        # Centre spot: mean radius < 20 % of wafer radius, compact cluster
        conf = 0.0
        if frac_of_radius < 0.25:
            conf = max(0, 1.0 - frac_of_radius / 0.25)
            # Penalise if the cluster is very spread out
            spread = max_r / wafer_r
            if spread > 0.3:
                conf *= 0.5

        return conf, {
            'mean_radius_mm': mean_r,
            'max_radius_mm': max_r,
            'fraction_of_wafer_radius': frac_of_radius,
        }

    def _detect_edge_cluster(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect edge concentration using distance-to-edge statistics."""
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        wafer_r = self.wafer.geometry.radius_mm
        dist_to_edge = wafer_r - radii

        mean_dte = float(np.mean(dist_to_edge))
        edge_excl = self.wafer.geometry.edge_exclusion_mm

        # Edge cluster: most defects within 3× edge exclusion of the boundary
        threshold = max(edge_excl * 3, 10.0)
        frac_near_edge = float(np.mean(dist_to_edge < threshold))

        conf = 0.0
        if frac_near_edge > 0.6:
            # Proximity score: how close to edge (closer → higher)
            proximity = max(0, 1.0 - mean_dte / threshold)
            conf = frac_near_edge * 0.5 + proximity * 0.5
            conf = min(1.0, conf)

        return conf, {
            'mean_dist_to_edge_mm': mean_dte,
            'frac_near_edge': frac_near_edge,
            'edge_threshold_mm': threshold,
        }

    def _detect_zone_pattern(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect angular sector confinement."""
        angles = np.arctan2(points[:, 1], points[:, 0])

        # Circular standard deviation
        circ_std = _circular_std(angles)
        circ_std_deg = float(np.degrees(circ_std))

        # Zone = confined to a narrow angular sector
        conf = 0.0
        if circ_std_deg < 30:
            conf = max(0, 1.0 - circ_std_deg / 30)

        # Mean angle (circular mean)
        mean_angle = float(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))

        return conf, {
            'circular_std_deg': circ_std_deg,
            'mean_angle_deg': float(np.degrees(mean_angle)),
        }

    def detect_repeating(
        self,
        points: np.ndarray,
        die_size_mm: tuple[float, float] = (10.0, 10.0),
    ) -> tuple[float, dict[str, Any]]:
        """Detect die-periodic patterns using spatial autocorrelation.

        Repeating patterns (reticle / mask defects) appear at every die
        location on the wafer.

        Parameters
        ----------
        points : np.ndarray, shape ``(n, 2)``
        die_size_mm : tuple
            ``(width, height)`` of each die.

        Returns
        -------
        tuple of (confidence, details)
        """
        if len(points) < 6:
            return 0.0, {}

        dx, dy = die_size_mm

        # Compute positions within each die (modular arithmetic)
        intra_die_x = np.mod(points[:, 0], dx)
        intra_die_y = np.mod(points[:, 1], dy)

        # If repeating, intra-die positions cluster tightly
        std_x = float(np.std(intra_die_x))
        std_y = float(np.std(intra_die_y))

        # Normalised spread (0 = perfectly repeating, 1 = uniform)
        spread_x = std_x / (dx / np.sqrt(12))  # uniform std = range/sqrt(12)
        spread_y = std_y / (dy / np.sqrt(12))
        spread = (spread_x + spread_y) / 2

        conf = 0.0
        if spread < 0.3:
            conf = max(0, 1.0 - spread / 0.3)

        # Count how many unique dies are affected
        die_col = np.floor(points[:, 0] / dx).astype(int)
        die_row = np.floor(points[:, 1] / dy).astype(int)
        n_dies_affected = len(set(zip(die_row, die_col)))

        # Need to span multiple dies to be "repeating"
        if n_dies_affected < 3:
            conf *= 0.3

        return conf, {
            'intra_die_std_x': std_x,
            'intra_die_std_y': std_y,
            'normalised_spread': float(spread),
            'n_dies_affected': n_dies_affected,
        }

    def _detect_loc(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect generic localized cluster (WM-811K 'Loc' type).

        A spatially compact cluster in the mid-wafer region that doesn't
        match center_spot, edge_cluster, or scratch.

        Heuristic: low spatial extent (diameter < 15 %% of wafer radius),
        mean radius in ``[0.25, 0.80]`` of wafer radius, and no strong
        linearity (aspect ratio < 3).
        """
        wafer_r = self.wafer.geometry.radius_mm

        # Cluster centroid and spatial extent
        centroid = points.mean(axis=0)
        mean_r = float(np.sqrt(centroid[0] ** 2 + centroid[1] ** 2))
        frac_r = mean_r / wafer_r

        # Cluster diameter (max pairwise distance approximation)
        diffs = points - centroid
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        cluster_diameter = float(np.max(dists) * 2)
        diameter_frac = cluster_diameter / wafer_r

        # Aspect ratio via PCA to exclude scratch-like clusters
        if len(points) >= 3:
            pca = PCA(n_components=2)
            pca.fit(points)
            projected = pca.transform(points)
            length = np.ptp(projected[:, 0])
            width = max(np.ptp(projected[:, 1]), 1e-10)
            aspect_ratio = length / width
        else:
            aspect_ratio = 1.0

        conf = 0.0
        # Must be in mid-wafer region, compact, and not elongated
        if 0.25 <= frac_r <= 0.80 and diameter_frac < 0.30 and aspect_ratio < 3.0:
            # Compactness score
            compact_score = max(0, 1.0 - diameter_frac / 0.30)
            # Radial positioning score (peak at 0.5)
            radial_score = 1.0 - abs(frac_r - 0.525) / 0.275
            radial_score = max(0, radial_score)
            conf = compact_score * 0.6 + radial_score * 0.4
            # Penalise if somewhat elongated
            if aspect_ratio > 2.0:
                conf *= 0.5

        return conf, {
            'mean_radius_mm': mean_r,
            'fraction_of_wafer_radius': frac_r,
            'cluster_diameter_mm': cluster_diameter,
            'diameter_fraction': diameter_frac,
            'aspect_ratio': float(aspect_ratio),
        }

    def _detect_near_full(self, points: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Detect catastrophic near-full wafer defect coverage.

        A ``near_full`` pattern indicates defects covering most of the wafer
        area — typically >60 %% spatial coverage with defects spanning ≥80 %%
        of the radial range.
        """
        wafer_r = self.wafer.geometry.radius_mm
        n = len(points)

        # Minimum defect count guard
        if n < 50:
            return 0.0, {'reason': 'too_few_defects', 'n_defects': n}

        # Radial span
        radii = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        radial_span = float(np.ptp(radii) / wafer_r)

        # Octant-based coverage: divide wafer into 8 angular sectors × 3 radial
        # bins = 24 bins. Coverage = fraction of bins containing ≥1 defect.
        angles = np.arctan2(points[:, 1], points[:, 0])
        n_angular = 8
        n_radial = 3
        angle_bins = np.floor((angles + np.pi) / (2 * np.pi) * n_angular).astype(int)
        angle_bins = np.clip(angle_bins, 0, n_angular - 1)
        radial_bins = np.floor(radii / wafer_r * n_radial).astype(int)
        radial_bins = np.clip(radial_bins, 0, n_radial - 1)
        occupied = set(zip(angle_bins.tolist(), radial_bins.tolist()))
        coverage = len(occupied) / (n_angular * n_radial)

        conf = 0.0
        if coverage > 0.60 and radial_span > 0.80:
            coverage_score = min(1.0, (coverage - 0.60) / 0.35)
            radial_score = min(1.0, (radial_span - 0.80) / 0.20)
            conf = coverage_score * 0.6 + radial_score * 0.4

        return conf, {
            'n_defects': n,
            'coverage_fraction': coverage,
            'radial_span_fraction': radial_span,
            'occupied_bins': len(occupied),
            'total_bins': n_angular * n_radial,
        }


# ======================================================================
# Circular statistics helpers
# ======================================================================


def _circular_range(angles: np.ndarray) -> float:
    """Compute the smallest arc that contains all angles (radians)."""
    if len(angles) < 2:
        return 0.0
    sorted_a = np.sort(angles % (2 * np.pi))
    gaps = np.diff(sorted_a)
    gaps = np.append(gaps, sorted_a[0] + 2 * np.pi - sorted_a[-1])
    max_gap = np.max(gaps)
    return float(2 * np.pi - max_gap)


def _circular_std(angles: np.ndarray) -> float:
    """Compute circular standard deviation (radians)."""
    C = np.mean(np.cos(angles))
    S = np.mean(np.sin(angles))
    R = np.sqrt(C**2 + S**2)
    # Circular std: sqrt(-2 * ln(R))
    if R < 1e-10:
        return float(np.pi)  # maximum dispersion
    return float(np.sqrt(-2 * np.log(R)))

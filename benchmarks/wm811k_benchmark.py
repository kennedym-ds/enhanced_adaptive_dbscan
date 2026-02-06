"""WM-811K benchmark utilities for wafer defect clustering.

Provides helpers to:
- Convert WM-811K pixel maps to coordinate-level defect data
- Run WaferClusterer + DefectPatternClassifier on converted data
- Report per-class precision / recall / F1

WM-811K data is NOT bundled.  The benchmark gracefully skips when data is absent.

References
----------
Wu, M-J., Jang, J-S. R., & Chen, J-L. (2015).
    Wafer Map Failure Pattern Recognition and Similarity Ranking for
    Large-Scale Data Sets.  IEEE Transactions on Semiconductor Manufacturing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── WM-811K canonical pattern names (9 types) ────────────────────────────
WM811K_PATTERN_NAMES: list[str] = [
    'Center',
    'Donut',
    'Edge-Loc',
    'Edge-Ring',
    'Loc',
    'Near-full',
    'Random',
    'Scratch',
    'none',
]


# ── Pixel-to-coordinate conversion ───────────────────────────────────────


def pixel_to_coordinates(
    pixel_map: np.ndarray,
    diameter_mm: float = 300.0,
) -> np.ndarray:
    """Convert a binary pixel map to (x, y) coordinates in mm.

    The pixel map is mapped so that the grid center corresponds to the wafer
    center (0, 0) and the full grid spans the wafer diameter.

    Parameters
    ----------
    pixel_map : np.ndarray
        2-D binary array where nonzero entries indicate defect sites.
    diameter_mm : float
        Wafer diameter in mm (default 300).

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` array of ``[x, y]`` coordinates in mm, or
        ``(0, 2)`` if no defect pixels are present.
    """
    rows, cols = np.nonzero(pixel_map)
    if len(rows) == 0:
        return np.empty((0, 2), dtype=np.float64)

    h, w = pixel_map.shape
    radius = diameter_mm / 2.0

    # Map pixel indices to [-radius, +radius]
    x = (cols - (w - 1) / 2.0) / ((w - 1) / 2.0) * radius
    y = ((h - 1) / 2.0 - rows) / ((h - 1) / 2.0) * radius  # flip y

    return np.column_stack([x, y])


# ── Benchmark runner ─────────────────────────────────────────────────────


def run_benchmark(
    wafer_maps: Sequence[np.ndarray] | None = None,
    labels: Sequence[int] | None = None,
    *,
    diameter_mm: float = 300.0,
    data_path: str | None = None,
) -> pd.DataFrame:
    """Run the WM-811K benchmark and return per-class metrics.

    Parameters
    ----------
    wafer_maps : list of np.ndarray or None
        Pre-loaded binary pixel maps.  If *None*, the function attempts to
        load data from *data_path*.
    labels : list of int or None
        Ground-truth WM-811K label indices matching *wafer_maps*.
    diameter_mm : float
        Wafer diameter used for coordinate conversion.
    data_path : str or None
        Path to WM-811K dataset directory.  If the path doesn't exist and
        *wafer_maps* is None, returns an empty DataFrame (graceful skip).

    Returns
    -------
    pd.DataFrame
        Columns: ``pattern``, ``precision``, ``recall``, ``f1``, ``support``.
        Empty DataFrame when no data is available or input is empty.
    """
    empty = pd.DataFrame(columns=['pattern', 'precision', 'recall', 'f1', 'support'])

    # ── resolve data source ──────────────────────────────────────────
    if wafer_maps is None or labels is None:
        if data_path is not None and Path(data_path).exists():
            wafer_maps, labels = _load_wm811k(data_path)
        else:
            if data_path is not None:
                log.warning('WM-811K data not found at %s — skipping benchmark.', data_path)
            else:
                log.warning('No wafer maps provided and no data_path given — skipping.')
            return empty

    if len(wafer_maps) == 0:
        return empty

    # ── lazy import heavy deps ───────────────────────────────────────
    from wafer_defect_clustering import WaferClusterer, WaferMap
    from wafer_defect_clustering.patterns import DefectPatternClassifier

    predicted_patterns: list[str] = []

    for pmap in wafer_maps:
        coords = pixel_to_coordinates(pmap, diameter_mm=diameter_mm)
        if coords.shape[0] < 5:
            predicted_patterns.append('none')
            continue

        wafer = WaferMap(diameter_mm=diameter_mm)
        wafer.add_defects(x=coords[:, 0], y=coords[:, 1])

        try:
            clusterer = WaferClusterer(
                min_cluster_size=5,
                edge_compensation=False,
            )
            cluster_labels = clusterer.fit_predict(wafer)
        except Exception:
            predicted_patterns.append('none')
            continue

        # Classify predominant pattern across clusters
        try:
            classifier = DefectPatternClassifier()
            results = classifier.classify(wafer, cluster_labels)
            if results:
                # Pick the most confident pattern
                best = max(results.values(), key=lambda r: r.confidence)
                predicted_patterns.append(_map_to_wm811k(best.pattern))
            else:
                predicted_patterns.append('none')
        except Exception:
            predicted_patterns.append('none')

    # ── compute per-class metrics ────────────────────────────────────
    gt_names = [
        WM811K_PATTERN_NAMES[int(lb)] if 0 <= int(lb) < len(WM811K_PATTERN_NAMES) else 'none'
        for lb in labels
    ]

    rows = []
    unique_patterns = sorted(set(gt_names) | set(predicted_patterns))
    for pat in unique_patterns:
        tp = sum(1 for g, p in zip(gt_names, predicted_patterns) if g == pat and p == pat)
        fp = sum(1 for g, p in zip(gt_names, predicted_patterns) if g != pat and p == pat)
        fn = sum(1 for g, p in zip(gt_names, predicted_patterns) if g == pat and p != pat)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append(
            {
                'pattern': pat,
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'support': support,
            }
        )

    return pd.DataFrame(rows)


# ── Internal helpers ─────────────────────────────────────────────────────


def _map_to_wm811k(pattern_name: str) -> str:
    """Map internal pattern names to WM-811K canonical names."""
    mapping = {
        'center_spot': 'Center',
        'ring': 'Donut',
        'edge_cluster': 'Edge-Loc',
        'edge_ring': 'Edge-Ring',
        'zone_pattern': 'Loc',
        'random': 'Random',
        'scratch': 'Scratch',
        'repeating': 'none',
    }
    return mapping.get(pattern_name, 'none')


def _load_wm811k(data_path: str):
    """Attempt to load WM-811K pickle/npy from *data_path*.

    Returns (wafer_maps, labels) or raises FileNotFoundError.
    This is a placeholder — real WM-811K loading depends on the
    download format (typically a pickled dict or .npz).
    """
    import pickle

    p = Path(data_path)

    # Common WM-811K distribution format: LSWMD.pkl
    pkl = p / 'LSWMD.pkl'
    if pkl.exists():
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, pd.DataFrame):
            maps = data['waferMap'].tolist()
            labels = data['failureType'].tolist()
            return maps, labels

    # Fallback: .npz with keys 'maps', 'labels'
    npz = p / 'wm811k.npz'
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        return list(d['maps']), list(d['labels'])

    raise FileNotFoundError(f'No WM-811K data found at {data_path}')

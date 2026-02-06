"""Deep learning pattern classifier for wafer defect maps (optional).

Provides :class:`DeepPatternClassifier` which:

1. **Rasterizes** a point-cloud :class:`WaferMap` into a 2-D image
   (``(1, H, W)`` tensor).
2. **Classifies** each cluster (or full wafer) using either:
   - A pre-trained deep model (ViT-Tiny / MobileNet) via ``torch`` or
     ``onnxruntime`` (both optional).
   - A lightweight rule-based fallback when no model is loaded.
3. **Open-set detection** — returns a ``novelty_score`` in ``[0, 1]``
   indicating how unlike any known pattern the input looks.

Design
------
* ``torch`` and ``onnxruntime`` are **not** required at install time.
  Import is attempted lazily; if unavailable the fallback heuristic is used.
* Model weights are stored externally and loaded via ``model_path``.
* The rasteriser maps wafer coordinates to a square grid centred at (0, 0)
  spanning the wafer diameter.

References
----------
Frittoli, V. et al. (2022).  Deep open-set recognition for silicon wafer
production monitoring.  *Pattern Recognition*, 124, 108488.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .patterns import PatternResult
from .wafer import WaferMap

log = logging.getLogger(__name__)

# WM-811K canonical class names used by the deep model
_DEEP_PATTERN_LABELS: list[str] = [
    'center_spot',
    'donut',
    'edge_cluster',
    'edge_ring',
    'loc',
    'near_full',
    'random',
    'scratch',
    'none',
]


class DeepPatternClassifier:
    """Deep-learning wafer pattern classifier with open-set detection.

    Parameters
    ----------
    model_path : str or None
        Path to a pre-trained model (``.pt`` / ``.onnx``).  When *None*,
        a lightweight rule-based fallback is used (no DL required).
    device : str
        PyTorch device string (``'cpu'``, ``'cuda'``).  Ignored when
        ``model_path`` is *None* or when using ONNX runtime.
    novelty_threshold : float
        Maximum softmax probability below which a pattern is considered
        "novel" (open-set).  Default ``0.4``.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str = 'cpu',
        novelty_threshold: float = 0.4,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.novelty_threshold = novelty_threshold
        self._model: Any = None

        if model_path is not None:
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str) -> None:
        """Load a deep model from disk (torch or ONNX)."""
        if model_path.endswith('.onnx'):
            self._load_onnx(model_path)
        else:
            self._load_torch(model_path)

    def _load_torch(self, model_path: str) -> None:
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                'PyTorch is required for deep pattern classification. '
                'Install with: pip install torch'
            ) from exc
        self._model = torch.load(model_path, map_location=self.device, weights_only=False)
        self._model.eval()  # type: ignore[union-attr]
        self._backend = 'torch'

    def _load_onnx(self, model_path: str) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                'onnxruntime is required for ONNX model inference. '
                'Install with: pip install onnxruntime'
            ) from exc
        self._model = ort.InferenceSession(model_path)
        self._backend = 'onnx'

    # ------------------------------------------------------------------
    # Rasterisation
    # ------------------------------------------------------------------

    def rasterize(
        self,
        wafer: WaferMap,
        resolution: int = 96,
    ) -> np.ndarray:
        """Convert a point-cloud wafer map to a 2-D binary image.

        Parameters
        ----------
        wafer : WaferMap
            Wafer with defects populated via ``add_defects()``.
        resolution : int
            Output image size (square): ``(1, resolution, resolution)``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(1, resolution, resolution)`` with
            values in ``[0, 1]``.
        """
        img = np.zeros((1, resolution, resolution), dtype=np.float32)

        if wafer.n_defects == 0:
            return img

        coords = wafer.coordinates
        x = coords[:, 0].astype(np.float64)
        y = coords[:, 1].astype(np.float64)
        radius = wafer.geometry.diameter_mm / 2.0

        # Map (x, y) in [-radius, +radius] to pixel indices [0, resolution)
        col = ((x + radius) / (2.0 * radius) * (resolution - 1)).astype(int)
        row = ((radius - y) / (2.0 * radius) * (resolution - 1)).astype(int)  # flip y

        # Clip to bounds
        col = np.clip(col, 0, resolution - 1)
        row = np.clip(row, 0, resolution - 1)

        img[0, row, col] = 1.0
        return img

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(
        self,
        wafer: WaferMap,
        labels: np.ndarray,
    ) -> dict[int, PatternResult]:
        """Classify each cluster using the deep model or fallback.

        Parameters
        ----------
        wafer : WaferMap
            Wafer with defect data.
        labels : np.ndarray
            Cluster labels (``-1`` = noise).

        Returns
        -------
        dict
            Mapping ``cluster_id → PatternResult``.
        """
        if len(labels) == 0:
            return {}

        unique_labels = set(labels)
        unique_labels.discard(-1)

        results: dict[int, PatternResult] = {}
        for cid in sorted(unique_labels):
            mask = labels == cid
            if mask.sum() < 3:
                continue

            # Build per-cluster wafer for rasterisation
            cluster_wafer = WaferMap(diameter_mm=wafer.geometry.diameter_mm)
            coords = wafer.coordinates
            cluster_wafer.add_defects(
                x=coords[mask, 0],
                y=coords[mask, 1],
            )
            results[cid] = self._classify_single(cluster_wafer)

        return results

    def classify_wafer(self, wafer: WaferMap) -> PatternResult:
        """Classify the full wafer map (no pre-clustering).

        Parameters
        ----------
        wafer : WaferMap
            Wafer with defect data.

        Returns
        -------
        PatternResult
        """
        if wafer.n_defects == 0:
            return PatternResult(
                pattern_type='none',
                confidence=1.0,
                details={'novelty_score': 0.0},
            )
        return self._classify_single(wafer)

    # ------------------------------------------------------------------
    # Internal classification
    # ------------------------------------------------------------------

    def _classify_single(self, wafer: WaferMap) -> PatternResult:
        """Classify a single wafer (or cluster sub-wafer)."""
        img = self.rasterize(wafer, resolution=96)

        if self._model is not None:
            return self._classify_with_model(img)

        # Fallback: lightweight heuristic based on rasterised image
        return self._classify_fallback(wafer, img)

    def _classify_with_model(self, img: np.ndarray) -> PatternResult:
        """Run deep model inference and return PatternResult."""
        if self._backend == 'torch':
            return self._infer_torch(img)
        else:
            return self._infer_onnx(img)

    def _infer_torch(self, img: np.ndarray) -> PatternResult:
        import torch  # type: ignore[import-untyped]

        with torch.no_grad():
            tensor = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        return self._result_from_probs(probs)

    def _infer_onnx(self, img: np.ndarray) -> PatternResult:
        input_name = self._model.get_inputs()[0].name
        outputs = self._model.run(None, {input_name: img[np.newaxis].astype(np.float32)})
        logits = outputs[0][0]

        # Softmax
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()

        return self._result_from_probs(probs)

    def _result_from_probs(self, probs: np.ndarray) -> PatternResult:
        """Build PatternResult from probability vector, including novelty."""
        max_prob = float(probs.max())
        pred_idx = int(probs.argmax())
        pattern_name = (
            _DEEP_PATTERN_LABELS[pred_idx] if pred_idx < len(_DEEP_PATTERN_LABELS) else 'none'
        )
        novelty_score = 1.0 - max_prob

        # If max confidence is below threshold → novel
        if max_prob < self.novelty_threshold:
            pattern_name = 'novel'

        return PatternResult(
            pattern_type=pattern_name,
            confidence=max_prob,
            details={
                'novelty_score': novelty_score,
                'method': 'deep',
            },
        )

    # ------------------------------------------------------------------
    # Fallback heuristic (no model loaded)
    # ------------------------------------------------------------------

    def _classify_fallback(
        self,
        wafer: WaferMap,
        img: np.ndarray,
    ) -> PatternResult:
        """Lightweight rule-based classification from rasterised image.

        Uses spatial statistics on the image to approximate known patterns.
        This is intentionally simple — the deep model is the real classifier.
        """
        h, w = img.shape[1], img.shape[2]
        cx, cy = h // 2, w // 2

        total_pixels = float(img.sum())
        if total_pixels == 0:
            return PatternResult(
                pattern_type='none',
                confidence=1.0,
                details={'novelty_score': 0.0, 'method': 'fallback'},
            )

        # ── Spatial features ─────────────────────────────────────────
        # Center concentration
        r = max(h, w) // 6
        center_mask = np.zeros_like(img[0], dtype=bool)
        for ri in range(max(0, cx - r), min(h, cx + r + 1)):
            for ci in range(max(0, cy - r), min(w, cy + r + 1)):
                if (ri - cx) ** 2 + (ci - cy) ** 2 <= r**2:
                    center_mask[ri, ci] = True
        center_frac = float(img[0][center_mask].sum()) / total_pixels

        # Edge concentration
        edge_band = max(h, w) // 8
        edge_mask = np.ones_like(img[0], dtype=bool)
        edge_mask[edge_band:-edge_band, edge_band:-edge_band] = False
        edge_frac = float(img[0][edge_mask].sum()) / total_pixels

        # Aspect ratio (bounding box of defect pixels)
        rows, cols = np.nonzero(img[0])
        row_span = rows.max() - rows.min() + 1
        col_span = cols.max() - cols.min() + 1
        aspect_ratio = max(row_span, col_span) / max(min(row_span, col_span), 1)

        # Coverage (fraction of image pixels occupied)
        coverage = total_pixels / (h * w)

        # ── Decision logic ───────────────────────────────────────────
        # Compute scores for each pattern
        scores: dict[str, float] = {}

        # Scratch: elongated
        scores['scratch'] = min(1.0, aspect_ratio / 10.0) * 0.8 if aspect_ratio > 3.0 else 0.0

        # Center spot: concentrated in center
        scores['center_spot'] = center_frac if center_frac > 0.4 else center_frac * 0.5

        # Edge cluster: concentrated at edges
        scores['edge_cluster'] = edge_frac if edge_frac > 0.5 else edge_frac * 0.5

        # Near-full: high coverage
        scores['near_full'] = coverage * 2.0 if coverage > 0.15 else 0.0

        # Random: low scores everywhere
        scores['random'] = 0.3 if max(scores.values(), default=0) < 0.35 else 0.1

        # None fallback
        scores['none'] = 0.1

        # ── Pick best ────────────────────────────────────────────────
        best_pattern = max(scores, key=lambda k: scores[k])
        best_score = scores[best_pattern]
        max_possible = max(scores.values())

        # Novelty: if no pattern is strongly matched
        novelty_score = max(0.0, 1.0 - max_possible)

        return PatternResult(
            pattern_type=best_pattern,
            confidence=min(1.0, best_score),
            details={
                'novelty_score': round(novelty_score, 4),
                'method': 'fallback',
                'scores': {k: round(v, 4) for k, v in scores.items()},
                'center_frac': round(center_frac, 4),
                'edge_frac': round(edge_frac, 4),
                'aspect_ratio': round(aspect_ratio, 2),
                'coverage': round(coverage, 6),
            },
        )

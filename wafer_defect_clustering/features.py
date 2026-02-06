"""
Defect feature encoding for semiconductor wafer defect clustering.

Semiconductor defects have heterogeneous attributes — continuous (size, kill
ratio), categorical (layer, classification code), and spatial (x, y).  This
module encodes them into a weighted feature vector suitable for distance-based
clustering with HDBSCAN.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = ['DefectFeatureEncoder']


class DefectFeatureEncoder:
    """Encode defect attributes into a weighted feature matrix for clustering.

    Spatial features ``(x, y)`` are always included.  Additional defect
    attributes are optionally encoded and weighted relative to spatial
    distance so that the clustering algorithm can jointly consider location
    and defect characteristics.

    Parameters
    ----------
    spatial_weight : float
        Weight applied to spatial coordinates (default 1.0).
    size_weight : float
        Weight for defect size feature.  Size is log-transformed before
        scaling because semiconductor defect sizes span orders of magnitude
        (0.1 µm to 100 µm).
    severity_weight : float
        Weight for kill / severity flag.
    layer_weight : float
        Weight for process-layer encoding.
    classcode_weight : float
        Weight for defect classification code.
    normalize : bool
        If ``True`` (default), each feature column is z-score normalised
        before weighting.

    Examples
    --------
    >>> encoder = DefectFeatureEncoder(size_weight=0.5, severity_weight=0.3)
    >>> X = encoder.fit_transform(wafer)
    >>> X.shape  # (n_defects, n_features)
    """

    def __init__(
        self,
        spatial_weight: float = 1.0,
        size_weight: float = 0.5,
        severity_weight: float = 0.3,
        layer_weight: float = 0.2,
        classcode_weight: float = 0.2,
        polar_weight: float = 0.0,
        normalize: bool = True,
    ) -> None:
        self.spatial_weight = spatial_weight
        self.size_weight = size_weight
        self.severity_weight = severity_weight
        self.layer_weight = layer_weight
        self.classcode_weight = classcode_weight
        self.polar_weight = polar_weight
        self.normalize = normalize

        # Fitted state
        self._scaler: StandardScaler | None = None
        self._ordinal_encoders: dict[str, OrdinalEncoder] = {}
        self._feature_names: list[str] = []
        self._weights: np.ndarray | None = None
        self._fitted: bool = False

    @property
    def feature_names(self) -> list[str]:
        """Names of features in the output matrix (after fitting)."""
        return list(self._feature_names)

    # ------------------------------------------------------------------
    # Building the raw feature matrix
    # ------------------------------------------------------------------

    def _build_raw(self, wafer: WaferMap, fit: bool) -> np.ndarray:
        """Collect all requested features into a single matrix."""
        columns = []
        names = []
        weights = []

        # Always include spatial coordinates
        coords = wafer.coordinates
        columns.append(coords)
        names.extend(['x', 'y'])
        weights.extend([self.spatial_weight, self.spatial_weight])

        # Optional polar coordinates (r, sin(θ), cos(θ))
        if self.polar_weight > 0:
            r = wafer.radii.reshape(-1, 1)
            theta = wafer.angles
            sin_theta = np.sin(theta).reshape(-1, 1)
            cos_theta = np.cos(theta).reshape(-1, 1)
            columns.extend([r, sin_theta, cos_theta])
            names.extend(['r', 'sin_theta', 'cos_theta'])
            weights.extend([self.polar_weight] * 3)

        # Optional attributes — only include if present on the wafer
        attr_map = {
            'size': self.size_weight,
            'kill': self.severity_weight,
            'layer': self.layer_weight,
            'classcode': self.classcode_weight,
        }

        for attr_name, weight in attr_map.items():
            if weight <= 0 or attr_name not in wafer.attribute_names:
                continue

            raw = wafer.get_attribute(attr_name)

            # Numerical vs categorical handling
            if attr_name == 'size':
                # Log-transform sizes (handles the 0.1–100 µm range)
                val = np.log1p(raw.astype(float)).reshape(-1, 1)
            elif attr_name == 'kill':
                val = raw.astype(float).reshape(-1, 1)
            elif attr_name in ('layer', 'classcode'):
                # Ordinal encode categoricals
                raw_2d = raw.reshape(-1, 1).astype(str)
                if fit:
                    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    val = enc.fit_transform(raw_2d)
                    self._ordinal_encoders[attr_name] = enc
                else:
                    enc = self._ordinal_encoders.get(attr_name)
                    if enc is None:
                        continue
                    val = enc.transform(raw_2d)
            else:
                val = raw.astype(float).reshape(-1, 1)

            columns.append(val)
            names.append(attr_name)
            weights.append(weight)

        raw_matrix = np.hstack(columns)
        self._feature_names = names
        self._weights = np.array(weights)
        return raw_matrix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, wafer: WaferMap) -> np.ndarray:
        """Fit the encoder on wafer data and return the feature matrix.

        Parameters
        ----------
        wafer : WaferMap
            Wafer with defect data and attributes.

        Returns
        -------
        np.ndarray of shape ``(n_defects, n_features)``
            Weighted, normalised feature matrix.
        """
        raw = self._build_raw(wafer, fit=True)

        if self.normalize:
            self._scaler = StandardScaler()
            scaled = self._scaler.fit_transform(raw)
        else:
            scaled = raw.copy()

        # Apply per-feature weights
        weighted = scaled * self._weights[np.newaxis, :]
        self._fitted = True
        return weighted

    def transform(self, wafer: WaferMap) -> np.ndarray:
        """Transform new wafer data using a previously fitted encoder.

        Parameters
        ----------
        wafer : WaferMap
            New wafer data.

        Returns
        -------
        np.ndarray
            Feature matrix with the same encoding as ``fit_transform``.

        Raises
        ------
        RuntimeError
            If the encoder has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError('Encoder has not been fitted. Call fit_transform() first.')

        raw = self._build_raw(wafer, fit=False)

        if self.normalize and self._scaler is not None:
            scaled = self._scaler.transform(raw)
        else:
            scaled = raw.copy()

        weighted = scaled * self._weights[np.newaxis, :]
        return weighted

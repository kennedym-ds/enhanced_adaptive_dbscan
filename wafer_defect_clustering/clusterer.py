"""
WaferClusterer — HDBSCAN-based defect clustering with wafer geometry awareness.

This is the main user-facing class.  It wraps HDBSCAN with semiconductor-
specific preprocessing (edge density compensation, defect feature encoding)
and post-processing (automatic defect pattern classification).

Fully sklearn-compatible: inherits ``BaseEstimator`` and ``ClusterMixin``,
exposes ``fit()``, ``fit_predict()``, ``predict()``, and ``labels_``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_samples
from sklearn.utils.validation import check_is_fitted

from .edge_compensation import apply_edge_compensation
from .features import DefectFeatureEncoder
from .patterns import DefectPatternClassifier
from .spatial import spatial_prefilter as _spatial_prefilter
from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = ['WaferClusterer']


class WaferClusterer(BaseEstimator, ClusterMixin):
    """HDBSCAN-based defect clustering with wafer geometry awareness.

    Wraps `hdbscan.HDBSCAN <https://hdbscan.readthedocs.io/>`_ with three
    layers of semiconductor domain logic:

    1. **Edge density compensation** — corrects for geometric bias at wafer
       boundaries using circle–circle intersection geometry.
    2. **Defect attribute encoding** — jointly clusters on location *and*
       defect characteristics (size, severity, layer, classcode).
    3. **Automatic pattern classification** — identifies scratches, rings,
       edge clusters, centre spots, zone patterns, and repeating patterns.

    Parameters
    ----------
    min_cluster_size : int
        Minimum defects to form a cluster (passed to HDBSCAN).
    min_samples : int or None
        Core-point neighbourhood size (passed to HDBSCAN).  ``None`` means
        HDBSCAN uses ``min_cluster_size``.
    edge_compensation : bool
        Whether to apply edge density compensation.
    compensation_bandwidth_mm : float
        Bandwidth for edge compensation (mm).
    feature_encoder : DefectFeatureEncoder or None
        Custom feature encoder.  ``None`` means spatial-only clustering.
    classify_patterns : bool
        Whether to run pattern classification after clustering.
    cluster_selection_method : str
        HDBSCAN cluster selection method: ``'eom'`` (excess of mass, default)
        or ``'leaf'`` (finer clusters).
    cluster_selection_epsilon : float
        Distance threshold for merging clusters (HDBSCAN parameter).
    allow_single_cluster : bool
        If ``True``, allows HDBSCAN to find a single cluster.

    Examples
    --------
    >>> from wafer_defect_clustering import WaferMap, WaferClusterer
    >>> wafer = WaferMap(diameter_mm=300)
    >>> wafer.add_defects(x=x_data, y=y_data, size=sizes)
    >>> clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=True)
    >>> labels = clusterer.fit_predict(wafer)
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        edge_compensation: bool = True,
        compensation_bandwidth_mm: float = 5.0,
        feature_encoder: DefectFeatureEncoder | None = None,
        classify_patterns: bool = True,
        cluster_selection_method: str = 'eom',
        cluster_selection_epsilon: float = 0.0,
        allow_single_cluster: bool = False,
        spatial_prefilter: bool = False,
        harvesting: bool = False,
        harvesting_max_iterations: int = 10,
        harvesting_min_silhouette: float = 0.3,
        pattern_classifier: str = 'geometric',
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.edge_compensation = edge_compensation
        self.compensation_bandwidth_mm = compensation_bandwidth_mm
        self.feature_encoder = feature_encoder
        self.classify_patterns = classify_patterns
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.allow_single_cluster = allow_single_cluster
        self.spatial_prefilter = spatial_prefilter
        self.harvesting = harvesting
        self.harvesting_max_iterations = harvesting_max_iterations
        self.harvesting_min_silhouette = harvesting_min_silhouette
        self.pattern_classifier = pattern_classifier

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: Any = None, y: Any = None, *, wafer: WaferMap | None = None) -> WaferClusterer:
        """Cluster defects on a wafer.

        Parameters
        ----------
        X : WaferMap, np.ndarray, or None
            If a ``WaferMap``, uses its coordinate (and attribute) data.
            If an ndarray of shape ``(n, 2)``, treats it as raw ``[x, y]``
            coordinates on a default 300 mm wafer.
            If ``None``, the *wafer* keyword argument must be provided.
        y : ignored
            Present for sklearn API compatibility.
        wafer : WaferMap, optional
            Explicit wafer object (alternative to passing as *X*).

        Returns
        -------
        self
        """
        wafer_obj = self._resolve_wafer(X, wafer)
        self.wafer_ = wafer_obj

        if wafer_obj.n_defects < self.min_cluster_size:
            self.labels_ = np.full(wafer_obj.n_defects, -1, dtype=int)
            self.probabilities_ = np.zeros(wafer_obj.n_defects, dtype=float)
            self.outlier_scores_ = np.ones(wafer_obj.n_defects, dtype=float)
            self.cluster_persistence_ = np.empty(0, dtype=float)
            self.pattern_results_ = {}
            self.hdbscan_ = None
            self.prefilter_mask_ = None
            return self

        # Step 0: Optional spatial pre-filter
        if self.spatial_prefilter:
            prefilter_mask = _spatial_prefilter(wafer_obj)
            self.prefilter_mask_ = prefilter_mask
            systematic_idx = np.where(prefilter_mask)[0]

            if len(systematic_idx) < self.min_cluster_size:
                # All filtered out — everything is noise
                self.labels_ = np.full(wafer_obj.n_defects, -1, dtype=int)
                self.probabilities_ = np.zeros(wafer_obj.n_defects, dtype=float)
                self.outlier_scores_ = np.ones(wafer_obj.n_defects, dtype=float)
                self.cluster_persistence_ = np.empty(0, dtype=float)
                self.pattern_results_ = {}
                self.hdbscan_ = None
                return self

            # Build a filtered sub-wafer for clustering
            filtered_wafer = WaferMap(
                diameter_mm=wafer_obj.geometry.diameter_mm,
                edge_exclusion_mm=wafer_obj.geometry.edge_exclusion_mm,
            )
            coords = wafer_obj.coordinates[prefilter_mask]
            attrs = {}
            for attr_name in wafer_obj.attribute_names:
                attrs[attr_name] = wafer_obj.get_attribute(attr_name)[prefilter_mask]
            filtered_wafer.add_defects(x=coords[:, 0], y=coords[:, 1], **attrs)
        else:
            self.prefilter_mask_ = None
            filtered_wafer = wafer_obj

        # Step 1: Encode features
        feature_matrix = self._encode_features(filtered_wafer)

        # Step 2: Build distance matrix (with optional edge compensation)
        if self.edge_compensation:
            D = apply_edge_compensation(
                filtered_wafer,
                bandwidth_mm=self.compensation_bandwidth_mm,
                feature_matrix=feature_matrix,
            )
            metric = 'precomputed'
            fit_data = D
        else:
            metric = 'euclidean'
            fit_data = feature_matrix

        # Step 3: Run HDBSCAN (sklearn.cluster.HDBSCAN)
        hdbscan_kwargs: dict[str, Any] = {
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': metric,
            'cluster_selection_method': self.cluster_selection_method,
            'cluster_selection_epsilon': self.cluster_selection_epsilon,
            'allow_single_cluster': self.allow_single_cluster,
        }
        # sklearn HDBSCAN does not support prediction_data parameter
        self.hdbscan_ = HDBSCAN(**hdbscan_kwargs)
        self.hdbscan_.fit(fit_data)

        # Step 4: Extract HDBSCAN outputs
        sub_labels = self.hdbscan_.labels_
        sub_probs = self.hdbscan_.probabilities_

        # sklearn HDBSCAN doesn't expose outlier_scores_ — derive from probabilities
        if hasattr(self.hdbscan_, 'outlier_scores_'):
            sub_outliers = self.hdbscan_.outlier_scores_
        else:
            sub_outliers = 1.0 - sub_probs

        # sklearn HDBSCAN exposes cluster_persistence_ but guard for safety
        if hasattr(self.hdbscan_, 'cluster_persistence_'):
            self.cluster_persistence_ = self.hdbscan_.cluster_persistence_
        else:
            self.cluster_persistence_ = np.empty(0, dtype=float)

        # Expand back to full wafer dimension if prefilter is active
        if self.prefilter_mask_ is not None:
            n_full = wafer_obj.n_defects
            self.labels_ = np.full(n_full, -1, dtype=int)
            self.probabilities_ = np.zeros(n_full, dtype=float)
            self.outlier_scores_ = np.ones(n_full, dtype=float)
            self.labels_[self.prefilter_mask_] = sub_labels
            self.probabilities_[self.prefilter_mask_] = sub_probs
            self.outlier_scores_[self.prefilter_mask_] = sub_outliers
        else:
            self.labels_ = sub_labels
            self.probabilities_ = sub_probs
            self.outlier_scores_ = sub_outliers

        # Store feature data for silhouette computation in summary()
        self.fit_data_ = fit_data
        self.fit_metric_ = metric

        # Step 5: Iterative harvesting (optional)
        if self.harvesting:
            self._run_harvesting(wafer_obj, fit_data, metric)

        # Step 6: Pattern classification (always on the full wafer)
        if self.classify_patterns:
            if self.pattern_classifier == 'deep':
                from .deep_patterns import DeepPatternClassifier

                deep_cls = DeepPatternClassifier()
                self.pattern_results_ = deep_cls.classify(wafer_obj, self.labels_)
            else:
                classifier = DefectPatternClassifier(wafer_obj)
                self.pattern_results_ = classifier.classify_all(self.labels_)
        else:
            self.pattern_results_ = {}

        return self

    def fit_predict(
        self, X: Any = None, y: Any = None, *, wafer: WaferMap | None = None
    ) -> np.ndarray:
        """Fit and return cluster labels.

        Parameters are the same as :meth:`fit`.
        """
        self.fit(X, y, wafer=wafer)
        return self.labels_

    # ------------------------------------------------------------------
    # Prediction on new data
    # ------------------------------------------------------------------

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new defect points.

        Uses HDBSCAN's ``approximate_predict``.  Only available when
        ``edge_compensation=False`` and ``prediction_data=True``.

        Parameters
        ----------
        X_new : np.ndarray, shape ``(n, 2)``
            New defect coordinates.

        Returns
        -------
        np.ndarray of cluster labels.
        """
        check_is_fitted(self, ['hdbscan_', 'labels_'])
        if self.hdbscan_ is None:
            return np.full(len(X_new), -1, dtype=int)
        if self.edge_compensation:
            raise ValueError(
                'predict() is not supported with edge_compensation=True '
                'because HDBSCAN approximate_predict requires euclidean metric. '
                'Set edge_compensation=False or re-fit.'
            )
        # sklearn.cluster.HDBSCAN does not support approximate_predict
        raise NotImplementedError(
            'predict() is not supported with sklearn.cluster.HDBSCAN. '
            'The approximate_predict functionality from the hdbscan package '
            'is not available in the sklearn implementation. '
            'Use fit_predict() on the full dataset instead.'
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Per-cluster summary DataFrame.

        Columns: ``cluster_id``, ``n_defects``, ``pattern_type``,
        ``pattern_confidence``, ``persistence``, ``mean_x``, ``mean_y``,
        ``mean_radius``, ``zone``, ``silhouette_score``.
        """
        check_is_fitted(self, ['labels_', 'wafer_'])
        coords = self.wafer_.coordinates

        # Compute per-sample silhouette scores if there are ≥ 2 clusters
        n_clusters = len(set(self.labels_) - {-1})
        if n_clusters >= 2 and hasattr(self, 'fit_data_'):
            metric = self.fit_metric_ if hasattr(self, 'fit_metric_') else 'euclidean'
            fit_data = self.fit_data_
            if metric == 'precomputed':
                # Ensure diagonal is exactly zero (floating-point noise)
                fit_data = fit_data.copy()
                np.fill_diagonal(fit_data, 0.0)
            sample_silhouettes = silhouette_samples(
                fit_data,
                self.labels_,
                metric=metric,
            )
        else:
            sample_silhouettes = None

        rows = []
        unique_labels = sorted(set(self.labels_))
        for cid in unique_labels:
            if cid == -1:
                continue
            mask = self.labels_ == cid
            pts = coords[mask]
            mean_x, mean_y = pts.mean(axis=0)
            mean_r = float(np.sqrt(mean_x**2 + mean_y**2))

            # Zone assignment
            radius = self.wafer_.geometry.radius_mm
            if mean_r < 0.33 * radius:
                zone = 'center'
            elif mean_r < 0.66 * radius:
                zone = 'middle'
            else:
                zone = 'edge'

            # Pattern info
            pr = self.pattern_results_.get(cid)
            pat_type = pr.pattern_type if pr else 'unclassified'
            pat_conf = pr.confidence if pr else 0.0
            compound_label = pr.compound_label if pr and pr.compound_label else pat_type
            is_compound = pr.is_compound if pr else False

            # Persistence
            persistence = (
                float(self.cluster_persistence_[cid])
                if (self.cluster_persistence_ is not None and cid < len(self.cluster_persistence_))
                else 0.0
            )

            # Silhouette score (mean of per-sample silhouettes for this cluster)
            if sample_silhouettes is not None:
                sil = float(np.mean(sample_silhouettes[mask]))
            else:
                sil = 0.0

            row: dict[str, Any] = {
                'cluster_id': cid,
                'n_defects': int(mask.sum()),
                'pattern_type': pat_type,
                'compound_label': compound_label,
                'is_compound': is_compound,
                'pattern_confidence': round(pat_conf, 3),
                'persistence': round(persistence, 3),
                'mean_x': round(float(mean_x), 2),
                'mean_y': round(float(mean_y), 2),
                'mean_radius': round(mean_r, 2),
                'zone': zone,
                'silhouette_score': round(sil, 3),
            }

            # Add harvest iteration if harvesting was used
            if hasattr(self, 'harvest_iterations_') and self.harvest_iterations_:
                row['harvest_iteration'] = self.harvest_iterations_.get(cid, 0)

            rows.append(row)

        return pd.DataFrame(rows)

    @property
    def n_clusters(self) -> int:
        """Number of clusters found (excluding noise)."""
        check_is_fitted(self, ['labels_'])
        return len(set(self.labels_) - {-1})

    @property
    def noise_fraction(self) -> float:
        """Fraction of defects labelled as noise."""
        check_is_fitted(self, ['labels_'])
        if len(self.labels_) == 0:
            return 0.0
        return float(np.mean(self.labels_ == -1))

    # ------------------------------------------------------------------
    # Iterative harvesting
    # ------------------------------------------------------------------

    def _run_harvesting(
        self,
        wafer_obj: WaferMap,
        fit_data: np.ndarray,
        metric: str,
    ) -> None:
        """Iteratively extract best-silhouette clusters.

        Replaces ``self.labels_``, ``self.probabilities_``, and
        ``self.outlier_scores_`` with harvested results.  Populates
        ``self.harvest_iterations_`` mapping cluster_id → iteration number.
        """
        n = len(self.labels_)
        final_labels = np.full(n, -1, dtype=int)
        final_probs = np.zeros(n, dtype=float)
        harvest_iter_map: dict[int, int] = {}

        # Work with indices into the original arrays
        remaining_mask = np.ones(n, dtype=bool)
        next_cluster_id = 0

        for iteration in range(1, self.harvesting_max_iterations + 1):
            remaining_idx = np.where(remaining_mask)[0]
            if len(remaining_idx) < self.min_cluster_size:
                break

            # Run HDBSCAN on remaining points
            if metric == 'precomputed':
                sub_data = fit_data[np.ix_(remaining_idx, remaining_idx)]
            else:
                sub_data = fit_data[remaining_idx]

            hdb = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=metric,
                cluster_selection_method=self.cluster_selection_method,
                cluster_selection_epsilon=self.cluster_selection_epsilon,
                allow_single_cluster=self.allow_single_cluster,
            )
            hdb.fit(sub_data)
            sub_labels = hdb.labels_
            sub_probs = hdb.probabilities_

            unique_clusters = sorted(set(sub_labels) - {-1})
            if len(unique_clusters) == 0:
                break

            # Compute per-cluster silhouette scores
            if len(unique_clusters) >= 2:
                if metric == 'precomputed':
                    sil_data = sub_data.copy()
                    np.fill_diagonal(sil_data, 0.0)
                else:
                    sil_data = sub_data
                sample_sils = silhouette_samples(sil_data, sub_labels, metric=metric)
            else:
                # Single cluster — can't compute silhouette, use a default
                sample_sils = np.where(sub_labels >= 0, 0.5, 0.0)

            # Find the cluster with highest mean silhouette
            best_cid = -1
            best_sil = -1.0
            for cid in unique_clusters:
                mask = sub_labels == cid
                mean_sil = float(np.mean(sample_sils[mask]))
                if mean_sil > best_sil:
                    best_sil = mean_sil
                    best_cid = cid

            if best_sil < self.harvesting_min_silhouette:
                logger.debug(
                    'Harvesting stopped at iteration %d: best silhouette %.3f < threshold %.3f',
                    iteration,
                    best_sil,
                    self.harvesting_min_silhouette,
                )
                break

            # Harvest the best cluster
            harvested_local = sub_labels == best_cid
            harvested_global = remaining_idx[harvested_local]

            final_labels[harvested_global] = next_cluster_id
            final_probs[harvested_global] = sub_probs[harvested_local]
            harvest_iter_map[next_cluster_id] = iteration

            logger.debug(
                'Harvested cluster %d (iteration %d): %d points, silhouette=%.3f',
                next_cluster_id,
                iteration,
                int(harvested_local.sum()),
                best_sil,
            )

            remaining_mask[harvested_global] = False
            next_cluster_id += 1

        self.labels_ = final_labels
        self.probabilities_ = final_probs
        self.outlier_scores_ = 1.0 - final_probs
        self.harvest_iterations_ = harvest_iter_map

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_wafer(X: Any, wafer: WaferMap | None) -> WaferMap:
        """Resolve the wafer object from flexible input."""
        if isinstance(X, WaferMap):
            return X
        if wafer is not None:
            return wafer
        if isinstance(X, np.ndarray):
            w = WaferMap(diameter_mm=300.0)
            w.add_defects(x=X[:, 0], y=X[:, 1])
            return w
        raise TypeError(
            'X must be a WaferMap, a numpy array of shape (n, 2), '
            'or None with wafer= keyword argument provided.'
        )

    def _encode_features(self, wafer: WaferMap) -> np.ndarray:
        """Encode features, falling back to raw coordinates."""
        if self.feature_encoder is not None:
            return self.feature_encoder.fit_transform(wafer)
        return wafer.coordinates

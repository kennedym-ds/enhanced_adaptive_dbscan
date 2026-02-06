"""
Wafer Defect Clustering
=======================

Semiconductor wafer defect clustering with HDBSCAN, edge density compensation,
and automatic defect pattern classification.

Every line in this library does something no general-purpose clustering
package provides: wafer geometry handling, edge bias correction, and
semiconductor-specific pattern recognition (scratches, rings, centre spots,
edge clusters, zone patterns, repeating die patterns).

Quick start
-----------
>>> from wafer_defect_clustering import WaferMap, WaferClusterer, plot_wafer_map
>>> wafer = WaferMap(diameter_mm=300, edge_exclusion_mm=3.0)
>>> wafer.add_defects(x=defect_x, y=defect_y, size=defect_sizes)
>>> clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=True)
>>> labels = clusterer.fit_predict(wafer)
>>> print(clusterer.summary())
>>> fig = plot_wafer_map(wafer, labels, show_zones=True)
>>> fig.show()
"""

from .clusterer import WaferClusterer
from .deep_patterns import DeepPatternClassifier
from .edge_compensation import (
    apply_edge_compensation,
    compute_adaptive_bandwidth,
    compute_area_coverage_fraction,
    compute_edge_density_weights,
)
from .features import DefectFeatureEncoder
from .klarf import load_klarf, parse_klarf_string
from .lot_analysis import ExcursionResult, LotAnalyzer, RecurringPattern
from .patterns import DefectPatternClassifier, PatternResult
from .spatial import SpatialTestResult, spatial_prefilter, spatial_randomness_test
from .streaming import StreamingClusterer
from .visualization import (
    plot_cluster_details,
    plot_pattern_summary,
    plot_radial_distribution,
    plot_wafer_grid,
    plot_wafer_map,
)
from .wafer import STANDARD_ZONES, WaferGeometry, WaferMap, ZoneDefinition

__version__ = '2.0.0'

__all__ = [
    # Core domain objects
    'WaferMap',
    'WaferGeometry',
    'ZoneDefinition',
    'STANDARD_ZONES',
    # Clustering
    'WaferClusterer',
    # Feature encoding
    'DefectFeatureEncoder',
    # Pattern classification
    'DefectPatternClassifier',
    'PatternResult',
    # Edge compensation
    'compute_adaptive_bandwidth',
    'compute_area_coverage_fraction',
    'compute_edge_density_weights',
    'apply_edge_compensation',
    # Spatial analysis
    'SpatialTestResult',
    'spatial_randomness_test',
    'spatial_prefilter',
    # KLARF ingestion
    'load_klarf',
    'parse_klarf_string',
    # Deep learning patterns
    'DeepPatternClassifier',
    # Streaming clustering
    'StreamingClusterer',
    # Lot analysis
    'LotAnalyzer',
    'RecurringPattern',
    'ExcursionResult',
    # Visualization
    'plot_wafer_map',
    'plot_wafer_grid',
    'plot_pattern_summary',
    'plot_radial_distribution',
    'plot_cluster_details',
]

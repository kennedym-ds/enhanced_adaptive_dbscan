# Wafer Defect Clustering

[![PyPI version](https://badge.fury.io/py/wafer-defect-clustering.svg)](https://badge.fury.io/py/wafer-defect-clustering)
[![CI](https://github.com/kennedym-ds/enhanced_adaptive_dbscan/workflows/CI/badge.svg)](https://github.com/kennedym-ds/enhanced_adaptive_dbscan/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Semiconductor wafer defect clustering with automatic pattern classification.

Built on [HDBSCAN](https://hdbscan.readthedocs.io/) with wafer-specific extensions that no general clustering library provides:

- **Edge density compensation** — corrects for geometric bias at wafer boundaries using circle-circle intersection geometry
- **Defect pattern classification** — automatically identifies scratches, rings, edge clusters, centre spots, zone patterns, and repeating die patterns
- **Wafer-aware visualization** — interactive Plotly wafer maps with zone overlays, die grids, and pattern annotations
- **Feature encoding** — handles heterogeneous defect attributes (size, severity, layer, classification code)
- **Full sklearn compatibility** — `BaseEstimator` / `ClusterMixin`, `fit()`, `fit_predict()`, `get_params()` / `set_params()`

## Why not just HDBSCAN?

HDBSCAN is an excellent general-purpose density-based clustering algorithm.  This library adds the **semiconductor domain layer** on top:

| Problem | HDBSCAN alone | This library |
|---|---|---|
| Edge bias | Treats all space equally — edge clusters appear less dense | Circle-circle intersection geometry compensates for truncated neighbourhoods |
| Pattern ID | Gives you cluster labels | Classifies clusters as *scratch*, *ring*, *centre spot*, *edge cluster*, *zone pattern*, or *repeating* |
| Wafer geometry | No concept of wafer shape | Built-in circular/square wafer, notch/flat, edge exclusion zones |
| Die mapping | N/A | Aggregates defects to die-level counts |
| Zone analysis | N/A | Standard centre/middle/edge zone masking |
| Defect features | You build your own feature matrix | `DefectFeatureEncoder` handles log-size, severity, layer encoding with configurable weights |

## Quick Start

```bash
pip install wafer-defect-clustering
```

```python
from wafer_defect_clustering import WaferMap, WaferClusterer, plot_wafer_map

# Create wafer and add inspection data
wafer = WaferMap(diameter_mm=300, edge_exclusion_mm=3.0)
wafer.add_defects(x=defect_x, y=defect_y, size=defect_sizes)

# Cluster with edge compensation
clusterer = WaferClusterer(min_cluster_size=5, edge_compensation=True)
labels = clusterer.fit_predict(wafer)

# Results
print(clusterer.summary())
#   cluster_id  n_defects pattern_type  pattern_confidence  ...  zone
# 0          0         23      scratch               0.871  ...  middle
# 1          1         15         ring               0.784  ...  edge
# 2          2          8  center_spot               0.912  ...  center

# Interactive wafer map
fig = plot_wafer_map(wafer, labels, show_zones=True)
fig.show()
```

## Core Concepts

### WaferMap — the domain object

```python
from wafer_defect_clustering import WaferMap

wafer = WaferMap(
    diameter_mm=300,           # 200, 300, or 450 mm
    edge_exclusion_mm=3.0,     # no-die zone at edge
    notch_angle_deg=270,       # 6 o'clock (industry standard for 300mm)
)

# Add defect inspection data (coordinates in mm from wafer centre)
wafer.add_defects(
    x=x_coords, y=y_coords,
    size=defect_sizes,         # µm² (optional)
    kill=kill_flags,           # 0/1 (optional)
    layer=layer_ids,           # process layer (optional)
    classcode=class_codes,     # defect classification (optional)
)

# Geometry-aware queries
wafer.distance_to_edge()               # mm to nearest edge
wafer.get_zone_mask('edge')             # boolean mask
wafer.get_zone_label()                  # 'center'/'middle'/'edge' per defect
wafer.to_die_map(die_size_mm=(10, 10))  # DataFrame of die-level counts
```

### WaferClusterer — HDBSCAN with wafer domain logic

```python
from wafer_defect_clustering import WaferClusterer, DefectFeatureEncoder

# Spatial-only clustering (default)
clusterer = WaferClusterer(
    min_cluster_size=5,
    edge_compensation=True,          # correct edge density bias
    compensation_bandwidth_mm=5.0,   # compensation kernel width
    classify_patterns=True,          # auto-classify cluster patterns
    cluster_selection_method='eom',  # HDBSCAN selection method
)

# With defect attribute encoding
encoder = DefectFeatureEncoder(
    spatial_weight=1.0,    # x, y importance
    size_weight=0.5,       # defect size importance
    severity_weight=0.3,   # kill flag importance
    layer_weight=0.2,      # process layer importance
)
clusterer = WaferClusterer(
    min_cluster_size=5,
    feature_encoder=encoder,
    edge_compensation=False,  # precomputed distances not needed with feature encoder
)

labels = clusterer.fit_predict(wafer)

# Access HDBSCAN outputs directly
clusterer.probabilities_        # membership probabilities
clusterer.outlier_scores_       # GLOSH outlier scores
clusterer.cluster_persistence_  # cluster persistence
clusterer.pattern_results_      # {cluster_id: PatternResult}
clusterer.n_clusters            # number of clusters (excl. noise)
clusterer.noise_fraction        # fraction labelled as noise
```

### DefectPatternClassifier — automatic pattern recognition

```python
from wafer_defect_clustering import DefectPatternClassifier

clf = DefectPatternClassifier(wafer)
results = clf.classify_all(labels)

for cluster_id, result in results.items():
    print(f"Cluster {cluster_id}: {result.pattern_type} "
          f"(confidence={result.confidence:.2f})")
    print(f"  Details: {result.details}")
```

**Detected patterns:**
- `scratch` — linear defects (PCA eccentricity + aspect ratio)
- `ring` — annular pattern at consistent radius (radial CV + angular span)
- `center_spot` — concentration near wafer centre (radial density)
- `edge_cluster` — concentration at wafer periphery (distance-to-edge stats)
- `zone_pattern` — confined to angular sector (circular std deviation)
- `repeating` — periodic die-grid pattern (intra-die position clustering)
- `random` — no geometric signature (default fallback)

### Visualization

```python
from wafer_defect_clustering import (
    plot_wafer_map,
    plot_wafer_grid,
    plot_radial_distribution,
    plot_pattern_summary,
    plot_cluster_details,
)

# Single wafer with clusters, zones, die grid
plot_wafer_map(wafer, labels, show_zones=True, show_dies=True)

# Lot overview (multiple wafers)
plot_wafer_grid(wafer_list, labels_list, ncols=5)

# Radial distribution histogram
plot_radial_distribution(wafer, labels)

# Pattern confidence bar chart
plot_pattern_summary(clusterer.pattern_results_)

# Detailed single-cluster view with pattern overlay
plot_cluster_details(wafer, labels, cluster_id=0,
                     pattern_result=clusterer.pattern_results_[0])
```

## Edge Density Compensation

The key innovation.  Near wafer edges, defect neighbourhoods are truncated by the physical boundary — a cluster at the edge appears less dense than an identical cluster at the centre.

This library computes the **exact fractional area** of each point's neighbourhood circle that falls inside the wafer (circle-circle intersection), and scales distances accordingly:

```python
from wafer_defect_clustering import compute_edge_density_weights

weights = compute_edge_density_weights(wafer, bandwidth_mm=5.0)
# weights ≈ 1.0 at centre, > 1.0 near edge
```

The compensated distance matrix is passed to HDBSCAN with `metric='precomputed'`, ensuring edge clusters receive fair density estimates.

## API Reference

| Class / Function | Purpose |
|---|---|
| `WaferMap` | Wafer geometry + defect data model |
| `WaferGeometry` | Physical wafer parameters (dataclass) |
| `ZoneDefinition` | Named radial zone on wafer |
| `WaferClusterer` | HDBSCAN wrapper with edge compensation + pattern classification |
| `DefectFeatureEncoder` | Encode defect attributes into weighted features |
| `DefectPatternClassifier` | Classify clusters into known defect patterns |
| `PatternResult` | Pattern classification result (dataclass) |
| `compute_edge_density_weights` | Per-defect edge compensation weights |
| `apply_edge_compensation` | Build compensated distance matrix |
| `plot_wafer_map` | Interactive wafer map plot |
| `plot_wafer_grid` | Lot-level wafer grid |
| `plot_radial_distribution` | Radial histogram |
| `plot_pattern_summary` | Pattern confidence bar chart |
| `plot_cluster_details` | Single-cluster detail with pattern overlay |

## Requirements

- Python >= 3.10
- [hdbscan](https://hdbscan.readthedocs.io/) >= 0.8.40
- numpy, scikit-learn, scipy, plotly, pandas

## License

MIT

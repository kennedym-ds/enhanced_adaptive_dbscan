# Wafer Defect Clustering - AI Coding Agent Instructions

## Project Overview
HDBSCAN-based semiconductor wafer defect clustering with edge density compensation, automatic pattern classification, and Plotly visualization. Single-purpose, production-focused.

## Architecture

### Package: `wafer_defect_clustering/`
| Module | Purpose |
|---|---|
| `wafer.py` | `WaferGeometry`, `WaferMap`, `ZoneDefinition` — wafer geometry & defect data model |
| `edge_compensation.py` | Circle-circle intersection density bias correction at wafer boundaries |
| `features.py` | `DefectFeatureEncoder` — encode size/kill/layer/classcode for clustering |
| `patterns.py` | `DefectPatternClassifier` — classify clusters as scratch/ring/center_spot/edge_cluster/zone_pattern/repeating/random |
| `clusterer.py` | `WaferClusterer(BaseEstimator, ClusterMixin)` — main HDBSCAN wrapper with domain logic |
| `visualization.py` | Plotly wafer maps, grids, pattern summaries, radial distributions |

### Key Dependencies
- `scikit-learn>=1.5` — includes `sklearn.cluster.HDBSCAN` (core clustering engine)
- `numpy>=1.26`, `scipy>=1.14` — computation
- `plotly>=5.24`, `pandas>=2.2` — visualization & data handling
- Python `>=3.10`

## Development Workflows

### Environment Setup
```bash
Ctrl+Shift+P → "Tasks: Run Task" → "venv: create (.venv, py -3.12)"
Ctrl+Shift+P → "Tasks: Run Task" → "install: project (editable)"
```

### Testing
```bash
pytest tests/ -v --tb=short                                    # full suite
pytest -q --cov=wafer_defect_clustering --cov-report=term-missing  # with coverage
```

### Code Quality
```bash
ruff check .        # lint
ruff format .       # format
mypy .              # type check
pre-commit run --all-files  # all checks
```

## Coding Patterns

### Sklearn API Compliance
`WaferClusterer` inherits `BaseEstimator, ClusterMixin` and exposes:
- `fit(X, y=None)`, `fit_predict(X, y=None)`, `predict(X)`
- `labels_`, `probabilities_`, `outlier_scores_`, `cluster_persistence_`
- `get_params()` / `set_params()` via BaseEstimator
- `check_is_fitted()` validation before predict

### Edge Compensation Pattern
```python
# compute_area_coverage_fraction accepts scalar or ndarray (vectorized)
weights = compute_edge_density_weights(wafer, bandwidth_mm=5.0)
D_comp = apply_edge_compensation(wafer, bandwidth_mm=5.0)  # symmetric distance matrix
# sklearn.cluster.HDBSCAN uses metric='precomputed' with D_comp
```

### WaferMap Data Flow
```python
wafer = WaferMap(geometry=WaferGeometry(diameter_mm=300.0))
wafer.add_defects(xy_coords)  # (N, 2) array
clusterer = WaferClusterer(edge_compensation=True)
labels = clusterer.fit_predict(wafer)
df = clusterer.summary()  # includes silhouette_score column
fig = plot_wafer_map(wafer, labels)
```

### Polar Feature Encoding
```python
encoder = DefectFeatureEncoder(polar_weight=0.4, size_weight=0.5)
X = encoder.fit_transform(wafer)  # adds r, sin(θ), cos(θ) columns
```

### Pattern Classification
`DefectPatternClassifier` runs all detectors, picks highest-confidence match:
- scratch: PCA eccentricity + aspect ratio
- ring: low radial CV + wide angular span
- center_spot: high radial density near center
- edge_cluster: proximity to wafer edge
- zone_pattern: low circular std (localized angular region)
- repeating: intra-die modular position clustering
- random: fallback when no pattern scores above threshold

## Testing Conventions
- `test_wafer.py` — WaferGeometry, WaferMap, zones
- `test_edge_compensation.py` — coverage fractions, weights, distance matrices
- `test_features.py` — DefectFeatureEncoder
- `test_patterns.py` — pattern classification per type
- `test_clusterer.py` — WaferClusterer fit/predict/sklearn API
- `test_visualization.py` — Plotly figure generation

## Anti-Patterns
❌ Don't bypass edge compensation for edge-heavy wafer data
❌ Don't break sklearn API compatibility on WaferClusterer
❌ Don't use `ndarray.ptp()` — removed in NumPy 2.0, use `np.ptp()` instead
❌ Don't add optional/heavy dependencies without clear justification

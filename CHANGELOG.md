# Changelog

All notable changes to Wafer Defect Clustering will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-02-06

### Added
- **Spatial Intelligence** — `spatial_randomness_test()` (Clark-Evans statistic), `spatial_prefilter()` (k-NN graph pre-filter) in new `spatial.py` module
- **Adaptive Bandwidth** — `compute_adaptive_bandwidth()` with local density estimation; `bandwidth_mm='auto'` support in edge compensation pipeline
- **Iterative Harvesting** — `WaferClusterer(harvesting=True)` for multi-pass clustering with silhouette-guided stopping
- **Spatial Pre-filter** — `WaferClusterer(spatial_prefilter=True)` removes isolated noise before HDBSCAN
- **KLARF File Ingestion** — `load_klarf()` and `parse_klarf_string()` for KLARF 1.x format; `WaferMap.from_klarf()` classmethod; optional `klarfkit` backend
- **WM-811K Benchmark** — `pixel_to_coordinates()`, `run_benchmark()` in `benchmarks/wm811k_benchmark.py`; classification report against WM-811K ground truth
- **Deep Learning Patterns** — `DeepPatternClassifier` with `rasterize()`, `classify()`, `classify_wafer()`; optional torch/ONNX backends; fall-back rule-based heuristic; open-set novelty detection via `novelty_score`
- **Streaming Clustering** — `StreamingClusterer` for incremental micro-batch HDBSCAN with configurable `batch_size`
- **Lot-Level Analysis** — `LotAnalyzer` with `pattern_summary()`, `recurring_patterns()`, `excursion_check()`, `plot_lot_overview()`; `RecurringPattern` and `ExcursionResult` dataclasses
- 131 new tests (95 → 226) across 12 test modules

### Changed
- Edge compensation vectorised for large wafer maps (Phase 1 performance)
- `WaferClusterer.__init__` expanded: `spatial_prefilter`, `harvesting`, `harvesting_max_iterations`, `harvesting_min_silhouette`, `pattern_classifier` parameters
- Pattern classifier routing: `pattern_classifier='deep'` delegates to `DeepPatternClassifier`

## [1.0.0] - 2026-02-06

### Added
- `WaferMap` and `WaferGeometry` domain objects for wafer geometry and defect data
- `WaferClusterer` — sklearn-compatible HDBSCAN wrapper with semiconductor domain logic
- Edge density compensation via circle-circle intersection geometry
- `DefectFeatureEncoder` for heterogeneous defect attribute encoding
- `DefectPatternClassifier` with 7 pattern types: scratch, ring, center_spot, edge_cluster, zone_pattern, repeating, random
- Plotly visualization: `plot_wafer_map`, `plot_wafer_grid`, `plot_pattern_summary`, `plot_radial_distribution`, `plot_cluster_details`
- Zone-based spatial analysis (center/middle/edge)
- Die-level repeating pattern detection
- 95 tests across 6 test modules

### Changed
- Complete rewrite from `enhanced_adaptive_dbscan` multi-phase framework
- Switched from custom adaptive DBSCAN to HDBSCAN as clustering engine
- Minimum Python version raised to 3.10 (from 3.8)
- Streamlined dependencies: removed Flask, joblib, psutil, matplotlib, pyyaml

### Removed
- Multi-phase architecture (Phases 1-4)
- Custom DBSCAN implementation (`dbscan.py`, `density_engine.py`)
- Ensemble clustering (`ensemble_clustering.py`)
- Adaptive optimization framework (`adaptive_optimization.py`)
- Production pipeline, streaming engine, web API
- Benchmarks and Sphinx documentation

[2.0.0]: https://github.com/kennedym-ds/enhanced_adaptive_dbscan/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/kennedym-ds/enhanced_adaptive_dbscan/releases/tag/v1.0.0

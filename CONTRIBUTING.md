# Contributing to Wafer Defect Clustering

## Getting Started

```bash
git clone https://github.com/kennedym-ds/enhanced_adaptive_dbscan.git
cd enhanced_adaptive_dbscan
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -e .[dev]
pre-commit install
```

Or use VS Code tasks: `Ctrl+Shift+P` → "Tasks: Run Task" → `venv: create` then `install: project (editable)`.

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Write tests first, then implement
3. Run quality checks before committing

### Code Quality

```bash
pytest tests/ -v --tb=short                                     # tests
pytest -q --cov=wafer_defect_clustering --cov-report=term-missing  # coverage
ruff check .                                                     # lint
ruff format .                                                    # format
mypy .                                                           # type check
pre-commit run --all-files                                       # all checks
```

### Code Style

- Python 3.10+ with type hints on all public APIs
- Ruff for linting and formatting
- NumPy-style docstrings
- sklearn API compatibility for `WaferClusterer`

### Testing

Every change must include tests:

```python
import numpy as np
from wafer_defect_clustering import WaferMap, WaferGeometry, WaferClusterer

def test_your_feature():
    wafer = WaferMap(geometry=WaferGeometry(diameter_mm=300.0))
    wafer.add_defects(np.random.randn(50, 2) * 30)
    clusterer = WaferClusterer(min_cluster_size=5)
    labels = clusterer.fit_predict(wafer)
    assert len(labels) == 50
```

### Commit Messages

```
feat: add new pattern detector for cross-wafer defects
fix: correct edge compensation for small bandwidths
docs: update API reference for WaferClusterer
test: add parametrized tests for zone classification
```

## Pull Request Checklist

- [ ] Tests pass (`pytest`)
- [ ] New tests for new features
- [ ] Type hints on public APIs
- [ ] Docstrings updated
- [ ] CHANGELOG.md updated
- [ ] sklearn API compatibility preserved

## Project Structure

```
wafer_defect_clustering/       # Source
├── wafer.py                   # WaferMap, WaferGeometry
├── edge_compensation.py       # Density bias correction
├── features.py                # DefectFeatureEncoder
├── patterns.py                # DefectPatternClassifier
├── clusterer.py               # WaferClusterer (HDBSCAN)
├── visualization.py           # Plotly plots
└── __init__.py                # Public API
tests/                         # Test suite
├── test_wafer.py
├── test_edge_compensation.py
├── test_features.py
├── test_patterns.py
├── test_clusterer.py
└── test_visualization.py
```

## Anti-Patterns

- Don't use `ndarray.ptp()` — removed in NumPy 2.0, use `np.ptp()`
- Don't break sklearn `BaseEstimator`/`ClusterMixin` API on `WaferClusterer`
- Don't add heavy optional dependencies without justification
- Prefer vectorized NumPy over Python loops

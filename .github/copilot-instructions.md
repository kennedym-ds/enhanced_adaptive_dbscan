# GitHub Copilot Instructions - Enhanced Adaptive DBSCAN

## Project Overview

This is a comprehensive adaptive clustering framework featuring Enhanced Adaptive DBSCAN, Ensemble Clustering, and Adaptive Parameter Optimization. The project is designed for complex data analysis including semiconductor wafer defect detection, multi-density datasets, and automatic parameter tuning.

## Coding Style and Standards

### Python Style
- **Python Version**: Minimum Python 3.8, tested on 3.9-3.12
- **Formatting**: Use Ruff formatter with the following settings:
  - Line length: 100 characters
  - Quote style: single quotes
  - Indentation: spaces (4 spaces per level)
- **Linting**: Follow Ruff linting rules (E, F, I, UP, B, SIM, PL)
- **Type Hints**: Add type hints where possible, but not mandatory (mypy checks are lenient)
- **Imports**: Organize imports automatically with Ruff (isort)

### Code Quality
- Follow PEP 8 conventions
- Write clear, descriptive variable and function names
- Add docstrings for public functions and classes
- Keep functions focused and modular
- Prefer composition over inheritance where appropriate

## Testing Requirements

### Test Framework
- Use **pytest** for all tests
- Tests are located in the `tests/` directory
- Run tests with: `pytest -q --cov=enhanced_adaptive_dbscan --cov-report=term-missing`

### Test Coverage
- Aim for high test coverage (project uses codecov)
- Write unit tests for new features and bug fixes
- Include edge cases and error conditions
- Use pytest fixtures for test data setup

### Test Style
- Test file names: `test_*.py`
- Test function names: `test_*`
- Use descriptive test names that explain what is being tested
- Follow existing test patterns in the repository

## Build and Validation Commands

### Installation
```bash
# Install in development mode with test dependencies
pip install -e .[test]

# Install with all dev dependencies (includes linting and docs)
pip install -e .[dev]
```

### Linting and Formatting
```bash
# Check code with Ruff
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .

# Type checking with mypy
mypy .
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=enhanced_adaptive_dbscan --cov-report=term-missing

# Run specific test file
pytest tests/test_dbscan.py
```

### Pre-commit Hooks
The project uses pre-commit hooks. Install them with:
```bash
pre-commit install
```

## Project Structure

### Main Package (`enhanced_adaptive_dbscan/`)
- `dbscan.py` - Core Enhanced Adaptive DBSCAN implementation
- `density_engine.py` - Multi-scale density analysis
- `ensemble_clustering.py` - Ensemble clustering with voting mechanisms
- `multi_density_clustering.py` - Multi-density region detection
- `adaptive_optimization.py` - Bayesian and genetic algorithm optimization
- `production_pipeline.py` - Production model lifecycle management
- `streaming_engine.py` - Real-time streaming clustering
- `deep_clustering.py` - Deep learning integration (autoencoders, DEC)
- `hdbscan_clustering.py` - Hierarchical DBSCAN implementation
- `scalable_indexing.py` - Approximate nearest neighbor search (Annoy/FAISS)
- `web_api.py` - Flask-based REST API
- `boundary_processor.py` - Boundary point detection
- `cluster_quality_analyzer.py` - Cluster quality metrics
- `utils.py` - Utility functions

### Key Dependencies
- **numpy** (>=1.26, <3.0) - Core numerical operations
- **scikit-learn** (>=1.5, <1.7) - Machine learning utilities
- **plotly** (>=5.24, <7.0) - Interactive visualizations
- **pandas** (>=2.2, <3.0) - Data manipulation
- **joblib** (>=1.3, <2.0) - Parallel processing

### Optional Dependencies
When features require additional libraries (FAISS, Annoy, TensorFlow), they are optional and imported conditionally. Always check if the library is available before using it.

## Security Best Practices

### Data Handling
- Never commit sensitive data, credentials, or API keys
- Validate all user inputs in API endpoints
- Use secure random number generation for cryptographic purposes
- Be cautious with pickle files (potential security risk)

### Dependencies
- Keep dependencies up to date
- Review security advisories before adding new dependencies
- Prefer well-maintained, popular libraries
- Specify version ranges to avoid breaking changes

### Code Safety
- Avoid using `eval()` or `exec()` on user input
- Sanitize file paths to prevent path traversal attacks
- Use parameterized queries if working with databases
- Handle exceptions gracefully without exposing sensitive information

## Documentation

### Code Documentation
- Add docstrings to all public classes and functions
- Use NumPy-style docstrings for consistency
- Include parameter types, return types, and examples where helpful
- Document complex algorithms with inline comments

### External Documentation
- Documentation is built with Sphinx
- Source files are in `docs/source/`
- Build docs with: `cd docs && make html`
- Documentation supports Markdown (via myst-parser) and reStructuredText

## Preferences and Conventions

### NumPy and Array Operations
- Prefer NumPy array operations over loops for performance
- Use vectorized operations when possible
- Document array shapes in docstrings (e.g., `X : ndarray of shape (n_samples, n_features)`)

### scikit-learn Compatibility
- Follow scikit-learn API conventions where applicable
- Implement `fit`, `predict`, `fit_predict` methods for clustering classes
- Use `check_array` and `check_X_y` for input validation
- Include `random_state` parameters for reproducibility

### Error Handling
- Use specific exception types
- Provide informative error messages
- Validate inputs early in functions
- Use assertions for internal consistency checks

### Performance
- Consider memory efficiency for large datasets
- Use joblib for parallelization where appropriate
- Profile code before optimizing
- Document performance characteristics in docstrings

## CI/CD Pipeline

The project uses GitHub Actions for CI/CD:
- Tests run on Ubuntu and Windows with Python 3.9-3.12
- Linting and type checking run on Python 3.12
- Code coverage is uploaded to Codecov
- Documentation is built automatically
- Wheels are built for releases

When making changes:
- Ensure all tests pass locally before pushing
- Run linting and formatting before committing
- Update tests if changing behavior
- Update documentation if adding features

## Exclusions

### Do Not Modify
- `.git/` - Git internal files
- `.github/agents/` - Agent configuration files
- `build/`, `dist/` - Build artifacts
- `*.egg-info/` - Package metadata
- `__pycache__/` - Python cache files
- `.pytest_cache/` - Pytest cache
- `htmlcov/` - Coverage reports
- `.mypy_cache/` - MyPy cache

### Auto-generated Files
- Do not manually edit files in `build/` or `dist/`
- Coverage reports are auto-generated
- Documentation builds are automated

## Additional Notes

- The project supports both basic clustering and advanced features (deep learning, streaming)
- Some advanced features have optional dependencies
- Maintain backward compatibility with Python 3.8
- Keep the README.md up to date with major changes
- Version bumping uses `scripts/bump_version.py`

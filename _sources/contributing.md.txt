# Contributing

We welcome contributions! Here's how to get started.

## Setup
- Use Python 3.12.
- In VS Code, run tasks:
  - `venv: create (.venv, py -3.12)`
  - `install: deps (requirements.txt)`
  - `install: project (editable)`

## Development workflow
- Lint: `lint: flake8`
- Test: `test: pytest`
- Build: `build: sdist+wheel`
- Docs: `docs: html (Sphinx)`

## Branching & PRs
- Create feature branches from `main`.
- Add/adjust tests for behavior changes.
- Update docs when you change public APIs.

## Releasing
- Use VS Code tasks under `release:*` to bump, build, and upload.
- Use TestPyPI first; then PyPI when ready.

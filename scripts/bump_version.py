#!/usr/bin/env python3
"""
Simple version bumper for PEP 621 pyproject.toml and optional package __init__.

Usage:
  python scripts/bump_version.py [major|minor|patch|pre]

- Reads [project].version from pyproject.toml
- Bumps according to the part:
    major: X.Y.Z -> (X+1).0.0
    minor: X.Y.Z -> X.(Y+1).0
    patch: X.Y.Z -> X.Y.(Z+1)
    pre:   X.Y.Z -> X.Y.Z.postN (incremental post-release)
- Writes the new version back to pyproject.toml
- Optionally updates enhanced_adaptive_dbscan/__init__.py __version__ if present

Environment:
- No external deps; uses tomllib (Python 3.11+) or tomli fallback if available.
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # rely on simple regex fallback

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PKG_INIT = ROOT / "enhanced_adaptive_dbscan" / "__init__.py"


def read_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    if tomllib:
        data = tomllib.loads(text)
        return data["project"]["version"]
    # Fallback regex parse
    m = re.search(r"^version\s*=\s*\"([^\"]+)\"", text, flags=re.M)
    if not m:
        raise RuntimeError("Could not find project.version in pyproject.toml")
    return m.group(1)


def write_version(new_ver: str) -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    text = re.sub(r"^(version\s*=\s*)\"[^\"]+\"", rf"\\1\"{new_ver}\"", text, flags=re.M)
    PYPROJECT.write_text(text, encoding="utf-8")

    # Optionally update __version__ in package __init__.py if present
    if PKG_INIT.exists():
        init_text = PKG_INIT.read_text(encoding="utf-8")
        if "__version__" in init_text:
            init_text = re.sub(r"^__version__\s*=\s*\"[^\"]*\"", f'__version__ = "{new_ver}"', init_text, flags=re.M)
        else:
            init_text = init_text.rstrip() + f"\n\n__version__ = \"{new_ver}\"\n"
        PKG_INIT.write_text(init_text, encoding="utf-8")


def bump(ver: str, part: str) -> str:
    # Handle PEP 440 post releases for 'pre' option
    if part == "pre":
        m = re.search(r"^(\d+\.\d+\.\d+)(?:\.post(\d+))?$", ver)
        if not m:
            # normalize to X.Y.Z first
            core = normalize_core(ver)
            return f"{core}.post1"
        core, post = m.group(1), m.group(2)
        n = int(post) + 1 if post else 1
        return f"{core}.post{n}"

    core = normalize_core(ver)
    major, minor, patch = map(int, core.split("."))
    if part == "major":
        return f"{major+1}.0.0"
    if part == "minor":
        return f"{major}.{minor+1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch+1}"
    raise SystemExit("Usage: python scripts/bump_version.py [major|minor|patch|pre]")


def normalize_core(ver: str) -> str:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", ver)
    if not m:
        raise ValueError(f"Unsupported version format: {ver}")
    return ".".join(m.groups())


def main(argv: list[str]) -> int:
    part = argv[1] if len(argv) > 1 else "patch"
    if part not in {"major", "minor", "patch", "pre"}:
        print("Invalid part. Choose from: major|minor|patch|pre", file=sys.stderr)
        return 2
    current = read_version()
    new_ver = bump(current, part)
    write_version(new_ver)
    print(new_ver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

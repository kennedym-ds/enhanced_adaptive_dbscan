"""
KLARF file ingestion for semiconductor wafer defect data.

`KLARF <https://en.wikipedia.org/wiki/KLARF>`_ (KLA Results File) is the
industry-standard format produced by KLA and other wafer inspection tools.
This module provides a lightweight pure-Python parser for the most common
KLARF 1.x fields so that ``wafer_defect_clustering`` can directly ingest
fab inspection data without an external library.

If ``klarfkit`` is installed it will be preferred for parsing.  Otherwise
the built-in fallback parser handles the core fields:
coordinates, defect size, classcode, wafer ID, lot ID, and die info.

Public API
----------
- :func:`load_klarf` — load a KLARF file and return a :class:`WaferMap`
- :func:`parse_klarf_string` — parse KLARF text and return raw dicts
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = [
    'load_klarf',
    'parse_klarf_string',
]


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def load_klarf(
    path: str | Path,
    *,
    wafer_id: int = 0,
) -> WaferMap:
    """Load a KLARF file and return a :class:`WaferMap`.

    Parameters
    ----------
    path : str or Path
        Path to the KLARF file.
    wafer_id : int
        Zero-based index of the wafer to load from a multi-wafer file.

    Returns
    -------
    WaferMap
        Wafer with defect coordinates and attributes populated.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file cannot be parsed as valid KLARF.
    IndexError
        If *wafer_id* is out of range.
    """
    from .wafer import WaferMap  # deferred to avoid circular import

    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f'KLARF file not found: {filepath}')

    text = filepath.read_text(encoding='utf-8', errors='replace')
    wafers = parse_klarf_string(text)

    if not wafers:
        raise ValueError(
            f'No wafer data found in KLARF file: {filepath}. '
            'Check that the file contains valid KLARF defect records.'
        )

    if wafer_id < 0 or wafer_id >= len(wafers):
        raise IndexError(
            f'wafer_id={wafer_id} out of range — file contains {len(wafers)} wafer(s).'
        )

    wd = wafers[wafer_id]

    diameter = wd.get('diameter_mm', 300.0)
    wafer = WaferMap(diameter_mm=diameter, edge_exclusion_mm=3.0)

    attrs: dict[str, Any] = {}
    if 'size' in wd and wd['size'] is not None:
        attrs['size'] = np.asarray(wd['size'], dtype=float)
    if 'classcode' in wd and wd['classcode'] is not None:
        attrs['classcode'] = np.asarray(wd['classcode'], dtype=int)
    if 'die_x' in wd and wd['die_x'] is not None:
        attrs['die_x'] = np.asarray(wd['die_x'], dtype=float)
    if 'die_y' in wd and wd['die_y'] is not None:
        attrs['die_y'] = np.asarray(wd['die_y'], dtype=float)

    wafer.add_defects(
        x=np.asarray(wd['x'], dtype=float),
        y=np.asarray(wd['y'], dtype=float),
        **attrs,
    )

    logger.info(
        'Loaded KLARF: wafer_id=%s, lot=%s, %d defects, diameter=%.0f mm',
        wd.get('wafer_id', '?'),
        wd.get('lot_id', '?'),
        wafer.n_defects,
        diameter,
    )

    return wafer


def parse_klarf_string(text: str) -> list[dict[str, Any]]:
    """Parse KLARF text content and return a list of wafer data dicts.

    Each dict contains:
    - ``x``, ``y``: lists of coordinates in **mm** from wafer centre
    - ``size``: list of defect sizes (raw KLARF units)
    - ``classcode``: list of classification codes
    - ``wafer_id``: wafer ID string
    - ``lot_id``: lot ID string
    - ``diameter_mm``: wafer diameter in mm

    Parameters
    ----------
    text : str
        Raw KLARF file content.

    Returns
    -------
    list of dict
        One dict per wafer found in the file.

    Raises
    ------
    ValueError
        If no defect data can be extracted.
    """
    # Try klarfkit first if available
    try:
        return _parse_with_klarfkit(text)
    except ImportError:
        pass

    return _parse_fallback(text)


# ------------------------------------------------------------------
# Fallback pure-Python KLARF parser
# ------------------------------------------------------------------


def _parse_fallback(text: str) -> list[dict[str, Any]]:
    """Parse KLARF 1.x format with built-in parser."""
    wafers: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    lot_id: str | None = None
    diameter_mm: float = 300.0
    in_defect_list = False
    defect_spec: list[str] = []  # column names from DefectRecordSpec
    defects: list[list[str]] = []

    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        # Strip trailing semicolons for field parsing
        clean = line.rstrip(';').strip()

        # -- Global fields (before any WaferID) --
        if clean.startswith('LotID'):
            lot_id = _extract_quoted_or_token(clean, 'LotID')

        elif clean.startswith('SampleSize'):
            # SampleSize <count> <diameter_um>
            parts = clean.split()
            if len(parts) >= 3:
                try:
                    diameter_um = float(parts[2])
                    diameter_mm = diameter_um / 1000.0
                except ValueError:
                    pass

        # -- Per-wafer fields --
        elif clean.startswith('WaferID'):
            # Start a new wafer section
            if current and defects:
                _finalize_wafer(current, defect_spec, defects, lot_id, diameter_mm)
                wafers.append(current)
            current = {}
            defects = []
            in_defect_list = False
            current['wafer_id'] = _extract_quoted_or_token(clean, 'WaferID')

        elif clean.startswith('DefectRecordSpec'):
            # DefectRecordSpec <n_columns> COL1 COL2 ...
            parts = clean.split()
            if len(parts) >= 3:
                try:
                    n_cols = int(parts[1])
                    defect_spec = [p.upper() for p in parts[2 : 2 + n_cols]]
                except ValueError:
                    defect_spec = [p.upper() for p in parts[2:]]

        elif clean == 'DefectList':
            in_defect_list = True

        elif clean.startswith('EndOfFile') or clean.startswith('End'):
            if in_defect_list:
                in_defect_list = False
                # Finalize this wafer
                if current or defects:
                    if 'wafer_id' not in current:
                        current['wafer_id'] = f'W{len(wafers) + 1:02d}'
                    _finalize_wafer(current, defect_spec, defects, lot_id, diameter_mm)
                    wafers.append(current)
                    current = {}
                    defects = []

        elif in_defect_list:
            # Defect data line
            parts = clean.split()
            if parts and _is_numeric(parts[0]):
                defects.append(parts)

    # Handle file that doesn't end with EndOfFile
    if current and defects:
        if 'wafer_id' not in current:
            current['wafer_id'] = f'W{len(wafers) + 1:02d}'
        _finalize_wafer(current, defect_spec, defects, lot_id, diameter_mm)
        wafers.append(current)

    if not wafers:
        raise ValueError(
            'Could not parse any defect data from KLARF content. '
            'Ensure the file contains DefectRecordSpec and DefectList sections.'
        )

    return wafers


def _finalize_wafer(
    current: dict[str, Any],
    defect_spec: list[str],
    defects: list[list[str]],
    lot_id: str | None,
    diameter_mm: float,
) -> None:
    """Populate current wafer dict from defect records."""
    current['lot_id'] = lot_id
    current['diameter_mm'] = diameter_mm

    # Map column names to indices
    col_idx = {name: i for i, name in enumerate(defect_spec)}

    x_col = col_idx.get('XREL')
    y_col = col_idx.get('YREL')
    size_col = col_idx.get('DEFECTSIZE')
    class_col = col_idx.get('CLASSNUMBER')

    xs, ys, sizes, classcodes = [], [], [], []

    for row in defects:
        if x_col is not None and y_col is not None:
            try:
                # KLARF coordinates are in µm — convert to mm
                x_um = float(row[x_col])
                y_um = float(row[y_col])
                xs.append(x_um / 1000.0)
                ys.append(y_um / 1000.0)
            except (IndexError, ValueError):
                continue
        else:
            continue  # can't extract coordinates

        if size_col is not None and size_col < len(row):
            try:
                sizes.append(float(row[size_col]))
            except ValueError:
                sizes.append(0.0)
        else:
            sizes.append(0.0)

        if class_col is not None and class_col < len(row):
            try:
                classcodes.append(int(row[class_col]))
            except ValueError:
                classcodes.append(0)
        else:
            classcodes.append(0)

    current['x'] = xs
    current['y'] = ys
    current['size'] = sizes if sizes else None
    current['classcode'] = classcodes if classcodes else None


def _extract_quoted_or_token(line: str, field_name: str) -> str:
    """Extract a quoted string or bare token after a field name."""
    # Try quoted first: FieldName "value"
    match = re.search(r'"([^"]*)"', line)
    if match:
        return match.group(1)
    # Otherwise take the next token
    parts = line.split()
    idx = 0
    for i, p in enumerate(parts):
        if p.upper() == field_name.upper():
            idx = i + 1
            break
    if idx < len(parts):
        return parts[idx].strip('"').strip("'")
    return ''


def _is_numeric(s: str) -> bool:
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False


# ------------------------------------------------------------------
# klarfkit-based parser (optional)
# ------------------------------------------------------------------


def _parse_with_klarfkit(text: str) -> list[dict[str, Any]]:
    """Parse KLARF using the klarfkit package (if installed)."""
    import klarfkit  # noqa: F811 — import guarded by try/except

    parsed = klarfkit.parse(text)
    wafers: list[dict[str, Any]] = []

    for wafer_data in parsed.get('wafers', [parsed]):
        defects = wafer_data.get('defects', [])
        xs, ys, sizes, classcodes = [], [], [], []
        for d in defects:
            xs.append(d.get('xrel', 0) / 1000.0)
            ys.append(d.get('yrel', 0) / 1000.0)
            sizes.append(d.get('defect_size', 0))
            classcodes.append(d.get('class_number', 0))

        diameter_um = wafer_data.get('sample_size', {}).get('diameter', 300000)
        wafers.append(
            {
                'x': xs,
                'y': ys,
                'size': sizes,
                'classcode': classcodes,
                'wafer_id': wafer_data.get('wafer_id', ''),
                'lot_id': wafer_data.get('lot_id', ''),
                'diameter_mm': diameter_um / 1000.0,
            }
        )

    return wafers

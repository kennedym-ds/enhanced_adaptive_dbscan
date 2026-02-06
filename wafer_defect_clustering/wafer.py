"""
Wafer geometry and defect data model for semiconductor wafer defect clustering.

Provides WaferMap, WaferGeometry, and ZoneDefinition classes that encode
semiconductor manufacturing domain knowledge no general clustering library has:
- Standard wafer sizes (200mm, 300mm, 450mm)
- Notch/flat orientation conventions
- Edge exclusion zones
- Zone ring definitions (center, middle, edge)
- Die grid mapping
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ['WaferGeometry', 'WaferMap', 'ZoneDefinition']


@dataclass(frozen=True)
class WaferGeometry:
    """Physical parameters of a semiconductor wafer.

    Parameters
    ----------
    diameter_mm : float
        Wafer diameter in millimetres (common: 200, 300, 450).
    edge_exclusion_mm : float
        Width of the edge exclusion zone where no dies are placed (typically 2-3 mm).
    notch_angle_deg : float
        Notch position in degrees (0 = 3 o'clock, 90 = 12 o'clock, 270 = 6 o'clock).
        Industry standard for 300 mm wafers is 270 (6 o'clock).
    flat_length_mm : float or None
        Length of the orientation flat (200 mm wafers use a flat instead of a notch).
        If *None*, a notch is assumed.
    shape : str
        ``'circular'`` (default) or ``'square'`` (rare, used in panel-level packaging).
    """

    diameter_mm: float = 300.0
    edge_exclusion_mm: float = 3.0
    notch_angle_deg: float = 270.0
    flat_length_mm: float | None = None
    shape: str = 'circular'

    @property
    def radius_mm(self) -> float:
        return self.diameter_mm / 2.0

    @property
    def usable_radius_mm(self) -> float:
        """Radius of the usable (non-excluded) area."""
        return self.radius_mm - self.edge_exclusion_mm

    @property
    def notch_angle_rad(self) -> float:
        return np.deg2rad(self.notch_angle_deg)


@dataclass
class ZoneDefinition:
    """Defines a named radial zone on the wafer.

    Parameters
    ----------
    name : str
        Human-readable zone name (e.g. ``'center'``, ``'middle'``, ``'edge'``).
    inner_frac : float
        Inner boundary as a fraction of the wafer radius (0.0 = centre).
    outer_frac : float
        Outer boundary as a fraction of the wafer radius (1.0 = edge).
    """

    name: str
    inner_frac: float
    outer_frac: float

    def contains_radius(self, r: np.ndarray, wafer_radius: float) -> np.ndarray:
        """Return boolean mask for points whose *r* falls inside this zone."""
        return (r >= self.inner_frac * wafer_radius) & (r < self.outer_frac * wafer_radius)


# Industry-standard three-zone partition
STANDARD_ZONES: list[ZoneDefinition] = [
    ZoneDefinition('center', 0.0, 0.33),
    ZoneDefinition('middle', 0.33, 0.66),
    ZoneDefinition('edge', 0.66, 1.0),
]


class WaferMap:
    """Semiconductor wafer with defect data and geometry-aware operations.

    This is the central domain object.  It holds physical wafer parameters,
    defect coordinates, and optional defect attributes (size, kill flag, layer,
    classification code).

    Parameters
    ----------
    diameter_mm : float
        Wafer diameter (default 300 mm).
    edge_exclusion_mm : float
        Edge exclusion zone width in mm (default 3.0).
    notch_angle_deg : float
        Notch orientation in degrees (default 270 = 6 o'clock).
    flat_length_mm : float or None
        Orientation flat length; ``None`` means a notch is used.
    shape : str
        ``'circular'`` or ``'square'``.

    Examples
    --------
    >>> wafer = WaferMap(diameter_mm=300, edge_exclusion_mm=3.0)
    >>> wafer.add_defects(x=[10, -20, 50], y=[5, 30, -10], size=[0.5, 1.2, 0.3])
    >>> wafer.n_defects
    3
    """

    def __init__(
        self,
        diameter_mm: float = 300.0,
        edge_exclusion_mm: float = 3.0,
        notch_angle_deg: float = 270.0,
        flat_length_mm: float | None = None,
        shape: str = 'circular',
    ) -> None:
        self.geometry = WaferGeometry(
            diameter_mm=diameter_mm,
            edge_exclusion_mm=edge_exclusion_mm,
            notch_angle_deg=notch_angle_deg,
            flat_length_mm=flat_length_mm,
            shape=shape,
        )

        # Defect storage — populated via add_defects()
        self._x: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._attrs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_klarf(
        cls,
        path: Any,
        *,
        wafer_id: int = 0,
    ) -> WaferMap:
        """Load a KLARF file and return a :class:`WaferMap`.

        Parameters
        ----------
        path : str or Path
            Path to the KLARF file.
        wafer_id : int
            Zero-based wafer index for multi-wafer files.

        Returns
        -------
        WaferMap
        """
        from .klarf import load_klarf

        return load_klarf(path, wafer_id=wafer_id)

    # ------------------------------------------------------------------
    # Defect data management
    # ------------------------------------------------------------------

    def add_defects(
        self,
        x: Any,
        y: Any,
        *,
        size: Any = None,
        kill: Any = None,
        layer: Any = None,
        classcode: Any = None,
        **extra_attrs: Any,
    ) -> None:
        """Add defect inspection data.

        Coordinates are in **millimetres from the wafer centre**.

        Parameters
        ----------
        x, y : array-like
            Defect coordinates (mm from centre).
        size : array-like, optional
            Defect size in µm² (or any consistent unit).
        kill : array-like, optional
            Kill flag — ``1`` (or ``True``) if the defect kills the die.
        layer : array-like, optional
            Process layer identifier (int or str).
        classcode : array-like, optional
            Defect classification code (int or str).
        **extra_attrs
            Any additional per-defect attributes.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.shape != y.shape:
            raise ValueError(f'x and y must have the same shape, got {x.shape} vs {y.shape}')

        if self._x is None:
            self._x = x
            self._y = y
        else:
            self._x = np.concatenate([self._x, x])
            self._y = np.concatenate([self._y, y])

        def _append(key: str, val: Any) -> None:
            if val is None:
                return
            arr = np.asarray(val)
            if arr.shape[0] != x.shape[0]:
                raise ValueError(
                    f'Attribute "{key}" length {arr.shape[0]} != coordinate length {x.shape[0]}'
                )
            if key in self._attrs:
                self._attrs[key] = np.concatenate([self._attrs[key], arr])
            else:
                self._attrs[key] = arr

        _append('size', size)
        _append('kill', kill)
        _append('layer', layer)
        _append('classcode', classcode)
        for k, v in extra_attrs.items():
            _append(k, v)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def coordinates(self) -> np.ndarray:
        """``(N, 2)`` array of defect ``[x, y]`` positions in mm."""
        if self._x is None:
            return np.empty((0, 2), dtype=float)
        return np.column_stack([self._x, self._y])

    @property
    def n_defects(self) -> int:
        return 0 if self._x is None else len(self._x)

    @property
    def attribute_names(self) -> list[str]:
        """Names of stored defect attributes (excluding x, y)."""
        return list(self._attrs.keys())

    def get_attribute(self, name: str) -> np.ndarray:
        """Return a defect attribute array by name."""
        if name not in self._attrs:
            raise KeyError(f'Unknown attribute "{name}". Available: {self.attribute_names}')
        return self._attrs[name]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def radii(self) -> np.ndarray:
        """Distance of each defect from wafer centre (mm)."""
        coords = self.coordinates
        if len(coords) == 0:
            return np.empty(0, dtype=float)
        return np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)

    @property
    def angles(self) -> np.ndarray:
        """Angle of each defect from wafer centre in radians (``atan2(y, x)``)."""
        coords = self.coordinates
        if len(coords) == 0:
            return np.empty(0, dtype=float)
        return np.arctan2(coords[:, 1], coords[:, 0])

    def distance_to_edge(self) -> np.ndarray:
        """Distance from each defect to the nearest wafer edge (mm).

        For circular wafers: ``radius - r``.  Negative means outside the wafer.
        """
        if self.geometry.shape == 'circular':
            return self.geometry.radius_mm - self.radii
        elif self.geometry.shape == 'square':
            coords = self.coordinates
            half = self.geometry.diameter_mm / 2.0
            dx = half - np.abs(coords[:, 0])
            dy = half - np.abs(coords[:, 1])
            return np.minimum(dx, dy)
        else:
            raise ValueError(f'Unsupported shape: {self.geometry.shape}')

    def is_inside_wafer(self, x: Any = None, y: Any = None) -> np.ndarray:
        """Check whether coordinates fall inside the wafer boundary.

        If *x* and *y* are ``None``, checks stored defects.
        """
        if x is None and y is None:
            coords = self.coordinates
        else:
            coords = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])

        if self.geometry.shape == 'circular':
            r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
            return r <= self.geometry.radius_mm
        elif self.geometry.shape == 'square':
            half = self.geometry.diameter_mm / 2.0
            return (np.abs(coords[:, 0]) <= half) & (np.abs(coords[:, 1]) <= half)
        else:
            raise ValueError(f'Unsupported shape: {self.geometry.shape}')

    # ------------------------------------------------------------------
    # Zone operations
    # ------------------------------------------------------------------

    def get_zone_mask(
        self,
        zone: str = 'edge',
        ring_width_mm: float | None = None,
        zones: list[ZoneDefinition] | None = None,
    ) -> np.ndarray:
        """Boolean mask selecting defects in a named zone.

        Parameters
        ----------
        zone : str
            Zone name: ``'center'``, ``'middle'``, ``'edge'``, or ``'notch'``.
        ring_width_mm : float, optional
            If given, overrides the zone definition to select defects within
            *ring_width_mm* of the wafer edge (only used when ``zone='edge'``).
        zones : list of ZoneDefinition, optional
            Custom zone definitions.  Defaults to ``STANDARD_ZONES``.
        """
        r = self.radii
        radius = self.geometry.radius_mm

        if zone == 'notch':
            # ±15° arc centred on the notch angle
            angle_diff = np.abs(
                np.angle(np.exp(1j * (self.angles - self.geometry.notch_angle_rad)))
            )
            return (angle_diff < np.deg2rad(15)) & (r > 0.8 * radius)

        if zone == 'edge' and ring_width_mm is not None:
            return (radius - r) <= ring_width_mm

        defs = zones or STANDARD_ZONES
        for zd in defs:
            if zd.name == zone:
                return zd.contains_radius(r, radius)

        raise ValueError(f'Unknown zone "{zone}". Available: {[z.name for z in defs]}')

    def get_zone_label(self, zones: list[ZoneDefinition] | None = None) -> np.ndarray:
        """Return a string label for each defect's zone (``'center'``, ``'middle'``, ``'edge'``)."""
        defs = zones or STANDARD_ZONES
        r = self.radii
        radius = self.geometry.radius_mm
        labels = np.full(self.n_defects, 'unknown', dtype=object)
        for zd in defs:
            mask = zd.contains_radius(r, radius)
            labels[mask] = zd.name
        return labels

    # ------------------------------------------------------------------
    # Die-level aggregation
    # ------------------------------------------------------------------

    def to_die_map(
        self,
        die_size_mm: tuple[float, float] = (10.0, 10.0),
        origin: str = 'center',
    ) -> pd.DataFrame:
        """Aggregate defects to die-level counts.

        Parameters
        ----------
        die_size_mm : tuple of (width, height)
            Die dimensions in mm.
        origin : str
            ``'center'`` (default) — die grid centred on wafer centre.

        Returns
        -------
        DataFrame with columns ``die_row``, ``die_col``, ``count``, ``x_center``, ``y_center``.
        """
        coords = self.coordinates
        if len(coords) == 0:
            return pd.DataFrame(columns=['die_row', 'die_col', 'count', 'x_center', 'y_center'])

        dx, dy = die_size_mm
        col = np.floor(coords[:, 0] / dx).astype(int)
        row = np.floor(coords[:, 1] / dy).astype(int)

        df = pd.DataFrame({'die_row': row, 'die_col': col})
        counts = df.groupby(['die_row', 'die_col']).size().reset_index(name='count')
        counts['x_center'] = (counts['die_col'] + 0.5) * dx
        counts['y_center'] = (counts['die_row'] + 0.5) * dy
        return counts

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return all defect data as a DataFrame."""
        if self.n_defects == 0:
            return pd.DataFrame(columns=['x', 'y'])
        data: dict[str, Any] = {'x': self._x, 'y': self._y}
        data.update(self._attrs)
        return pd.DataFrame(data)

    def __repr__(self) -> str:
        return (
            f'WaferMap(diameter={self.geometry.diameter_mm}mm, '
            f'n_defects={self.n_defects}, '
            f'shape={self.geometry.shape!r})'
        )

"""
Semiconductor wafer defect visualization with Plotly.

Provides interactive wafer map plots with cluster colouring, zone overlays,
die grids, notch markers, and defect pattern annotations.  All functions
return ``plotly.graph_objects.Figure`` objects that can be displayed in
Jupyter notebooks, saved to HTML, or served via Dash/Flask.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .patterns import PatternResult
from .wafer import WaferMap

logger = logging.getLogger(__name__)

__all__ = [
    'plot_wafer_map',
    'plot_wafer_grid',
    'plot_pattern_summary',
    'plot_radial_distribution',
    'plot_cluster_details',
]


# Colour palette for clusters (distinct, colourblind-friendly-ish)
_CLUSTER_COLORS = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
    '#aec7e8',
    '#ffbb78',
    '#98df8a',
    '#ff9896',
    '#c5b0d5',
    '#c49c94',
    '#f7b6d2',
    '#c7c7c7',
    '#dbdb8d',
    '#9edae5',
]


def _wafer_boundary_trace(wafer: WaferMap, **kwargs: Any) -> go.Scatter:
    """Create a circular wafer boundary trace."""
    theta = np.linspace(0, 2 * np.pi, 200)
    r = wafer.geometry.radius_mm
    return go.Scatter(
        x=r * np.cos(theta),
        y=r * np.sin(theta),
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False,
        hoverinfo='skip',
        **kwargs,
    )


def _edge_exclusion_trace(wafer: WaferMap) -> go.Scatter:
    """Dashed circle showing the edge exclusion zone boundary."""
    theta = np.linspace(0, 2 * np.pi, 200)
    r = wafer.geometry.usable_radius_mm
    return go.Scatter(
        x=r * np.cos(theta),
        y=r * np.sin(theta),
        mode='lines',
        line=dict(color='gray', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip',
    )


def _notch_trace(wafer: WaferMap) -> go.Scatter:
    """Small triangle marking the notch location."""
    angle = wafer.geometry.notch_angle_rad
    r = wafer.geometry.radius_mm
    size = r * 0.03  # notch size relative to wafer

    cx, cy = r * np.cos(angle), r * np.sin(angle)
    perp = angle + np.pi / 2

    points_x = [
        cx - size * np.cos(angle),
        cx + size * np.cos(perp),
        cx - size * np.cos(perp),
        cx - size * np.cos(angle),  # close triangle
    ]
    points_y = [
        cy - size * np.sin(angle),
        cy + size * np.sin(perp),
        cy - size * np.sin(perp),
        cy - size * np.sin(angle),
    ]
    return go.Scatter(
        x=points_x,
        y=points_y,
        mode='lines',
        fill='toself',
        fillcolor='black',
        line=dict(color='black', width=1),
        showlegend=False,
        hoverinfo='skip',
    )


def _zone_ring_traces(wafer: WaferMap) -> list[go.Scatter]:
    """Dashed zone ring overlays (center/middle/edge boundaries)."""
    from .wafer import STANDARD_ZONES

    traces = []
    theta = np.linspace(0, 2 * np.pi, 200)
    r = wafer.geometry.radius_mm
    for zd in STANDARD_ZONES:
        if zd.outer_frac < 1.0:
            ring_r = zd.outer_frac * r
            traces.append(
                go.Scatter(
                    x=ring_r * np.cos(theta),
                    y=ring_r * np.sin(theta),
                    mode='lines',
                    line=dict(color='lightgray', width=0.5, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip',
                )
            )
    return traces


def plot_wafer_map(
    wafer: WaferMap,
    labels: np.ndarray | None = None,
    *,
    show_zones: bool = False,
    show_dies: bool = False,
    show_edge_exclusion: bool = True,
    show_notch: bool = True,
    title: str = 'Wafer Defect Map',
    marker_size: int = 4,
    width: int = 700,
    height: int = 700,
) -> go.Figure:
    """Interactive Plotly wafer map with cluster colouring.

    Parameters
    ----------
    wafer : WaferMap
        Wafer with defect data.
    labels : np.ndarray, optional
        Cluster labels.  ``-1`` = noise (shown in light gray).
        If ``None``, all defects are coloured uniformly.
    show_zones : bool
        Overlay zone ring boundaries.
    show_dies : bool
        Overlay a die grid (10 × 10 mm default).
    show_edge_exclusion : bool
        Show the edge exclusion zone boundary.
    show_notch : bool
        Show the notch marker.
    title : str
        Plot title.
    marker_size : int
        Defect marker size in pixels.
    width, height : int
        Figure dimensions.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    fig = go.Figure()

    # Wafer boundary
    fig.add_trace(_wafer_boundary_trace(wafer))

    if show_edge_exclusion:
        fig.add_trace(_edge_exclusion_trace(wafer))
    if show_notch:
        fig.add_trace(_notch_trace(wafer))
    if show_zones:
        for t in _zone_ring_traces(wafer):
            fig.add_trace(t)

    # Die grid
    if show_dies:
        _add_die_grid(fig, wafer)

    # Defects
    coords = wafer.coordinates
    if len(coords) == 0:
        _apply_wafer_layout(fig, wafer, title, width, height)
        return fig

    if labels is None:
        fig.add_trace(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode='markers',
                marker=dict(size=marker_size, color='steelblue'),
                name='defects',
                hovertemplate='x=%{x:.1f}mm<br>y=%{y:.1f}mm',
            )
        )
    else:
        unique_labels = sorted(set(labels))
        for cid in unique_labels:
            mask = labels == cid
            pts = coords[mask]
            if cid == -1:
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode='markers',
                        marker=dict(size=marker_size - 1, color='lightgray', opacity=0.5),
                        name='noise',
                        hovertemplate='noise<br>x=%{x:.1f}mm<br>y=%{y:.1f}mm',
                    )
                )
            else:
                color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode='markers',
                        marker=dict(size=marker_size, color=color),
                        name=f'cluster {cid}',
                        hovertemplate=f'cluster {cid}<br>x=%{{x:.1f}}mm<br>y=%{{y:.1f}}mm',
                    )
                )

    _apply_wafer_layout(fig, wafer, title, width, height)
    return fig


def plot_wafer_grid(
    wafers: list[WaferMap],
    labels_list: list[np.ndarray],
    *,
    ncols: int = 5,
    title: str = 'Lot Wafer Map Summary',
    subplot_size: int = 200,
) -> go.Figure:
    """Grid of mini wafer maps for lot-level overview.

    Parameters
    ----------
    wafers : list of WaferMap
    labels_list : list of np.ndarray
        One label array per wafer.
    ncols : int
        Number of columns in the grid.
    title : str
    subplot_size : int
        Size of each sub-plot in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    n = len(wafers)
    nrows = int(np.ceil(n / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f'Wafer {i}' for i in range(n)],
        horizontal_spacing=0.02,
        vertical_spacing=0.05,
    )

    for idx, (wafer, labels) in enumerate(zip(wafers, labels_list)):
        row = idx // ncols + 1
        col = idx % ncols + 1

        # Boundary
        theta = np.linspace(0, 2 * np.pi, 100)
        r = wafer.geometry.radius_mm
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(theta),
                y=r * np.sin(theta),
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=row,
            col=col,
        )

        # Defects
        coords = wafer.coordinates
        if len(coords) > 0:
            unique_labels = sorted(set(labels))
            for cid in unique_labels:
                mask = labels == cid
                pts = coords[mask]
                color = 'lightgray' if cid == -1 else _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        mode='markers',
                        marker=dict(size=2, color=color),
                        showlegend=False,
                        hoverinfo='skip',
                    ),
                    row=row,
                    col=col,
                )

        fig.update_xaxes(
            scaleanchor=f'y{idx + 1}', showticklabels=False, showgrid=False, row=row, col=col
        )
        fig.update_yaxes(showticklabels=False, showgrid=False, row=row, col=col)

    fig.update_layout(
        title_text=title,
        width=ncols * subplot_size,
        height=nrows * subplot_size,
        plot_bgcolor='white',
    )
    return fig


def plot_pattern_summary(
    pattern_results: dict[int, PatternResult],
    *,
    title: str = 'Defect Pattern Classification',
) -> go.Figure:
    """Bar chart of pattern types with confidence scores.

    Parameters
    ----------
    pattern_results : dict
        Output of ``DefectPatternClassifier.classify_all()`` or
        ``WaferClusterer.pattern_results_``.
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if not pattern_results:
        fig = go.Figure()
        fig.update_layout(
            title=title,
            annotations=[
                dict(
                    text='No clusters to classify',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.5,
                    y=0.5,
                )
            ],
        )
        return fig

    cluster_ids = sorted(pattern_results.keys())
    types = [pattern_results[c].pattern_type for c in cluster_ids]
    confs = [pattern_results[c].confidence for c in cluster_ids]
    colors = [_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)] for c in cluster_ids]

    fig = go.Figure(
        go.Bar(
            x=[f'Cluster {c}' for c in cluster_ids],
            y=confs,
            text=types,
            textposition='outside',
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=title,
        yaxis_title='Confidence',
        yaxis_range=[0, 1.1],
        xaxis_title='Cluster',
        plot_bgcolor='white',
    )
    return fig


def plot_radial_distribution(
    wafer: WaferMap,
    labels: np.ndarray | None = None,
    *,
    n_bins: int = 30,
    title: str = 'Radial Defect Distribution',
) -> go.Figure:
    """Histogram of defect distance from wafer centre.

    Useful for identifying ring and edge patterns.

    Parameters
    ----------
    wafer : WaferMap
    labels : np.ndarray, optional
        Cluster labels for colour-coding.
    n_bins : int
    title : str

    Returns
    -------
    plotly.graph_objects.Figure
    """
    radii = wafer.radii

    fig = go.Figure()
    if labels is None:
        fig.add_trace(go.Histogram(x=radii, nbinsx=n_bins, name='all defects'))
    else:
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            mask = labels == cid
            color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
            fig.add_trace(
                go.Histogram(
                    x=radii[mask],
                    nbinsx=n_bins,
                    name=f'cluster {cid}',
                    marker_color=color,
                    opacity=0.7,
                )
            )
        fig.update_layout(barmode='overlay')

    fig.update_layout(
        title=title,
        xaxis_title='Distance from centre (mm)',
        yaxis_title='Count',
        plot_bgcolor='white',
    )
    return fig


def plot_cluster_details(
    wafer: WaferMap,
    labels: np.ndarray,
    cluster_id: int,
    *,
    pattern_result: PatternResult | None = None,
    title: str | None = None,
    width: int = 600,
    height: int = 600,
) -> go.Figure:
    """Detailed view of a single cluster with pattern overlay.

    Shows the cluster defects on the wafer, with the fitted pattern geometry
    overlaid (line for scratch, circle for ring, etc.).

    Parameters
    ----------
    wafer : WaferMap
    labels : np.ndarray
    cluster_id : int
    pattern_result : PatternResult, optional
    title : str, optional
    width, height : int

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if title is None:
        pat_label = pattern_result.pattern_type if pattern_result else '?'
        title = f'Cluster {cluster_id} — {pat_label}'

    fig = go.Figure()
    fig.add_trace(_wafer_boundary_trace(wafer))

    coords = wafer.coordinates
    mask = labels == cluster_id
    other = ~mask & (labels != -1)

    # Background clusters (faint)
    if other.any():
        fig.add_trace(
            go.Scatter(
                x=coords[other, 0],
                y=coords[other, 1],
                mode='markers',
                marker=dict(size=2, color='lightgray', opacity=0.3),
                showlegend=False,
                hoverinfo='skip',
            )
        )

    # Target cluster
    pts = coords[mask]
    color = _CLUSTER_COLORS[cluster_id % len(_CLUSTER_COLORS)]
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0],
            y=pts[:, 1],
            mode='markers',
            marker=dict(size=6, color=color),
            name=f'cluster {cluster_id}',
            hovertemplate='x=%{x:.1f}mm<br>y=%{y:.1f}mm',
        )
    )

    # Pattern overlay
    if pattern_result is not None:
        _add_pattern_overlay(fig, pts, pattern_result)

    _apply_wafer_layout(fig, wafer, title, width, height)
    return fig


# ======================================================================
# Layout and annotation helpers
# ======================================================================


def _apply_wafer_layout(
    fig: go.Figure, wafer: WaferMap, title: str, width: int, height: int
) -> None:
    r = wafer.geometry.radius_mm * 1.08
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis=dict(range=[-r, r], scaleanchor='y', showgrid=False, zeroline=False),
        yaxis=dict(range=[-r, r], showgrid=False, zeroline=False),
        plot_bgcolor='white',
    )


def _add_die_grid(fig: go.Figure, wafer: WaferMap, die_size_mm: tuple = (10.0, 10.0)) -> None:
    """Add a die-grid overlay (clipped to wafer boundary)."""
    dx, dy = die_size_mm
    r = wafer.geometry.radius_mm
    for x in np.arange(-r, r + dx, dx):
        fig.add_shape(type='line', x0=x, y0=-r, x1=x, y1=r, line=dict(color='lightblue', width=0.3))
    for y in np.arange(-r, r + dy, dy):
        fig.add_shape(type='line', x0=-r, y0=y, x1=r, y1=y, line=dict(color='lightblue', width=0.3))


def _add_pattern_overlay(fig: go.Figure, points: np.ndarray, pr: PatternResult) -> None:
    """Overlay fitted geometry for the classified pattern."""
    d = pr.details

    if pr.pattern_type == 'scratch':
        # Overlay the fitted line
        angle_rad = np.radians(d.get('angle_deg', 0))
        cx, cy = points.mean(axis=0)
        half_len = d.get('length_mm', 10) / 2
        x0 = cx - half_len * np.cos(angle_rad)
        y0 = cy - half_len * np.sin(angle_rad)
        x1 = cx + half_len * np.cos(angle_rad)
        y1 = cy + half_len * np.sin(angle_rad)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                showlegend=False,
            )
        )

    elif pr.pattern_type == 'ring':
        mean_r = d.get('mean_radius_mm', 50)
        theta = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(
            go.Scatter(
                x=mean_r * np.cos(theta),
                y=mean_r * np.sin(theta),
                mode='lines',
                line=dict(color='red', width=1.5, dash='dash'),
                showlegend=False,
            )
        )

    elif pr.pattern_type == 'center_spot':
        max_r = d.get('max_radius_mm', 20)
        theta = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(
            go.Scatter(
                x=max_r * np.cos(theta),
                y=max_r * np.sin(theta),
                mode='lines',
                line=dict(color='red', width=1.5, dash='dot'),
                showlegend=False,
            )
        )

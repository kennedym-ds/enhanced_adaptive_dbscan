# enhanced_adaptive_dbscan/density_engine.py

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class DensityRegion:
    """Represents a density region with its characteristics."""
    region_id: int
    density_type: str  # 'low', 'medium', 'high'
    points: np.ndarray
    indices: np.ndarray
    relative_density: float
    boundary_points: np.ndarray
    region_center: np.ndarray
    stability_score: float = 0.0

@dataclass
class DensityAnalysis:
    """Complete density analysis results."""
    regions: List[DensityRegion]
    density_histogram: np.ndarray
    density_thresholds: Dict[str, float]
    global_density_stats: Dict[str, float]
    density_map: np.ndarray  # Per-point density values
    region_assignments: np.ndarray  # Per-point region assignments

class RelativeDensityComputer:
    """
    Computes relative density distribution and partitions data into density regions.
    
    This is inspired by MDBSCAN's multi-density approach, where different regions
    of the dataset may have vastly different density characteristics.
    """
    
    def __init__(self, k: int = 20, density_bins: int = 50, 
                 low_density_threshold: float = 0.3, 
                 high_density_threshold: float = 0.7):
        """
        Initialize the relative density computer.
        
        Parameters:
        - k: Number of neighbors for density estimation
        - density_bins: Number of bins for density histogram
        - low_density_threshold: Percentile threshold for low density regions
        - high_density_threshold: Percentile threshold for high density regions
        """
        self.k = k
        self.density_bins = density_bins
        self.low_density_threshold = low_density_threshold
        self.high_density_threshold = high_density_threshold
        
    def compute_point_densities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute local density for each point using k-NN distances.
        
        Parameters:
        - X: Data points (n_samples, n_features)
        
        Returns:
        - densities: Local density for each point
        """
        n_samples = X.shape[0]
        tree = KDTree(X)
        k = min(self.k + 1, n_samples)
        
        # Get k nearest neighbors (including self)
        distances, indices = tree.query(X, k=k)
        
        if k > 1:
            # Exclude the point itself (distance 0)
            mean_distances = np.mean(distances[:, 1:], axis=1)
        else:
            mean_distances = distances[:, 0]
            
        # Convert to density (inverse of distance)
        densities = 1.0 / (mean_distances + 1e-8)
        
        return densities
    
    def compute_relative_densities(self, densities: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Compute relative densities and determine thresholds.
        
        Parameters:
        - densities: Raw density values
        
        Returns:
        - relative_densities: Normalized density values [0,1]
        - thresholds: Dictionary with density threshold values
        """
        # Normalize densities to [0, 1] range
        min_density = np.min(densities)
        max_density = np.max(densities)
        density_range = max_density - min_density
        
        if density_range > 1e-8:
            relative_densities = (densities - min_density) / density_range
        else:
            relative_densities = np.ones_like(densities) * 0.5
            
        # Compute threshold values based on percentiles
        low_threshold = np.percentile(relative_densities, self.low_density_threshold * 100)
        high_threshold = np.percentile(relative_densities, self.high_density_threshold * 100)
        
        thresholds = {
            'low': low_threshold,
            'high': high_threshold,
            'min': np.min(relative_densities),
            'max': np.max(relative_densities),
            'mean': np.mean(relative_densities),
            'std': np.std(relative_densities)
        }
        
        return relative_densities, thresholds
    
    def create_density_histogram(self, relative_densities: np.ndarray) -> np.ndarray:
        """
        Create histogram of density distribution.
        
        Parameters:
        - relative_densities: Normalized density values
        
        Returns:
        - histogram: Density histogram
        """
        histogram, _ = np.histogram(relative_densities, bins=self.density_bins, range=(0, 1))
        return histogram
    
    def partition_by_density(self, X: np.ndarray, relative_densities: np.ndarray, 
                           thresholds: Dict[str, float]) -> List[DensityRegion]:
        """
        Partition data into density regions.
        
        Parameters:
        - X: Data points
        - relative_densities: Normalized density values
        - thresholds: Density threshold values
        
        Returns:
        - regions: List of density regions
        """
        regions = []
        
        # Define region types based on thresholds
        low_mask = relative_densities <= thresholds['low']
        high_mask = relative_densities >= thresholds['high']
        medium_mask = ~(low_mask | high_mask)
        
        region_masks = [
            ('low', low_mask),
            ('medium', medium_mask), 
            ('high', high_mask)
        ]
        
        region_id = 0
        for density_type, mask in region_masks:
            if np.any(mask):
                indices = np.where(mask)[0]
                points = X[indices]
                region_densities = relative_densities[indices]
                
                # Compute region statistics
                avg_density = np.mean(region_densities)
                region_center = np.mean(points, axis=0)
                
                # Find boundary points (points with neighbors in other regions)
                boundary_indices = self._find_boundary_points(X, indices, mask)
                boundary_points = X[boundary_indices] if len(boundary_indices) > 0 else np.array([])
                
                region = DensityRegion(
                    region_id=region_id,
                    density_type=density_type,
                    points=points,
                    indices=indices,
                    relative_density=avg_density,
                    boundary_points=boundary_points,
                    region_center=region_center
                )
                
                regions.append(region)
                region_id += 1
                
        return regions
    
    def _find_boundary_points(self, X: np.ndarray, region_indices: np.ndarray, 
                            region_mask: np.ndarray) -> np.ndarray:
        """
        Find boundary points between density regions.
        
        Parameters:
        - X: All data points
        - region_indices: Indices of points in current region
        - region_mask: Boolean mask for current region
        
        Returns:
        - boundary_indices: Indices of boundary points
        """
        if len(region_indices) == 0:
            return np.array([], dtype=int)
            
        tree = KDTree(X)
        boundary_indices = []
        
        # For each point in the region, check if it has neighbors outside the region
        for idx in region_indices:
            # Find neighbors
            distances, neighbor_indices = tree.query([X[idx]], k=min(self.k + 1, len(X)))
            neighbor_indices = neighbor_indices[0][1:]  # Exclude self
            
            # Check if any neighbors are outside the region
            if np.any(~region_mask[neighbor_indices]):
                boundary_indices.append(idx)
                
        return np.array(boundary_indices, dtype=int)


class DynamicBoundaryManager:
    """
    Manages boundary points between density regions and handles
    cross-region cluster assignments.
    """
    
    def __init__(self, boundary_tolerance: float = 0.1):
        """
        Initialize boundary manager.
        
        Parameters:
        - boundary_tolerance: Tolerance for boundary point assignment
        """
        self.boundary_tolerance = boundary_tolerance
        
    def analyze_boundaries(self, regions: List[DensityRegion], X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze boundary characteristics between regions.
        
        Parameters:
        - regions: List of density regions
        - X: All data points
        
        Returns:
        - boundary_analysis: Dictionary with boundary information
        """
        boundary_analysis = {
            'inter_region_distances': {},
            'boundary_point_counts': {},
            'transition_zones': {}
        }
        
        # Compute inter-region characteristics
        for i, region_i in enumerate(regions):
            for j, region_j in enumerate(regions):
                if i < j:  # Avoid duplicate pairs
                    key = f"{region_i.density_type}_{region_j.density_type}"
                    
                    # Compute minimum distance between regions
                    if len(region_i.points) > 0 and len(region_j.points) > 0:
                        distances = np.linalg.norm(
                            region_i.points[:, np.newaxis] - region_j.points[np.newaxis, :], 
                            axis=2
                        )
                        min_distance = np.min(distances)
                        boundary_analysis['inter_region_distances'][key] = min_distance
                        
                        # Count boundary points
                        boundary_count = len(region_i.boundary_points) + len(region_j.boundary_points)
                        boundary_analysis['boundary_point_counts'][key] = boundary_count
        
        return boundary_analysis
    
    def assign_boundary_points(self, boundary_points: np.ndarray, regions: List[DensityRegion]) -> np.ndarray:
        """
        Assign boundary points to appropriate regions.
        
        Parameters:
        - boundary_points: Points on region boundaries
        - regions: Available regions
        
        Returns:
        - assignments: Region assignments for boundary points
        """
        if len(boundary_points) == 0 or len(regions) == 0:
            return np.array([], dtype=int)
            
        assignments = np.full(len(boundary_points), -1, dtype=int)
        
        for i, point in enumerate(boundary_points):
            best_region = -1
            min_distance = float('inf')
            
            # Find closest region center
            for region in regions:
                if len(region.points) > 0:
                    distance = np.linalg.norm(point - region.region_center)
                    if distance < min_distance:
                        min_distance = distance
                        best_region = region.region_id
                        
            assignments[i] = best_region
            
        return assignments


class MultiScaleDensityEngine:
    """
    Unified multi-scale density analysis engine that combines:
    - Relative density computation (MDBSCAN-inspired)
    - Dynamic boundary management
    - Foundation for RL parameter optimization
    - Enhanced stability analysis
    
    This serves as the core multiplier component enabling all subsequent
    clustering improvements to work synergistically.
    """
    
    def __init__(self, 
                 k: int = 20,
                 density_bins: int = 50,
                 low_density_threshold: float = 0.3,
                 high_density_threshold: float = 0.7,
                 boundary_tolerance: float = 0.1,
                 enable_stability_analysis: bool = True):
        """
        Initialize the Multi-Scale Density Engine.
        
        Parameters:
        - k: Number of neighbors for density estimation
        - density_bins: Number of bins for density histogram
        - low_density_threshold: Percentile threshold for low density regions
        - high_density_threshold: Percentile threshold for high density regions
        - boundary_tolerance: Tolerance for boundary point assignment
        - enable_stability_analysis: Whether to compute stability metrics
        """
        self.k = k
        self.density_bins = density_bins
        self.low_density_threshold = low_density_threshold
        self.high_density_threshold = high_density_threshold
        self.boundary_tolerance = boundary_tolerance
        self.enable_stability_analysis = enable_stability_analysis
        
        # Initialize components
        self.density_computer = RelativeDensityComputer(
            k=k, 
            density_bins=density_bins,
            low_density_threshold=low_density_threshold,
            high_density_threshold=high_density_threshold
        )
        
        self.boundary_manager = DynamicBoundaryManager(
            boundary_tolerance=boundary_tolerance
        )
        
        # Cache for performance
        self._last_analysis = None
        self._cache_key = None
        
    def analyze_density_landscape(self, X: np.ndarray, force_recompute: bool = False) -> DensityAnalysis:
        """
        Perform comprehensive multi-scale density analysis.
        
        Parameters:
        - X: Data points (n_samples, n_features)
        - force_recompute: Whether to force recomputation even if cached
        
        Returns:
        - analysis: Complete density analysis results
        """
        # Check cache
        cache_key = hash(X.tobytes()) if X.size > 0 else 0
        if not force_recompute and self._cache_key == cache_key and self._last_analysis is not None:
            logger.debug("Using cached density analysis")
            return self._last_analysis
            
        logger.info(f"Computing density landscape for {len(X)} points")
        
        # Step 1: Compute point densities
        densities = self.density_computer.compute_point_densities(X)
        
        # Step 2: Compute relative densities and thresholds
        relative_densities, thresholds = self.density_computer.compute_relative_densities(densities)
        
        # Step 3: Create density histogram
        histogram = self.density_computer.create_density_histogram(relative_densities)
        
        # Step 4: Partition into density regions
        regions = self.density_computer.partition_by_density(X, relative_densities, thresholds)
        
        # Step 5: Analyze boundaries
        boundary_analysis = self.boundary_manager.analyze_boundaries(regions, X)
        
        # Step 6: Create region assignments array
        region_assignments = np.full(len(X), -1, dtype=int)
        for region in regions:
            region_assignments[region.indices] = region.region_id
            
        # Step 7: Compute global statistics
        global_stats = {
            'mean_density': np.mean(relative_densities),
            'std_density': np.std(relative_densities),
            'min_density': np.min(relative_densities),
            'max_density': np.max(relative_densities),
            'density_range': np.max(relative_densities) - np.min(relative_densities),
            'num_regions': len(regions),
            'boundary_analysis': boundary_analysis
        }
        
        # Create analysis result
        analysis = DensityAnalysis(
            regions=regions,
            density_histogram=histogram,
            density_thresholds=thresholds,
            global_density_stats=global_stats,
            density_map=relative_densities,
            region_assignments=region_assignments
        )
        
        # Cache results
        self._last_analysis = analysis
        self._cache_key = cache_key
        
        logger.info(f"Density analysis complete: {len(regions)} regions identified")
        for region in regions:
            logger.debug(f"Region {region.region_id} ({region.density_type}): "
                        f"{len(region.points)} points, density={region.relative_density:.3f}")
        
        return analysis
    
    def get_region_specific_parameters(self, analysis: DensityAnalysis, base_eps: float, base_min_pts: int) -> Dict[int, Dict[str, float]]:
        """
        Generate region-specific clustering parameters.
        
        Parameters:
        - analysis: Density analysis results
        - base_eps: Base epsilon value
        - base_min_pts: Base minimum points value
        
        Returns:
        - parameters: Dictionary mapping region_id to parameters
        """
        parameters = {}
        
        for region in analysis.regions:
            # Adjust parameters based on density characteristics
            if region.density_type == 'low':
                # For low density regions, use larger epsilon and smaller min_pts
                eps_multiplier = 1.5
                min_pts_multiplier = 0.7
            elif region.density_type == 'high':
                # For high density regions, use smaller epsilon and larger min_pts  
                eps_multiplier = 0.7
                min_pts_multiplier = 1.3
            else:  # medium
                # For medium density regions, use standard parameters
                eps_multiplier = 1.0
                min_pts_multiplier = 1.0
                
            parameters[region.region_id] = {
                'eps': base_eps * eps_multiplier,
                'min_pts': max(1, int(base_min_pts * min_pts_multiplier)),
                'density_type': region.density_type,
                'relative_density': region.relative_density
            }
            
        return parameters
    
    def compute_stability_metrics(self, analysis: DensityAnalysis, X: np.ndarray) -> Dict[int, float]:
        """
        Compute stability metrics for each density region.
        
        Parameters:
        - analysis: Density analysis results
        - X: Original data points
        
        Returns:
        - stability_scores: Dictionary mapping region_id to stability score
        """
        if not self.enable_stability_analysis:
            return {}
            
        stability_scores = {}
        
        for region in analysis.regions:
            if len(region.points) < 3:
                stability_scores[region.region_id] = 0.0
                continue
                
            # Compute intra-region variance
            distances = np.linalg.norm(region.points - region.region_center, axis=1)
            intra_variance = np.var(distances)
            
            # Compute boundary stability (fewer boundary points = more stable)
            boundary_ratio = len(region.boundary_points) / len(region.points) if len(region.points) > 0 else 1.0
            
            # Combine metrics (higher is more stable)
            stability = 1.0 / (1.0 + intra_variance) * (1.0 - boundary_ratio)
            stability_scores[region.region_id] = max(0.0, min(1.0, stability))
            
        return stability_scores
        
    def get_analysis_summary(self, analysis: DensityAnalysis) -> str:
        """
        Generate a human-readable summary of the density analysis.
        
        Parameters:
        - analysis: Density analysis results
        
        Returns:
        - summary: Text summary of the analysis
        """
        stats = analysis.global_density_stats
        regions_by_type = defaultdict(int)
        
        for region in analysis.regions:
            regions_by_type[region.density_type] += 1
            
        summary = f"""
Multi-Scale Density Analysis Summary:
=====================================
Total Points: {len(analysis.density_map)}
Density Range: {stats['min_density']:.3f} - {stats['max_density']:.3f}
Mean Density: {stats['mean_density']:.3f} (±{stats['std_density']:.3f})

Regions Identified: {stats['num_regions']}
  - Low Density: {regions_by_type['low']} regions
  - Medium Density: {regions_by_type['medium']} regions  
  - High Density: {regions_by_type['high']} regions

Density Thresholds:
  - Low: {analysis.density_thresholds['low']:.3f}
  - High: {analysis.density_thresholds['high']:.3f}
"""
        return summary.strip()

# enhanced_adaptive_dbscan/cluster_quality_analyzer.py

"""
Cluster Quality Analyzer (Phase 2)

This module implements comprehensive cluster quality assessment and optimization
techniques that provide detailed analysis of clustering results and suggest
improvements for achieving optimal performance.
"""

import numpy as np
from sklearn.neighbors import KDTree
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Union
import logging
from .density_engine import DensityAnalysis, DensityRegion
from .multi_density_clustering import ClusterRegion, HierarchicalCluster, ClusterHierarchy
from .boundary_processor import BoundaryAnalysis, BoundaryPoint, BoundaryRegion

logger = logging.getLogger(__name__)

@dataclass
class ClusterQualityMetrics:
    """Comprehensive quality metrics for a single cluster."""
    cluster_id: int
    
    # Basic metrics
    size: int
    density: float
    compactness: float
    separation: float
    
    # Advanced metrics
    silhouette_score: float
    cohesion_score: float
    stability_score: float
    boundary_quality: float
    
    # Geometric metrics
    aspect_ratio: float
    convex_hull_ratio: float
    inertia: float
    
    # Context metrics
    relative_density: float
    isolation_score: float
    merge_potential: float
    
    # Overall quality
    overall_score: float
    quality_grade: str  # 'A', 'B', 'C', 'D', 'F'

@dataclass
class GlobalQualityMetrics:
    """Quality metrics for the entire clustering result."""
    
    # Cluster-level aggregations
    num_clusters: int
    num_noise_points: int
    total_points: int
    
    # Global clustering metrics
    overall_silhouette: float
    calinski_harabasz_index: float
    davies_bouldin_index: float
    
    # Distribution metrics
    cluster_size_variance: float
    density_distribution_score: float
    coverage_ratio: float
    
    # Quality distribution
    high_quality_clusters: int
    medium_quality_clusters: int
    low_quality_clusters: int
    
    # Improvement potential
    merge_potential_score: float
    split_potential_score: float
    noise_reduction_potential: float
    
    # Overall assessment
    clustering_quality_score: float
    quality_grade: str
    recommendations: List[str]

@dataclass
class QualityAnalysisResult:
    """Complete quality analysis results."""
    cluster_metrics: Dict[int, ClusterQualityMetrics]
    global_metrics: GlobalQualityMetrics
    quality_improvements: Dict[str, float]
    optimization_suggestions: List[Dict[str, any]]
    detailed_report: str


class ClusterQualityAnalyzer:
    """
    Advanced cluster quality analysis system that provides comprehensive
    assessment of clustering results and optimization recommendations.
    
    This component is essential for achieving the target performance improvements
    by identifying suboptimal clusters and suggesting specific improvements.
    """
    
    def __init__(self, 
                 quality_threshold: float = 0.6,
                 silhouette_weight: float = 0.3,
                 separation_weight: float = 0.25,
                 compactness_weight: float = 0.25,
                 stability_weight: float = 0.2):
        """
        Initialize the Cluster Quality Analyzer.
        
        Parameters:
        - quality_threshold: Minimum acceptable quality score
        - silhouette_weight: Weight for silhouette score in overall quality
        - separation_weight: Weight for separation score in overall quality
        - compactness_weight: Weight for compactness score in overall quality
        - stability_weight: Weight for stability score in overall quality
        """
        self.quality_threshold = quality_threshold
        self.silhouette_weight = silhouette_weight
        self.separation_weight = separation_weight
        self.compactness_weight = compactness_weight
        self.stability_weight = stability_weight
        
        # Internal state
        self.last_analysis_ = None
        self.quality_history_ = []
        
    def comprehensive_quality_analysis(self, X: np.ndarray,
                                     clusters: Dict[int, ClusterRegion],
                                     density_analysis: DensityAnalysis,
                                     boundary_analysis: Optional[BoundaryAnalysis] = None,
                                     hierarchy: Optional[ClusterHierarchy] = None) -> QualityAnalysisResult:
        """
        Perform comprehensive quality analysis of clustering results.
        
        Parameters:
        - X: Original data points
        - clusters: Identified clusters
        - density_analysis: Density analysis results
        - boundary_analysis: Optional boundary analysis results
        - hierarchy: Optional hierarchical clustering results
        
        Returns:
        - analysis_result: Complete quality analysis results
        """
        logger.info(f"Starting comprehensive quality analysis for {len(clusters)} clusters")
        
        # Analyze individual cluster quality
        cluster_metrics = self._analyze_cluster_quality(X, clusters, density_analysis, boundary_analysis)
        logger.debug(f"Analyzed quality for {len(cluster_metrics)} clusters")
        
        # Analyze global clustering quality
        global_metrics = self._analyze_global_quality(X, clusters, cluster_metrics, density_analysis)
        logger.debug("Completed global quality analysis")
        
        # Identify quality improvements
        quality_improvements = self._identify_quality_improvements(cluster_metrics, global_metrics, boundary_analysis)
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(cluster_metrics, global_metrics, hierarchy)
        
        # Create detailed report
        detailed_report = self._generate_detailed_report(cluster_metrics, global_metrics, optimization_suggestions)
        
        analysis_result = QualityAnalysisResult(
            cluster_metrics=cluster_metrics,
            global_metrics=global_metrics,
            quality_improvements=quality_improvements,
            optimization_suggestions=optimization_suggestions,
            detailed_report=detailed_report
        )
        
        self.last_analysis_ = analysis_result
        self.quality_history_.append({
            'timestamp': 'now',  # In practice, use actual timestamp
            'overall_quality': global_metrics.clustering_quality_score,
            'num_clusters': global_metrics.num_clusters,
            'high_quality_ratio': global_metrics.high_quality_clusters / max(1, global_metrics.num_clusters)
        })
        
        logger.info(f"Quality analysis complete. Overall score: {global_metrics.clustering_quality_score:.3f}, "
                   f"Grade: {global_metrics.quality_grade}")
        
        return analysis_result
    
    def _analyze_cluster_quality(self, X: np.ndarray,
                               clusters: Dict[int, ClusterRegion],
                               density_analysis: DensityAnalysis,
                               boundary_analysis: Optional[BoundaryAnalysis]) -> Dict[int, ClusterQualityMetrics]:
        """
        Analyze quality metrics for individual clusters.
        
        Parameters:
        - X: Original data points
        - clusters: Identified clusters
        - density_analysis: Density analysis results
        - boundary_analysis: Optional boundary analysis results
        
        Returns:
        - cluster_metrics: Quality metrics for each cluster
        """
        cluster_metrics = {}
        
        # Create cluster assignment array for global metrics
        cluster_labels = self._create_cluster_labels(X, clusters)
        
        for cluster_id, cluster in clusters.items():
            if len(cluster.points) < 2:
                # Skip singleton clusters
                continue
            
            # Basic metrics
            size = len(cluster.points)
            density = self._compute_cluster_density(cluster)
            compactness = self._compute_cluster_compactness(cluster)
            separation = self._compute_cluster_separation(cluster, clusters)
            
            # Advanced metrics
            silhouette = self._compute_cluster_silhouette(cluster, X, cluster_labels)
            cohesion = self._compute_cluster_cohesion(cluster)
            stability = cluster.stability_score
            boundary_quality = self._compute_boundary_quality(cluster, boundary_analysis)
            
            # Geometric metrics
            aspect_ratio = self._compute_aspect_ratio(cluster)
            convex_hull_ratio = self._compute_convex_hull_ratio(cluster)
            inertia = self._compute_cluster_inertia(cluster)
            
            # Context metrics
            relative_density = self._compute_relative_density(cluster, density_analysis)
            isolation = self._compute_isolation_score(cluster, clusters)
            merge_potential = self._compute_merge_potential(cluster, clusters)
            
            # Calculate overall quality score
            overall_score = self._compute_overall_quality(
                silhouette, separation, compactness, stability
            )
            
            # Assign quality grade
            quality_grade = self._assign_quality_grade(overall_score)
            
            cluster_metrics[cluster_id] = ClusterQualityMetrics(
                cluster_id=cluster_id,
                size=size,
                density=density,
                compactness=compactness,
                separation=separation,
                silhouette_score=silhouette,
                cohesion_score=cohesion,
                stability_score=stability,
                boundary_quality=boundary_quality,
                aspect_ratio=aspect_ratio,
                convex_hull_ratio=convex_hull_ratio,
                inertia=inertia,
                relative_density=relative_density,
                isolation_score=isolation,
                merge_potential=merge_potential,
                overall_score=overall_score,
                quality_grade=quality_grade
            )
        
        return cluster_metrics
    
    def _create_cluster_labels(self, X: np.ndarray, clusters: Dict[int, ClusterRegion]) -> np.ndarray:
        """
        Create cluster labels array for global metrics computation.
        
        Parameters:
        - X: Original data points
        - clusters: Identified clusters
        
        Returns:
        - labels: Cluster labels for each point (-1 for noise)
        """
        labels = np.full(len(X), -1, dtype=int)
        
        # Build combined tree for efficient lookup
        all_cluster_points = []
        cluster_point_to_id = {}
        
        for cluster_id, cluster in clusters.items():
            for point in cluster.points:
                point_tuple = tuple(point)
                all_cluster_points.append(point)
                cluster_point_to_id[point_tuple] = cluster_id
        
        if not all_cluster_points:
            return labels
        
        all_cluster_points = np.array(all_cluster_points)
        tree = KDTree(all_cluster_points)
        
        # Assign labels
        for i, point in enumerate(X):
            distance, idx = tree.query([point], k=1)
            if distance[0] < 1e-10:  # Exact match
                cluster_point = all_cluster_points[idx[0]]
                point_tuple = tuple(cluster_point.flatten())  # Convert numpy array to tuple
                if point_tuple in cluster_point_to_id:
                    labels[i] = cluster_point_to_id[point_tuple]
        
        return labels
    
    def _compute_cluster_density(self, cluster: ClusterRegion) -> float:
        """Compute density metric for a cluster."""
        if len(cluster.points) < 2:
            return 0.0
        
        # Compute pairwise distances
        center = cluster.cluster_center
        distances = np.linalg.norm(cluster.points - center, axis=1)
        mean_distance = np.mean(distances)
        
        # Density is inverse of mean distance
        density = 1.0 / (1.0 + mean_distance)
        return density
    
    def _compute_cluster_compactness(self, cluster: ClusterRegion) -> float:
        """Compute compactness metric for a cluster."""
        if len(cluster.points) < 2:
            return 1.0
        
        # Within-cluster sum of squares
        center = cluster.cluster_center
        distances_squared = np.sum((cluster.points - center) ** 2, axis=1)
        wcss = np.sum(distances_squared)
        
        # Normalize by cluster size and dimensionality
        normalized_wcss = wcss / (len(cluster.points) * cluster.points.shape[1])
        
        # Compactness is inverse of normalized WCSS
        compactness = 1.0 / (1.0 + normalized_wcss)
        return compactness
    
    def _compute_cluster_separation(self, cluster: ClusterRegion, all_clusters: Dict[int, ClusterRegion]) -> float:
        """Compute separation metric for a cluster relative to others."""
        if len(all_clusters) <= 1:
            return 1.0
        
        min_distance = float('inf')
        
        for other_id, other_cluster in all_clusters.items():
            if other_id == cluster.cluster_id:
                continue
            
            # Distance between cluster centers
            center_distance = np.linalg.norm(cluster.cluster_center - other_cluster.cluster_center)
            min_distance = min(min_distance, center_distance)
        
        # Compute cluster spread for normalization
        center = cluster.cluster_center
        distances = np.linalg.norm(cluster.points - center, axis=1)
        cluster_spread = np.std(distances) if len(distances) > 1 else 1.0
        
        # Normalize separation by cluster spread
        normalized_separation = min_distance / (cluster_spread + 1e-8)
        
        # Convert to 0-1 score
        separation = min(1.0, normalized_separation / 5.0)  # Scale factor of 5
        return separation
    
    def _compute_cluster_silhouette(self, cluster: ClusterRegion, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score for a cluster."""
        if len(cluster.points) < 2 or len(np.unique(labels[labels != -1])) < 2:
            return 0.0
        
        try:
            # Find indices of points in this cluster
            cluster_indices = []
            for point in cluster.points:
                # Find the point in X
                distances = np.linalg.norm(X - point, axis=1)
                closest_idx = np.argmin(distances)
                if distances[closest_idx] < 1e-10:  # Exact match
                    cluster_indices.append(closest_idx)
            
            if not cluster_indices:
                return 0.0
            
            # Compute silhouette for this cluster's points
            cluster_indices = np.array(cluster_indices)
            cluster_labels = labels[cluster_indices]
            cluster_points = X[cluster_indices]
            
            if len(np.unique(cluster_labels)) < 2:
                # All points have same label, silhouette not meaningful
                return 0.0
            
            silhouette = silhouette_score(cluster_points, cluster_labels)
            return max(-1.0, min(1.0, silhouette))
            
        except Exception as e:
            logger.debug(f"Error computing silhouette for cluster {cluster.cluster_id}: {e}")
            return 0.0
    
    def _compute_cluster_cohesion(self, cluster: ClusterRegion) -> float:
        """Compute cohesion score for a cluster."""
        if len(cluster.points) < 2:
            return 1.0
        
        # Average pairwise distance within cluster
        points = cluster.points
        n_points = len(points)
        
        total_distance = 0.0
        count = 0
        
        # Sample pairs to avoid O(n^2) computation for large clusters
        max_pairs = min(1000, n_points * (n_points - 1) // 2)
        
        if n_points * (n_points - 1) // 2 <= max_pairs:
            # Compute all pairs
            for i in range(n_points):
                for j in range(i + 1, n_points):
                    distance = np.linalg.norm(points[i] - points[j])
                    total_distance += distance
                    count += 1
        else:
            # Sample pairs
            indices = np.random.choice(n_points, size=(max_pairs, 2), replace=True)
            for i, j in indices:
                if i != j:
                    distance = np.linalg.norm(points[i] - points[j])
                    total_distance += distance
                    count += 1
        
        if count == 0:
            return 1.0
        
        avg_distance = total_distance / count
        
        # Cohesion is inverse of average distance
        cohesion = 1.0 / (1.0 + avg_distance)
        return cohesion
    
    def _compute_boundary_quality(self, cluster: ClusterRegion, 
                                boundary_analysis: Optional[BoundaryAnalysis]) -> float:
        """Compute boundary quality score for a cluster."""
        if boundary_analysis is None:
            return 0.5  # Neutral score when no boundary analysis available
        
        # Find boundary points that belong to this cluster
        cluster_boundary_points = []
        for bp in boundary_analysis.boundary_points:
            # Check if this boundary point is near this cluster
            distance_to_center = np.linalg.norm(bp.coordinates - cluster.cluster_center)
            cluster_spread = np.std(np.linalg.norm(cluster.points - cluster.cluster_center, axis=1))
            
            if distance_to_center <= cluster_spread * 1.5:  # Within extended cluster radius
                cluster_boundary_points.append(bp)
        
        if not cluster_boundary_points:
            return 1.0  # No boundary issues
        
        # Average confidence of boundary points
        avg_confidence = np.mean([bp.confidence for bp in cluster_boundary_points])
        
        # High confidence boundary points indicate clear boundaries (good)
        # Low confidence indicates unclear boundaries (bad)
        boundary_quality = avg_confidence
        
        return boundary_quality
    
    def _compute_aspect_ratio(self, cluster: ClusterRegion) -> float:
        """Compute aspect ratio of cluster (measure of elongation)."""
        if len(cluster.points) < 3:
            return 1.0
        
        # Compute covariance matrix
        centered_points = cluster.points - cluster.cluster_center
        cov_matrix = np.cov(centered_points.T)
        
        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        
        if eigenvalues[1] > 1e-8:
            aspect_ratio = eigenvalues[0] / eigenvalues[1]
        else:
            aspect_ratio = float('inf')
        
        # Normalize to 0-1 range (1 = circular, 0 = very elongated)
        normalized_ratio = 1.0 / (1.0 + np.log(max(1.0, aspect_ratio)))
        
        return normalized_ratio
    
    def _compute_convex_hull_ratio(self, cluster: ClusterRegion) -> float:
        """Compute ratio of cluster points to convex hull area."""
        if len(cluster.points) < 3:
            return 1.0
        
        try:
            from scipy.spatial import ConvexHull
            
            # For 2D data
            if cluster.points.shape[1] == 2:
                hull = ConvexHull(cluster.points)
                hull_area = hull.volume  # In 2D, volume is area
                
                # Estimate cluster area using point density
                # This is a simplified approach
                cluster_area = len(cluster.points) * hull_area / len(hull.vertices)
                
                ratio = cluster_area / hull_area if hull_area > 0 else 1.0
                return min(1.0, ratio)
            else:
                # For higher dimensions, use a simplified approach
                return 0.5  # Neutral score
                
        except Exception:
            # If convex hull computation fails, return neutral score
            return 0.5
    
    def _compute_cluster_inertia(self, cluster: ClusterRegion) -> float:
        """Compute cluster inertia (normalized within-cluster sum of squares)."""
        if len(cluster.points) < 2:
            return 0.0
        
        # Within-cluster sum of squares
        center = cluster.cluster_center
        distances_squared = np.sum((cluster.points - center) ** 2, axis=1)
        inertia = np.sum(distances_squared)
        
        # Normalize by cluster size
        normalized_inertia = inertia / len(cluster.points)
        
        return normalized_inertia
    
    def _compute_relative_density(self, cluster: ClusterRegion, density_analysis: DensityAnalysis) -> float:
        """Compute relative density of cluster compared to its region."""
        # Find the region this cluster belongs to
        for region in density_analysis.regions:
            if region.region_id == cluster.region_id:
                return region.relative_density
        
        # Fallback: compute relative to global density
        return 0.5
    
    def _compute_isolation_score(self, cluster: ClusterRegion, all_clusters: Dict[int, ClusterRegion]) -> float:
        """Compute how isolated this cluster is from others."""
        if len(all_clusters) <= 1:
            return 1.0
        
        distances_to_others = []
        
        for other_id, other_cluster in all_clusters.items():
            if other_id == cluster.cluster_id:
                continue
            
            distance = np.linalg.norm(cluster.cluster_center - other_cluster.cluster_center)
            distances_to_others.append(distance)
        
        if not distances_to_others:
            return 1.0
        
        # Isolation is based on minimum distance to other clusters
        min_distance = min(distances_to_others)
        
        # Normalize by cluster spread
        cluster_spread = np.std(np.linalg.norm(cluster.points - cluster.cluster_center, axis=1))
        normalized_isolation = min_distance / (cluster_spread + 1e-8)
        
        # Convert to 0-1 score
        isolation = min(1.0, normalized_isolation / 3.0)  # Scale factor of 3
        
        return isolation
    
    def _compute_merge_potential(self, cluster: ClusterRegion, all_clusters: Dict[int, ClusterRegion]) -> float:
        """Compute potential for merging this cluster with others."""
        if len(all_clusters) <= 1:
            return 0.0
        
        best_merge_score = 0.0
        
        for other_id, other_cluster in all_clusters.items():
            if other_id == cluster.cluster_id:
                continue
            
            # Compute merge compatibility
            center_distance = np.linalg.norm(cluster.cluster_center - other_cluster.cluster_center)
            
            # Normalize by combined cluster spreads
            spread1 = np.std(np.linalg.norm(cluster.points - cluster.cluster_center, axis=1))
            spread2 = np.std(np.linalg.norm(other_cluster.points - other_cluster.cluster_center, axis=1))
            combined_spread = spread1 + spread2
            
            if combined_spread > 0:
                normalized_distance = center_distance / combined_spread
                merge_score = 1.0 / (1.0 + normalized_distance)
            else:
                merge_score = 1.0
            
            best_merge_score = max(best_merge_score, merge_score)
        
        return best_merge_score
    
    def _compute_overall_quality(self, silhouette: float, separation: float, 
                               compactness: float, stability: float) -> float:
        """Compute overall quality score from component metrics."""
        # Normalize silhouette from [-1, 1] to [0, 1]
        normalized_silhouette = (silhouette + 1.0) / 2.0
        
        overall = (self.silhouette_weight * normalized_silhouette +
                  self.separation_weight * separation +
                  self.compactness_weight * compactness +
                  self.stability_weight * stability)
        
        return max(0.0, min(1.0, overall))
    
    def _assign_quality_grade(self, score: float) -> str:
        """Assign letter grade based on quality score."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _analyze_global_quality(self, X: np.ndarray,
                              clusters: Dict[int, ClusterRegion],
                              cluster_metrics: Dict[int, ClusterQualityMetrics],
                              density_analysis: DensityAnalysis) -> GlobalQualityMetrics:
        """Analyze global clustering quality metrics."""
        
        # Basic counts
        num_clusters = len(clusters)
        total_points = len(X)
        clustered_points = sum(len(cluster.points) for cluster in clusters.values())
        num_noise_points = total_points - clustered_points
        
        # Create labels for global metrics
        labels = self._create_cluster_labels(X, clusters)
        valid_labels = labels[labels != -1]
        valid_points = X[labels != -1]
        
        # Global clustering metrics
        if len(valid_points) > 0 and len(np.unique(valid_labels)) > 1:
            try:
                overall_silhouette = silhouette_score(valid_points, valid_labels)
                calinski_harabasz = calinski_harabasz_score(valid_points, valid_labels)
                davies_bouldin = davies_bouldin_score(valid_points, valid_labels)
            except Exception as e:
                logger.debug(f"Error computing global metrics: {e}")
                overall_silhouette = 0.0
                calinski_harabasz = 0.0
                davies_bouldin = float('inf')
        else:
            overall_silhouette = 0.0
            calinski_harabasz = 0.0
            davies_bouldin = float('inf')
        
        # Distribution metrics
        cluster_sizes = [metrics.size for metrics in cluster_metrics.values()]
        cluster_size_variance = np.var(cluster_sizes) if cluster_sizes else 0.0
        
        # Density distribution score
        densities = [metrics.relative_density for metrics in cluster_metrics.values()]
        density_distribution_score = 1.0 - np.std(densities) if densities else 0.0
        
        # Coverage ratio
        coverage_ratio = clustered_points / total_points if total_points > 0 else 0.0
        
        # Quality distribution
        high_quality = sum(1 for m in cluster_metrics.values() if m.overall_score >= 0.8)
        medium_quality = sum(1 for m in cluster_metrics.values() if 0.6 <= m.overall_score < 0.8)
        low_quality = sum(1 for m in cluster_metrics.values() if m.overall_score < 0.6)
        
        # Improvement potential
        merge_potential_score = np.mean([m.merge_potential for m in cluster_metrics.values()]) if cluster_metrics else 0.0
        split_potential_score = sum(1 for m in cluster_metrics.values() if m.aspect_ratio < 0.3) / max(1, len(cluster_metrics))
        noise_reduction_potential = num_noise_points / total_points if total_points > 0 else 0.0
        
        # Overall clustering quality
        if cluster_metrics:
            avg_cluster_quality = np.mean([m.overall_score for m in cluster_metrics.values()])
        else:
            avg_cluster_quality = 0.0
        
        # Normalize silhouette for overall score
        normalized_silhouette = (overall_silhouette + 1.0) / 2.0
        
        clustering_quality_score = (0.4 * avg_cluster_quality + 
                                  0.3 * normalized_silhouette + 
                                  0.2 * coverage_ratio + 
                                  0.1 * density_distribution_score)
        
        quality_grade = self._assign_quality_grade(clustering_quality_score)
        
        # Generate recommendations
        recommendations = self._generate_global_recommendations(
            clustering_quality_score, merge_potential_score, split_potential_score,
            noise_reduction_potential, low_quality, coverage_ratio
        )
        
        return GlobalQualityMetrics(
            num_clusters=num_clusters,
            num_noise_points=num_noise_points,
            total_points=total_points,
            overall_silhouette=overall_silhouette,
            calinski_harabasz_index=calinski_harabasz,
            davies_bouldin_index=davies_bouldin,
            cluster_size_variance=cluster_size_variance,
            density_distribution_score=density_distribution_score,
            coverage_ratio=coverage_ratio,
            high_quality_clusters=high_quality,
            medium_quality_clusters=medium_quality,
            low_quality_clusters=low_quality,
            merge_potential_score=merge_potential_score,
            split_potential_score=split_potential_score,
            noise_reduction_potential=noise_reduction_potential,
            clustering_quality_score=clustering_quality_score,
            quality_grade=quality_grade,
            recommendations=recommendations
        )
    
    def _generate_global_recommendations(self, quality_score: float, merge_potential: float,
                                       split_potential: float, noise_potential: float,
                                       low_quality_count: int, coverage_ratio: float) -> List[str]:
        """Generate global optimization recommendations."""
        recommendations = []
        
        if quality_score < 0.6:
            recommendations.append("Overall clustering quality is low - consider parameter tuning")
        
        if merge_potential > 0.4:
            recommendations.append("High merge potential detected - consider increasing merge threshold")
        
        if split_potential > 0.3:
            recommendations.append("Elongated clusters detected - consider density-based splitting")
        
        if noise_potential > 0.2:
            recommendations.append("High noise ratio - consider adjusting epsilon or min_pts parameters")
        
        if low_quality_count > 0:
            recommendations.append(f"{low_quality_count} low-quality clusters need attention")
        
        if coverage_ratio < 0.8:
            recommendations.append("Low coverage ratio - many points classified as noise")
        
        if not recommendations:
            recommendations.append("Clustering quality is good - minor fine-tuning may still help")
        
        return recommendations
    
    def _identify_quality_improvements(self, cluster_metrics: Dict[int, ClusterQualityMetrics],
                                     global_metrics: GlobalQualityMetrics,
                                     boundary_analysis: Optional[BoundaryAnalysis]) -> Dict[str, float]:
        """Identify specific quality improvement opportunities."""
        improvements = {}
        
        # Cluster-level improvements
        low_quality_clusters = [m for m in cluster_metrics.values() if m.overall_score < self.quality_threshold]
        improvements['low_quality_cluster_ratio'] = len(low_quality_clusters) / max(1, len(cluster_metrics))
        
        # Boundary improvements
        if boundary_analysis:
            improvements['boundary_clarity_potential'] = boundary_analysis.quality_improvements.get('boundary_clarity_improvement', 0.0)
            improvements['merge_reduction_potential'] = boundary_analysis.quality_improvements.get('potential_merge_improvement', 0.0)
        else:
            improvements['boundary_clarity_potential'] = 0.0
            improvements['merge_reduction_potential'] = 0.0
        
        # Global improvements
        improvements['silhouette_improvement_potential'] = max(0.0, 0.8 - global_metrics.overall_silhouette)
        improvements['coverage_improvement_potential'] = max(0.0, 0.9 - global_metrics.coverage_ratio)
        improvements['noise_reduction_potential'] = global_metrics.noise_reduction_potential
        
        return improvements
    
    def _generate_optimization_suggestions(self, cluster_metrics: Dict[int, ClusterQualityMetrics],
                                         global_metrics: GlobalQualityMetrics,
                                         hierarchy: Optional[ClusterHierarchy]) -> List[Dict[str, any]]:
        """Generate specific optimization suggestions."""
        suggestions = []
        
        # Cluster-specific suggestions
        for cluster_id, metrics in cluster_metrics.items():
            if metrics.overall_score < self.quality_threshold:
                suggestion = {
                    'type': 'cluster_improvement',
                    'cluster_id': cluster_id,
                    'priority': 'high' if metrics.overall_score < 0.4 else 'medium',
                    'issues': [],
                    'actions': []
                }
                
                if metrics.separation < 0.5:
                    suggestion['issues'].append('poor separation')
                    suggestion['actions'].append('consider merging with nearby cluster')
                
                if metrics.compactness < 0.5:
                    suggestion['issues'].append('low compactness')
                    suggestion['actions'].append('consider splitting elongated cluster')
                
                if metrics.boundary_quality < 0.5:
                    suggestion['issues'].append('unclear boundaries')
                    suggestion['actions'].append('apply boundary refinement')
                
                suggestions.append(suggestion)
        
        # Global suggestions
        if global_metrics.merge_potential_score > 0.4:
            suggestions.append({
                'type': 'global_merge',
                'priority': 'medium',
                'description': 'Multiple clusters have high merge potential',
                'action': 'increase merge threshold or apply cross-region merging'
            })
        
        if global_metrics.noise_reduction_potential > 0.2:
            suggestions.append({
                'type': 'noise_reduction',
                'priority': 'medium',
                'description': 'High noise ratio detected',
                'action': 'adjust epsilon parameter or apply noise recovery techniques'
            })
        
        # Hierarchy-based suggestions
        if hierarchy and len(hierarchy.clusters) > len(cluster_metrics):
            suggestions.append({
                'type': 'hierarchical_optimization',
                'priority': 'low',
                'description': 'Hierarchical structure available for optimization',
                'action': 'consider using optimal hierarchy cut for better clustering'
            })
        
        return suggestions
    
    def _generate_detailed_report(self, cluster_metrics: Dict[int, ClusterQualityMetrics],
                                global_metrics: GlobalQualityMetrics,
                                optimization_suggestions: List[Dict[str, any]]) -> str:
        """Generate detailed quality analysis report."""
        
        report_lines = [
            "=== CLUSTER QUALITY ANALYSIS REPORT ===",
            "",
            f"Overall Clustering Quality: {global_metrics.clustering_quality_score:.3f} (Grade: {global_metrics.quality_grade})",
            f"Number of Clusters: {global_metrics.num_clusters}",
            f"Total Points: {global_metrics.total_points}",
            f"Noise Points: {global_metrics.num_noise_points} ({global_metrics.num_noise_points/global_metrics.total_points*100:.1f}%)",
            f"Coverage Ratio: {global_metrics.coverage_ratio:.3f}",
            "",
            "=== GLOBAL METRICS ===",
            f"Silhouette Score: {global_metrics.overall_silhouette:.3f}",
            f"Calinski-Harabasz Index: {global_metrics.calinski_harabasz_index:.3f}",
            f"Davies-Bouldin Index: {global_metrics.davies_bouldin_index:.3f}",
            "",
            "=== CLUSTER QUALITY DISTRIBUTION ===",
            f"High Quality Clusters (A-B): {global_metrics.high_quality_clusters}",
            f"Medium Quality Clusters (C-D): {global_metrics.medium_quality_clusters}",
            f"Low Quality Clusters (F): {global_metrics.low_quality_clusters}",
            "",
            "=== IMPROVEMENT POTENTIAL ===",
            f"Merge Potential: {global_metrics.merge_potential_score:.3f}",
            f"Split Potential: {global_metrics.split_potential_score:.3f}",
            f"Noise Reduction Potential: {global_metrics.noise_reduction_potential:.3f}",
            "",
            "=== INDIVIDUAL CLUSTER ANALYSIS ==="
        ]
        
        # Add individual cluster details
        for cluster_id, metrics in sorted(cluster_metrics.items()):
            report_lines.extend([
                f"",
                f"Cluster {cluster_id} (Grade: {metrics.quality_grade}, Score: {metrics.overall_score:.3f}):",
                f"  Size: {metrics.size} points",
                f"  Silhouette: {metrics.silhouette_score:.3f}",
                f"  Separation: {metrics.separation:.3f}",
                f"  Compactness: {metrics.compactness:.3f}",
                f"  Stability: {metrics.stability_score:.3f}",
                f"  Boundary Quality: {metrics.boundary_quality:.3f}"
            ])
        
        # Add optimization suggestions
        if optimization_suggestions:
            report_lines.extend([
                "",
                "=== OPTIMIZATION SUGGESTIONS ==="
            ])
            
            for i, suggestion in enumerate(optimization_suggestions, 1):
                if suggestion['type'] == 'cluster_improvement':
                    report_lines.append(f"{i}. Cluster {suggestion['cluster_id']} ({suggestion['priority']} priority):")
                    report_lines.append(f"   Issues: {', '.join(suggestion['issues'])}")
                    report_lines.append(f"   Actions: {', '.join(suggestion['actions'])}")
                else:
                    report_lines.append(f"{i}. {suggestion['description']} ({suggestion['priority']} priority)")
                    report_lines.append(f"   Action: {suggestion['action']}")
        
        # Add recommendations
        if global_metrics.recommendations:
            report_lines.extend([
                "",
                "=== RECOMMENDATIONS ==="
            ])
            for i, rec in enumerate(global_metrics.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        return "\n".join(report_lines)

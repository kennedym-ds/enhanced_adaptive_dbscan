"""
Enhanced Adaptive DBSCAN - Phase 3: Advanced Ensemble DBSCAN (AEDBSCAN)
Ensemble Clustering Module

This module implements the ensemble clustering framework for Phase 3, providing:
- Consensus clustering engine
- Parameter ensemble generation
- Voting mechanisms
- Stability assessment
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import DBSCAN
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EnsembleMember:
    """Represents a single member of the clustering ensemble"""
    parameters: Dict[str, Any]
    labels: np.ndarray
    quality_scores: Dict[str, float]
    execution_time: float
    n_clusters: int
    n_noise: int
    stability_score: float = 0.0


@dataclass
class EnsembleResult:
    """Container for ensemble clustering results"""
    consensus_labels: np.ndarray
    ensemble_members: List[EnsembleMember]
    consensus_matrix: np.ndarray
    overall_quality: Dict[str, float]
    stability_metrics: Dict[str, float]
    confidence_scores: np.ndarray
    parameter_diversity: float
    execution_stats: Dict[str, Any]


class ParameterEnsemble:
    """
    Generates diverse parameter sets for ensemble clustering
    """
    
    def __init__(self, 
                 base_eps: float = 0.5, 
                 base_min_samples: int = 5,
                 diversity_factor: float = 0.3,
                 n_members: int = 10):
        """
        Initialize parameter ensemble generator
        
        Parameters:
        -----------
        base_eps : float
            Base epsilon value for parameter generation
        base_min_samples : int
            Base min_samples value for parameter generation
        diversity_factor : float
            Factor controlling parameter diversity (0.1-0.5)
        n_members : int
            Number of ensemble members to generate
        """
        self.base_eps = base_eps
        self.base_min_samples = base_min_samples
        self.diversity_factor = diversity_factor
        self.n_members = n_members
        
    def generate_parameter_sets(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate diverse parameter sets for ensemble clustering
        
        Parameters:
        -----------
        X : np.ndarray
            Input data for parameter adaptation
            
        Returns:
        --------
        List[Dict[str, Any]]
            List of parameter dictionaries
        """
        parameter_sets = []
        
        # Analyze data characteristics for adaptive parameter generation
        n_samples, n_features = X.shape
        
        # Calculate data-driven parameter ranges
        distances = self._calculate_pairwise_distances(X)
        eps_range = self._calculate_eps_range(distances)
        min_samples_range = self._calculate_min_samples_range(n_samples, n_features)
        
        # Generate diverse parameter combinations
        for i in range(self.n_members):
            # Use different strategies for parameter selection
            if i < self.n_members // 3:
                # Conservative parameters (smaller eps, larger min_samples)
                eps = np.random.uniform(eps_range[0], eps_range[1] * 0.7)
                min_samples = np.random.randint(min_samples_range[1], min_samples_range[2])
            elif i < 2 * self.n_members // 3:
                # Moderate parameters (middle range)
                eps = np.random.uniform(eps_range[1] * 0.7, eps_range[1] * 1.3)
                min_samples = np.random.randint(min_samples_range[0], min_samples_range[1])
            else:
                # Aggressive parameters (larger eps, smaller min_samples)
                eps = np.random.uniform(eps_range[1] * 1.3, eps_range[2])
                min_samples = np.random.randint(min_samples_range[0], min_samples_range[1])
            
            params = {
                'eps': max(eps, 0.001),  # Ensure positive eps
                'min_samples': max(min_samples, 2),  # Ensure min_samples >= 2
                'algorithm': 'auto',
                'leaf_size': 30,
                'p': 2,
                'metric': 'euclidean'
            }
            
            parameter_sets.append(params)
        
        # Calculate and store parameter diversity
        self.parameter_diversity = self._calculate_parameter_diversity(parameter_sets)
        
        return parameter_sets
    
    def _calculate_pairwise_distances(self, X: np.ndarray, sample_size: int = 1000) -> np.ndarray:
        """Calculate pairwise distances for parameter estimation"""
        # Handle edge cases
        if len(X) <= 1:
            return np.array([1.0])  # Default distance for single point or empty data
            
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
            
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X_sample)
        
        # Return upper triangle (excluding diagonal)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        
        # Handle case where we have no pairs
        if len(upper_triangle) == 0:
            return np.array([1.0])  # Default distance
            
        return upper_triangle
    
    def _calculate_eps_range(self, distances: np.ndarray) -> Tuple[float, float, float]:
        """Calculate appropriate eps range based on data distances"""
        # Handle edge cases
        if len(distances) == 0 or len(distances) == 1:
            return (0.1, 0.5, 1.0)  # Default range
            
        percentiles = np.percentile(distances, [10, 50, 90])
        
        min_eps = max(percentiles[0] * 0.5, 0.01)  # Ensure minimum eps
        mid_eps = max(percentiles[1], min_eps * 2)
        max_eps = max(percentiles[2] * 1.5, mid_eps * 2)
        
        return (min_eps, mid_eps, max_eps)
    
    def _calculate_min_samples_range(self, n_samples: int, n_features: int) -> Tuple[int, int, int]:
        """Calculate appropriate min_samples range"""
        # Rule of thumb: min_samples should be at least dimensionality + 1
        min_min_samples = max(2, n_features + 1)
        
        # Adaptive based on dataset size
        if n_samples < 100:
            mid_min_samples = max(min_min_samples + 1, min(5, n_samples // 10))
            max_min_samples = max(mid_min_samples + 1, min(10, n_samples // 5))
        elif n_samples < 1000:
            mid_min_samples = max(min_min_samples + 1, min(10, n_samples // 50))
            max_min_samples = max(mid_min_samples + 1, min(20, n_samples // 20))
        else:
            mid_min_samples = max(min_min_samples + 1, min(15, n_samples // 100))
            max_min_samples = max(mid_min_samples + 1, min(30, n_samples // 50))
        
        # Ensure proper ordering
        mid_min_samples = max(mid_min_samples, min_min_samples + 1)
        max_min_samples = max(max_min_samples, mid_min_samples + 1)
        
        return (min_min_samples, mid_min_samples, max_min_samples)
    
    def _calculate_parameter_diversity(self, parameter_sets: List[Dict[str, Any]]) -> float:
        """Calculate diversity score for parameter sets"""
        if not parameter_sets:
            return 0.0
        
        eps_values = [p['eps'] for p in parameter_sets]
        min_samples_values = [p['min_samples'] for p in parameter_sets]
        
        # Normalize values for diversity calculation
        eps_normalized = (np.array(eps_values) - np.min(eps_values)) / (np.max(eps_values) - np.min(eps_values) + 1e-8)
        min_samples_normalized = (np.array(min_samples_values) - np.min(min_samples_values)) / (np.max(min_samples_values) - np.min(min_samples_values) + 1e-8)
        
        # Calculate diversity as average pairwise distance
        diversity_sum = 0.0
        count = 0
        
        for i in range(len(parameter_sets)):
            for j in range(i + 1, len(parameter_sets)):
                distance = np.sqrt((eps_normalized[i] - eps_normalized[j])**2 + 
                                 (min_samples_normalized[i] - min_samples_normalized[j])**2)
                diversity_sum += distance
                count += 1
        
        return diversity_sum / count if count > 0 else 0.0


class VotingMechanism:
    """
    Implements various voting strategies for ensemble consensus
    """
    
    def __init__(self, voting_strategy: str = 'weighted'):
        """
        Initialize voting mechanism
        
        Parameters:
        -----------
        voting_strategy : str
            Voting strategy: 'majority', 'weighted', 'quality_weighted'
        """
        self.voting_strategy = voting_strategy
        
    def vote_consensus(self, 
                      ensemble_members: List[EnsembleMember], 
                      X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate consensus labels using voting
        
        Parameters:
        -----------
        ensemble_members : List[EnsembleMember]
            List of ensemble members with their clustering results
        X : np.ndarray
            Original data for consensus building
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Consensus labels and confidence scores
        """
        n_samples = len(X)
        n_members = len(ensemble_members)
        
        if n_members == 0:
            return np.array([-1] * n_samples), np.zeros(n_samples)
        
        # Build consensus matrix
        consensus_matrix = self._build_consensus_matrix(ensemble_members, n_samples)
        
        # Apply voting strategy
        if self.voting_strategy == 'majority':
            consensus_labels, confidence_scores = self._majority_voting(consensus_matrix, ensemble_members)
        elif self.voting_strategy == 'weighted':
            consensus_labels, confidence_scores = self._weighted_voting(consensus_matrix, ensemble_members)
        elif self.voting_strategy == 'quality_weighted':
            consensus_labels, confidence_scores = self._quality_weighted_voting(consensus_matrix, ensemble_members)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
        
        return consensus_labels, confidence_scores
    
    def _build_consensus_matrix(self, ensemble_members: List[EnsembleMember], n_samples: int) -> np.ndarray:
        """Build consensus matrix showing co-clustering relationships"""
        consensus_matrix = np.zeros((n_samples, n_samples))
        
        for member in ensemble_members:
            labels = member.labels
            
            # For each pair of points, increment if they're in the same cluster
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        consensus_matrix[i, j] += 1
                        consensus_matrix[j, i] += 1
        
        # Normalize by number of ensemble members
        consensus_matrix /= len(ensemble_members)
        
        return consensus_matrix
    
    def _majority_voting(self, 
                        consensus_matrix: np.ndarray, 
                        ensemble_members: List[EnsembleMember]) -> Tuple[np.ndarray, np.ndarray]:
        """Simple majority voting for consensus"""
        n_samples = consensus_matrix.shape[0]
        
        # Use hierarchical clustering on consensus matrix
        from sklearn.cluster import AgglomerativeClustering
        
        # Convert consensus matrix to distance matrix
        distance_matrix = 1 - consensus_matrix
        
        # Determine optimal number of clusters using ensemble results
        n_clusters_votes = [max(member.labels) + 1 if max(member.labels) >= 0 else 1 
                           for member in ensemble_members]
        optimal_clusters = int(np.median(n_clusters_votes))
        optimal_clusters = max(1, optimal_clusters)
        
        # Apply hierarchical clustering
        if optimal_clusters == 1:
            consensus_labels = np.zeros(n_samples, dtype=int)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=optimal_clusters,
                metric='precomputed',
                linkage='average'
            )
            consensus_labels = clustering.fit_predict(distance_matrix)
        
        # Calculate confidence scores based on consensus strength
        confidence_scores = np.zeros(n_samples)
        for i in range(n_samples):
            # Confidence based on how often this point clusters with its assigned cluster
            same_cluster_points = np.where(consensus_labels == consensus_labels[i])[0]
            if len(same_cluster_points) > 1:
                confidence_scores[i] = np.mean(consensus_matrix[i, same_cluster_points])
            else:
                confidence_scores[i] = 0.5  # Medium confidence for singleton clusters
        
        return consensus_labels, confidence_scores
    
    def _weighted_voting(self, 
                        consensus_matrix: np.ndarray, 
                        ensemble_members: List[EnsembleMember]) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted voting based on member execution time"""
        # Weight by inverse execution time (faster = better)
        weights = np.array([1.0 / (member.execution_time + 1e-6) for member in ensemble_members])
        weights = weights / np.sum(weights)  # Normalize
        
        # Apply weights to consensus matrix
        weighted_consensus = np.zeros_like(consensus_matrix)
        
        for idx, member in enumerate(ensemble_members):
            member_consensus = np.zeros_like(consensus_matrix)
            labels = member.labels
            n_samples = len(labels)
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        member_consensus[i, j] = 1
                        member_consensus[j, i] = 1
            
            weighted_consensus += weights[idx] * member_consensus
        
        # Use majority voting logic with weighted consensus
        return self._majority_voting(weighted_consensus, ensemble_members)
    
    def _quality_weighted_voting(self, 
                               consensus_matrix: np.ndarray, 
                               ensemble_members: List[EnsembleMember]) -> Tuple[np.ndarray, np.ndarray]:
        """Quality-weighted voting based on clustering quality scores"""
        # Weight by average quality score
        weights = []
        for member in ensemble_members:
            if member.quality_scores:
                avg_quality = np.mean(list(member.quality_scores.values()))
                weights.append(max(avg_quality, 0.1))  # Ensure positive weights
            else:
                weights.append(0.5)  # Default weight
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize
        
        # Apply weights to consensus matrix
        weighted_consensus = np.zeros_like(consensus_matrix)
        
        for idx, member in enumerate(ensemble_members):
            member_consensus = np.zeros_like(consensus_matrix)
            labels = member.labels
            n_samples = len(labels)
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] != -1 and labels[j] != -1 and labels[i] == labels[j]:
                        member_consensus[i, j] = 1
                        member_consensus[j, i] = 1
            
            weighted_consensus += weights[idx] * member_consensus
        
        # Use majority voting logic with weighted consensus
        return self._majority_voting(weighted_consensus, ensemble_members)


class ConsensusClusteringEngine:
    """
    Main engine for consensus-based ensemble clustering
    """
    
    def __init__(self, 
                 n_ensemble_members: int = 10,
                 voting_strategy: str = 'weighted',
                 diversity_threshold: float = 0.3,
                 parallel_execution: bool = True,
                 random_state: Optional[int] = None):
        """
        Initialize consensus clustering engine
        
        Parameters:
        -----------
        n_ensemble_members : int
            Number of ensemble members
        voting_strategy : str
            Voting strategy for consensus
        diversity_threshold : float
            Minimum required parameter diversity
        parallel_execution : bool
            Whether to execute ensemble members in parallel
        random_state : Optional[int]
            Random state for reproducibility
        """
        self.n_ensemble_members = n_ensemble_members
        self.voting_strategy = voting_strategy
        self.diversity_threshold = diversity_threshold
        self.parallel_execution = parallel_execution
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Initialize components
        self.parameter_ensemble = ParameterEnsemble(n_members=n_ensemble_members)
        self.voting_mechanism = VotingMechanism(voting_strategy=voting_strategy)
        
    def fit_consensus_clustering(self, X: np.ndarray, base_params: Optional[Dict[str, Any]] = None) -> EnsembleResult:
        """
        Fit ensemble clustering and generate consensus
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        base_params : Optional[Dict[str, Any]]
            Base parameters for ensemble generation
            
        Returns:
        --------
        EnsembleResult
            Complete ensemble clustering results
        """
        logger.info(f"Starting consensus clustering with {self.n_ensemble_members} ensemble members")
        
        # Generate parameter sets
        if base_params:
            self.parameter_ensemble.base_eps = base_params.get('eps', 0.5)
            self.parameter_ensemble.base_min_samples = base_params.get('min_samples', 5)
        
        parameter_sets = self.parameter_ensemble.generate_parameter_sets(X)
        
        # Validate parameter diversity
        if self.parameter_ensemble.parameter_diversity < self.diversity_threshold:
            logger.warning(f"Parameter diversity ({self.parameter_ensemble.parameter_diversity:.3f}) "
                          f"below threshold ({self.diversity_threshold})")
        
        # Fit ensemble members
        ensemble_members = self._fit_ensemble_members(X, parameter_sets)
        
        # Generate consensus
        consensus_labels, confidence_scores = self.voting_mechanism.vote_consensus(ensemble_members, X)
        
        # Build consensus matrix
        consensus_matrix = self.voting_mechanism._build_consensus_matrix(ensemble_members, len(X))
        
        # Calculate overall quality metrics
        overall_quality = self._calculate_overall_quality(X, ensemble_members, consensus_labels)
        
        # Calculate stability metrics
        stability_metrics = self._calculate_stability_metrics(ensemble_members, consensus_labels)
        
        # Gather execution statistics
        execution_stats = self._gather_execution_stats(ensemble_members)
        
        # Create result object
        result = EnsembleResult(
            consensus_labels=consensus_labels,
            ensemble_members=ensemble_members,
            consensus_matrix=consensus_matrix,
            overall_quality=overall_quality,
            stability_metrics=stability_metrics,
            confidence_scores=confidence_scores,
            parameter_diversity=self.parameter_ensemble.parameter_diversity,
            execution_stats=execution_stats
        )
        
        logger.info(f"Consensus clustering completed. "
                   f"Consensus clusters: {len(np.unique(consensus_labels[consensus_labels >= 0]))}, "
                   f"Overall quality: {overall_quality.get('weighted_avg', 0.0):.3f}")
        
        return result
    
    def _fit_ensemble_members(self, X: np.ndarray, parameter_sets: List[Dict[str, Any]]) -> List[EnsembleMember]:
        """Fit all ensemble members"""
        ensemble_members = []
        
        if self.parallel_execution:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=min(self.n_ensemble_members, 4)) as executor:
                future_to_params = {
                    executor.submit(self._fit_single_member, X, params, idx): (params, idx) 
                    for idx, params in enumerate(parameter_sets)
                }
                
                for future in as_completed(future_to_params):
                    params, idx = future_to_params[future]
                    try:
                        member = future.result()
                        ensemble_members.append(member)
                    except Exception as exc:
                        logger.warning(f"Ensemble member {idx} failed: {exc}")
        else:
            # Sequential execution
            for idx, params in enumerate(parameter_sets):
                try:
                    member = self._fit_single_member(X, params, idx)
                    ensemble_members.append(member)
                except Exception as exc:
                    logger.warning(f"Ensemble member {idx} failed: {exc}")
        
        return ensemble_members
    
    def _fit_single_member(self, X: np.ndarray, params: Dict[str, Any], member_id: int) -> EnsembleMember:
        """Fit a single ensemble member"""
        import time
        
        start_time = time.time()
        
        # Create and fit DBSCAN with given parameters
        dbscan = DBSCAN(**params)
        labels = dbscan.fit_predict(X)
        
        execution_time = time.time() - start_time
        
        # Calculate quality scores
        quality_scores = self._calculate_member_quality(X, labels)
        
        # Count clusters and noise
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        
        return EnsembleMember(
            parameters=params,
            labels=labels,
            quality_scores=quality_scores,
            execution_time=execution_time,
            n_clusters=n_clusters,
            n_noise=n_noise
        )
    
    def _calculate_member_quality(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate quality scores for a single ensemble member"""
        quality_scores = {}
        
        try:
            # Only calculate if we have valid clusters
            if len(np.unique(labels)) > 1 and not all(labels == -1):
                # Silhouette score (handle noise points)
                if len(np.unique(labels[labels != -1])) > 1:
                    silhouette_labels = labels[labels != -1]
                    silhouette_X = X[labels != -1]
                    if len(silhouette_X) > 0:
                        quality_scores['silhouette'] = silhouette_score(silhouette_X, silhouette_labels)
                
                # Davies-Bouldin score
                try:
                    from sklearn.metrics import davies_bouldin_score
                    if len(np.unique(labels[labels != -1])) > 1:
                        db_labels = labels[labels != -1]
                        db_X = X[labels != -1]
                        if len(db_X) > 0:
                            quality_scores['davies_bouldin'] = davies_bouldin_score(db_X, db_labels)
                except ImportError:
                    pass
                
                # Calinski-Harabasz score
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    if len(np.unique(labels[labels != -1])) > 1:
                        ch_labels = labels[labels != -1]
                        ch_X = X[labels != -1]
                        if len(ch_X) > 0:
                            quality_scores['calinski_harabasz'] = calinski_harabasz_score(ch_X, ch_labels)
                except ImportError:
                    pass
                
                # Noise ratio (lower is better)
                quality_scores['noise_ratio'] = np.sum(labels == -1) / len(labels)
                
                # Cluster count score (moderate number of clusters preferred)
                n_clusters = len(np.unique(labels[labels != -1]))
                ideal_clusters = min(int(np.sqrt(len(X))), 10)
                cluster_score = 1.0 - abs(n_clusters - ideal_clusters) / ideal_clusters
                quality_scores['cluster_count'] = max(cluster_score, 0.0)
                
        except Exception as e:
            logger.warning(f"Error calculating quality scores: {e}")
            quality_scores['error'] = 1.0
        
        return quality_scores
    
    def _calculate_overall_quality(self, 
                                 X: np.ndarray, 
                                 ensemble_members: List[EnsembleMember], 
                                 consensus_labels: np.ndarray) -> Dict[str, float]:
        """Calculate overall ensemble quality metrics"""
        overall_quality = {}
        
        try:
            # Consensus quality (same metrics as individual members)
            consensus_quality = self._calculate_member_quality(X, consensus_labels)
            overall_quality.update({f"consensus_{k}": v for k, v in consensus_quality.items()})
            
            # Average member quality
            if ensemble_members:
                for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'noise_ratio', 'cluster_count']:
                    scores = [member.quality_scores.get(metric, 0.0) for member in ensemble_members 
                             if metric in member.quality_scores]
                    if scores:
                        overall_quality[f"avg_{metric}"] = np.mean(scores)
                        overall_quality[f"std_{metric}"] = np.std(scores)
            
            # Calculate weighted average quality (higher is better)
            positive_metrics = ['silhouette', 'calinski_harabasz', 'cluster_count']
            negative_metrics = ['davies_bouldin', 'noise_ratio']
            
            weighted_score = 0.0
            weight_sum = 0.0
            
            for metric in positive_metrics:
                if f"consensus_{metric}" in overall_quality:
                    weighted_score += overall_quality[f"consensus_{metric}"]
                    weight_sum += 1.0
            
            for metric in negative_metrics:
                if f"consensus_{metric}" in overall_quality:
                    weighted_score += (1.0 - overall_quality[f"consensus_{metric}"])
                    weight_sum += 1.0
            
            if weight_sum > 0:
                overall_quality['weighted_avg'] = weighted_score / weight_sum
            
        except Exception as e:
            logger.warning(f"Error calculating overall quality: {e}")
            overall_quality['error'] = 1.0
        
        return overall_quality
    
    def _calculate_stability_metrics(self, 
                                   ensemble_members: List[EnsembleMember], 
                                   consensus_labels: np.ndarray) -> Dict[str, float]:
        """Calculate ensemble stability metrics"""
        stability_metrics = {}
        
        try:
            if len(ensemble_members) < 2:
                return {'stability_error': 1.0}
            
            # Calculate pairwise ARI between ensemble members
            ari_scores = []
            nmi_scores = []
            
            for i in range(len(ensemble_members)):
                for j in range(i + 1, len(ensemble_members)):
                    labels1 = ensemble_members[i].labels
                    labels2 = ensemble_members[j].labels
                    
                    ari = adjusted_rand_score(labels1, labels2)
                    nmi = normalized_mutual_info_score(labels1, labels2)
                    
                    ari_scores.append(ari)
                    nmi_scores.append(nmi)
            
            if ari_scores:
                stability_metrics['mean_ari'] = np.mean(ari_scores)
                stability_metrics['std_ari'] = np.std(ari_scores)
                stability_metrics['min_ari'] = np.min(ari_scores)
            
            if nmi_scores:
                stability_metrics['mean_nmi'] = np.mean(nmi_scores)
                stability_metrics['std_nmi'] = np.std(nmi_scores)
                stability_metrics['min_nmi'] = np.min(nmi_scores)
            
            # Calculate consensus stability (how well consensus represents individual members)
            consensus_ari_scores = []
            for member in ensemble_members:
                ari = adjusted_rand_score(member.labels, consensus_labels)
                consensus_ari_scores.append(ari)
            
            if consensus_ari_scores:
                stability_metrics['consensus_ari_mean'] = np.mean(consensus_ari_scores)
                stability_metrics['consensus_ari_std'] = np.std(consensus_ari_scores)
                stability_metrics['consensus_ari_min'] = np.min(consensus_ari_scores)
            
            # Overall stability score (higher is better)
            stability_components = [
                stability_metrics.get('mean_ari', 0.0),
                stability_metrics.get('mean_nmi', 0.0),
                stability_metrics.get('consensus_ari_mean', 0.0)
            ]
            stability_metrics['overall_stability'] = np.mean(stability_components)
            
        except Exception as e:
            logger.warning(f"Error calculating stability metrics: {e}")
            stability_metrics['stability_error'] = 1.0
        
        return stability_metrics
    
    def _gather_execution_stats(self, ensemble_members: List[EnsembleMember]) -> Dict[str, Any]:
        """Gather execution statistics"""
        if not ensemble_members:
            return {'error': 'No ensemble members'}
        
        execution_times = [member.execution_time for member in ensemble_members]
        n_clusters_list = [member.n_clusters for member in ensemble_members]
        n_noise_list = [member.n_noise for member in ensemble_members]
        
        return {
            'total_execution_time': np.sum(execution_times),
            'avg_execution_time': np.mean(execution_times),
            'std_execution_time': np.std(execution_times),
            'min_execution_time': np.min(execution_times),
            'max_execution_time': np.max(execution_times),
            'avg_n_clusters': np.mean(n_clusters_list),
            'std_n_clusters': np.std(n_clusters_list),
            'avg_n_noise': np.mean(n_noise_list),
            'std_n_noise': np.std(n_noise_list),
            'successful_members': len(ensemble_members),
            'parameter_diversity': self.parameter_ensemble.parameter_diversity
        }

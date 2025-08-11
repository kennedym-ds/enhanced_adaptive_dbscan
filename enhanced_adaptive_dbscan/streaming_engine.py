# enhanced_adaptive_dbscan/streaming_engine.py

import numpy as np
import pandas as pd
from collections import deque, defaultdict
from threading import Thread, Lock
import time
import queue
import logging
from typing import Optional, Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .dbscan import EnhancedAdaptiveDBSCAN
from .ensemble_clustering import ConsensusClusteringEngine
from .adaptive_optimization import AdaptiveTuningEngine

logger = logging.getLogger(__name__)

@dataclass
class StreamingDataPoint:
    """Represents a single data point in the stream."""
    data: np.ndarray
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None
    point_id: Optional[str] = None

@dataclass 
class StreamingClusterResult:
    """Result of streaming clustering operation."""
    labels: np.ndarray
    cluster_centers: np.ndarray
    quality_metrics: Dict[str, float]
    processing_time: float
    timestamp: float
    data_window_size: int

@dataclass
class StreamingConfig:
    """Configuration for streaming clustering."""
    window_size: int = 1000
    update_frequency: int = 100  # Update clusters every N points
    drift_detection_threshold: float = 0.1
    concept_drift_window: int = 500
    max_memory_points: int = 10000
    quality_threshold: float = 0.5
    enable_adaptive_parameters: bool = True
    enable_concept_drift_detection: bool = True

class ConceptDriftDetector:
    """Detects concept drift in streaming data."""
    
    def __init__(self, window_size: int = 500, threshold: float = 0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.recent_data = deque(maxlen=window_size)
        self.reference_data = None
        
    def add_data(self, data: np.ndarray) -> bool:
        """Add data and check for concept drift."""
        self.recent_data.extend(data)
        
        if len(self.recent_data) < self.window_size:
            return False
            
        if self.reference_data is None:
            self.reference_data = np.array(list(self.recent_data))
            return False
            
        # Compare distributions using statistical tests
        drift_detected = self._detect_drift()
        
        if drift_detected:
            # Update reference data
            self.reference_data = np.array(list(self.recent_data))
            logger.info("Concept drift detected - updating reference distribution")
            
        return bool(drift_detected)
        
    def detect_drift(self, reference: np.ndarray, new_data: np.ndarray) -> Tuple[bool, float]:
        """Detect drift between reference and new data distributions."""
        from scipy import stats
        
        try:
            if len(reference.shape) > 1 and reference.shape[1] > 1:
                # For multivariate data, use first dimension
                ref_1d = reference[:, 0]
                new_1d = new_data[:, 0]
            else:
                ref_1d = reference.flatten()
                new_1d = new_data.flatten()
                
            # Use Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_1d, new_1d)
            drift_detected = bool(p_value < self.threshold)  # Ensure Python bool
            
            return drift_detected, float(p_value)
            
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
            return False, 1.0
        
    def _detect_drift(self) -> bool:
        """Statistical test for distribution drift."""
        from scipy import stats
        
        recent_array = np.array(list(self.recent_data))
        
        # Use Kolmogorov-Smirnov test for each dimension
        p_values = []
        for dim in range(min(self.reference_data.shape[1], recent_array.shape[1])):
            try:
                _, p_value = stats.ks_2samp(
                    self.reference_data[:, dim],
                    recent_array[:, dim]
                )
                p_values.append(p_value)
            except Exception:
                p_values.append(1.0)  # Conservative fallback
                
        # If any dimension shows significant drift
        min_p_value = min(p_values) if p_values else 1.0
        return min_p_value < self.threshold

class StreamingDataManager:
    """Manages streaming data buffers and windowing."""
    
    def __init__(self, config):
        self.config = config
        # Handle both dict and StreamingConfig object
        if isinstance(config, dict):
            max_memory_points = config.get('max_memory_points', 10000)
            window_size = config.get('window_size', 1000)
        else:
            max_memory_points = config.max_memory_points
            window_size = config.window_size
            
        self.data_buffer = deque(maxlen=max_memory_points)
        self.current_window = deque(maxlen=window_size)
        self.lock = Lock()
        
    def add_point(self, point: StreamingDataPoint):
        """Add a new data point to the stream."""
        with self.lock:
            self.data_buffer.append(point)
            self.current_window.append(point)
            
    def get_current_window(self) -> List[StreamingDataPoint]:
        """Get current data window."""
        with self.lock:
            return list(self.current_window)
            
    def get_recent_data(self, n_points: int) -> List[StreamingDataPoint]:
        """Get most recent n data points."""
        with self.lock:
            return list(self.data_buffer)[-n_points:]

class StreamingClusteringEngine:
    """Main streaming clustering engine."""
    
    def __init__(self, 
                 config: StreamingConfig,
                 clustering_method: str = 'adaptive_dbscan',
                 **clustering_params):
        self.config = config
        self.clustering_method = clustering_method
        self.clustering_params = clustering_params
        
        # Handle both dict and StreamingConfig object
        if isinstance(config, dict):
            concept_drift_window = config.get('concept_drift_window', 100)
            drift_detection_threshold = config.get('drift_detection_threshold', 0.05)
            enable_concept_drift_detection = config.get('enable_concept_drift_detection', True)
        else:
            concept_drift_window = config.concept_drift_window
            drift_detection_threshold = config.drift_detection_threshold
            enable_concept_drift_detection = config.enable_concept_drift_detection
        
        # Initialize components
        self.data_manager = StreamingDataManager(config)
        self.drift_detector = ConceptDriftDetector(
            concept_drift_window, 
            drift_detection_threshold
        ) if enable_concept_drift_detection else None
        
        # Initialize clustering model
        self.clusterer = self._initialize_clusterer()
        
        # Streaming state
        self.is_streaming = False
        self.points_processed = 0
        self.last_update = 0
        self.current_labels = None
        self.current_centers = None
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        
        # Callbacks
        self.update_callbacks: List[Callable] = []
        
    def _initialize_clusterer(self):
        """Initialize the clustering algorithm."""
        if self.clustering_method == 'adaptive_dbscan':
            return EnhancedAdaptiveDBSCAN(**self.clustering_params)
        elif self.clustering_method == 'ensemble':
            return ConsensusClusteringEngine(**self.clustering_params)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
    
    def add_update_callback(self, callback: Callable[[StreamingClusterResult], None]):
        """Add callback for cluster updates."""
        self.update_callbacks.append(callback)
        
    def start_streaming(self):
        """Start the streaming clustering process."""
        self.is_streaming = True
        logger.info("Started streaming clustering engine")
        
    def stop_streaming(self):
        """Stop the streaming clustering process."""
        self.is_streaming = False
        logger.info("Stopped streaming clustering engine")
        
    def process_point(self, point: StreamingDataPoint) -> bool:
        """Process a single data point."""
        if not self.is_streaming:
            return False
            
        # Add point to data manager
        self.data_manager.add_point(point)
        self.points_processed += 1
        
        # Handle both dict and StreamingConfig object for update_frequency
        if isinstance(self.config, dict):
            update_frequency = self.config.get('update_frequency', 100)
        else:
            update_frequency = self.config.update_frequency
        
        # Check if we should update clusters
        should_update = (
            self.points_processed - self.last_update >= update_frequency
        )
        
        # Check for concept drift
        if self.drift_detector and should_update:
            window_data = self.data_manager.get_current_window()
            if window_data:
                data_array = np.array([p.data for p in window_data])
                drift_detected = self.drift_detector.add_data(data_array)
                if drift_detected:
                    should_update = True
                    logger.info(f"Concept drift detected at point {self.points_processed}")
        
        if should_update:
            self._update_clusters()
            self.last_update = self.points_processed
            
        return True
        
    def _update_clusters(self):
        """Update cluster assignments."""
        start_time = time.time()
        
        # Get current data window
        window_data = self.data_manager.get_current_window()
        if not window_data:
            return
            
        # Convert to numpy array
        X = np.array([point.data for point in window_data])
        
        try:
            # Perform clustering
            if self.clustering_method == 'adaptive_dbscan':
                self.clusterer.fit(X)
                labels = self.clusterer.labels_
                centers = getattr(self.clusterer, 'cluster_centers_', np.array([]))
            elif self.clustering_method == 'ensemble':
                labels = self.clusterer.fit_consensus_clustering(X)
                centers = self._compute_centers(X, labels)
            else:
                raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
                
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(X, labels)
            
            # Store results
            self.current_labels = labels
            self.current_centers = centers
            
            # Create result object
            processing_time = time.time() - start_time
            result = StreamingClusterResult(
                labels=labels,
                cluster_centers=centers,
                quality_metrics=quality_metrics,
                processing_time=processing_time,
                timestamp=time.time(),
                data_window_size=len(window_data)
            )
            
            # Track performance
            self.performance_history.append({
                'timestamp': result.timestamp,
                'processing_time': processing_time,
                'window_size': len(window_data),
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                'quality_score': quality_metrics.get('silhouette_score', 0.0)
            })
            
            # Notify callbacks
            for callback in self.update_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
                    
            logger.debug(f"Updated clusters: {len(set(labels))} clusters, "
                        f"processing time: {processing_time:.3f}s")
                        
        except Exception as e:
            logger.error(f"Clustering update failed: {e}")
    
    def process_batch(self, data: np.ndarray) -> Dict[str, Any]:
        """Process a batch of data points."""
        if not self.is_streaming:
            self.start_streaming()
        
        # Convert batch to StreamingDataPoint objects
        for point in data:
            streaming_point = StreamingDataPoint(
                data=point,
                timestamp=time.time(),
                point_id=f"batch_point_{self.points_processed}"
            )
            self.process_point(streaming_point)
        
        # Get latest results
        return self.get_latest_results()
    
    def get_latest_results(self) -> Optional[Dict[str, Any]]:
        """Get the latest clustering results."""
        if self.current_labels is None:
            return None
            
        # Get current data window
        window_data = self.data_manager.get_current_window()
        if not window_data:
            return None
            
        unique_labels = set(self.current_labels) - {-1}
        
        return {
            'labels': self.current_labels.tolist() if self.current_labels is not None else [],
            'n_clusters': len(unique_labels),
            'noise_ratio': np.sum(self.current_labels == -1) / len(self.current_labels) if self.current_labels is not None else 0,
            'cluster_centers': self._format_cluster_centers(),
            'timestamp': time.time(),
            'data_window_size': len(window_data),
            'drift_detected': False  # Add drift detection result here if available
        }
    
    def _format_cluster_centers(self):
        """Format cluster centers for JSON serialization."""
        if self.current_centers is None:
            return []
        
        try:
            # If it's a numpy array with tolist method
            if hasattr(self.current_centers, 'tolist'):
                return self.current_centers.tolist()
            # If it's a dict (from EnhancedAdaptiveDBSCAN)
            elif isinstance(self.current_centers, dict):
                return [centroid.tolist() if hasattr(centroid, 'tolist') else list(centroid) 
                       for centroid in self.current_centers.values()]
            # If it's already a list
            elif isinstance(self.current_centers, list):
                return self.current_centers
            else:
                return []
        except Exception:
            return []

    def add_data(self, data: np.ndarray) -> None:
        """Add batch data for processing (convenience method)."""
        self.process_batch(data)
            
    def _compute_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centers."""
        unique_labels = set(labels) - {-1}  # Exclude noise
        centers = []
        
        for label in unique_labels:
            mask = labels == label
            if np.any(mask):
                center = np.mean(X[mask], axis=0)
                centers.append(center)
                
        return np.array(centers) if centers else np.array([])
        
    def _calculate_quality_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        # Only calculate if we have valid clusters
        unique_labels = set(labels) - {-1}
        if len(unique_labels) > 1 and len(X) > len(unique_labels):
            try:
                # Filter out noise points for metrics
                mask = labels != -1
                if np.sum(mask) > 1:
                    X_filtered = X[mask]
                    labels_filtered = labels[mask]
                    
                    if len(set(labels_filtered)) > 1:
                        metrics['silhouette_score'] = silhouette_score(X_filtered, labels_filtered)
                        metrics['davies_bouldin_score'] = davies_bouldin_score(X_filtered, labels_filtered)
                        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_filtered, labels_filtered)
                        
                metrics['n_clusters'] = len(unique_labels)
                metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)
                
            except Exception as e:
                logger.warning(f"Quality metric calculation failed: {e}")
                metrics['error'] = str(e)
                
        return metrics
        
    def get_current_state(self) -> Dict[str, Any]:
        """Get current streaming state."""
        return {
            'is_streaming': self.is_streaming,
            'points_processed': self.points_processed,
            'current_clusters': len(set(self.current_labels)) - (1 if self.current_labels is not None and -1 in self.current_labels else 0) if self.current_labels is not None else 0,
            'buffer_size': len(self.data_manager.data_buffer),
            'window_size': len(self.data_manager.current_window),
            'last_update': self.last_update
        }
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {}
            
        history = list(self.performance_history)
        processing_times = [h['processing_time'] for h in history]
        quality_scores = [h['quality_score'] for h in history if h['quality_score'] > 0]
        
        return {
            'avg_processing_time': np.mean(processing_times),
            'max_processing_time': np.max(processing_times),
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0.0,
            'total_updates': len(history),
            'throughput_points_per_second': self.points_processed / (time.time() - history[0]['timestamp']) if history else 0.0
        }

class StreamingDataSimulator:
    """Simulates streaming data for testing."""
    
    def __init__(self, data_generator: Callable[[], np.ndarray], 
                 rate_hz: float = 10.0):
        self.data_generator = data_generator
        self.rate_hz = rate_hz
        self.is_running = False
        self.thread = None
        
    def start(self, engine: StreamingClusteringEngine):
        """Start data simulation."""
        self.is_running = True
        self.thread = Thread(target=self._generate_data, args=(engine,))
        self.thread.start()
        
    def stop(self):
        """Stop data simulation."""
        self.is_running = False
        if self.thread:
            self.thread.join()
            
    def _generate_data(self, engine: StreamingClusteringEngine):
        """Generate data points at specified rate."""
        point_id = 0
        while self.is_running:
            try:
                data = self.data_generator()
                point = StreamingDataPoint(
                    data=data,
                    timestamp=time.time(),
                    point_id=f"sim_{point_id}"
                )
                engine.process_point(point)
                point_id += 1
                
                time.sleep(1.0 / self.rate_hz)
                
            except Exception as e:
                logger.error(f"Data generation error: {e}")
                break

# Example usage and factory functions
def create_blobs_simulator(n_centers: int = 3, noise_level: float = 0.1) -> StreamingDataSimulator:
    """Create a blob data simulator."""
    from sklearn.datasets import make_blobs
    
    def generate_blob():
        X, _ = make_blobs(n_samples=1, centers=n_centers, n_features=2, 
                         cluster_std=1.0 + np.random.normal(0, noise_level))
        return X[0]
        
    return StreamingDataSimulator(generate_blob, rate_hz=20.0)

def create_streaming_engine(method: str = 'adaptive_dbscan', **kwargs) -> StreamingClusteringEngine:
    """Factory function to create streaming engine."""
    config = StreamingConfig(**{k: v for k, v in kwargs.items() if k in StreamingConfig.__dataclass_fields__})
    clustering_params = {k: v for k, v in kwargs.items() if k not in StreamingConfig.__dataclass_fields__}
    
    return StreamingClusteringEngine(config, method, **clustering_params)

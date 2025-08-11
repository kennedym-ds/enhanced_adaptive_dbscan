# enhanced_adaptive_dbscan/web_api.py

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
import json
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict
import threading
from queue import Queue
import uuid

from .streaming_engine import (
    StreamingClusteringEngine, 
    StreamingConfig, 
    StreamingDataPoint,
    create_streaming_engine
)
from .dbscan import EnhancedAdaptiveDBSCAN
from .ensemble_clustering import ConsensusClusteringEngine
from .adaptive_optimization import AdaptiveTuningEngine

logger = logging.getLogger(__name__)

class ClusteringWebAPI:
    """Web API for clustering services."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for web dashboard
        
        self.host = host
        self.port = port
        self.debug = debug
        
        # Active sessions
        self.active_sessions: Dict[str, StreamingClusteringEngine] = {}
        self.session_lock = threading.Lock()
        
        # Setup routes
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.route('/', methods=['GET'])
        def home():
            """API documentation and status."""
            return jsonify({
                'name': 'Enhanced Adaptive DBSCAN API',
                'version': '1.0.0',
                'status': 'running',
                'endpoints': {
                    '/api/cluster/batch': 'POST - Batch clustering',
                    '/api/cluster/streaming/start': 'POST - Start streaming session',
                    '/api/cluster/streaming/data': 'POST - Send streaming data',
                    '/api/cluster/streaming/status': 'GET - Get streaming status',
                    '/api/cluster/streaming/stop': 'POST - Stop streaming session',
                    '/api/optimize': 'POST - Parameter optimization',
                    '/api/dashboard': 'GET - Web dashboard',
                    '/api/health': 'GET - Health check'
                }
            })
            
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': time.time(),
                'active_sessions': len(self.active_sessions)
            })
            
        @self.app.route('/api/cluster/batch', methods=['POST'])
        def cluster_batch():
            """Perform batch clustering."""
            try:
                data = request.get_json()
                
                # Validate input
                if 'data' not in data:
                    return jsonify({'error': 'Missing data field'}), 400
                    
                X = np.array(data['data'])
                method = data.get('method', 'adaptive_dbscan')
                params = data.get('parameters', {})
                
                # Perform clustering
                start_time = time.time()
                
                if method == 'adaptive_dbscan':
                    # Filter out parameters not supported by EnhancedAdaptiveDBSCAN
                    supported_params = {}
                    param_mapping = {
                        'eps': None,  # EnhancedAdaptiveDBSCAN doesn't use eps
                        'min_samples': 'min_scaling',  # Map min_samples to min_scaling
                    }
                    
                    for key, value in params.items():
                        if key in param_mapping:
                            mapped_key = param_mapping[key]
                            if mapped_key is not None:
                                supported_params[mapped_key] = value
                        else:
                            # Pass through other parameters directly
                            supported_params[key] = value
                    
                    clusterer = EnhancedAdaptiveDBSCAN(**supported_params)
                    clusterer.fit(X)
                    labels = clusterer.labels_.tolist()
                    
                    # Handle cluster centers (dict of {label: centroid})
                    cluster_centers_dict = getattr(clusterer, 'cluster_centers_', {})
                    if isinstance(cluster_centers_dict, dict) and cluster_centers_dict:
                        centers = []
                        for centroid in cluster_centers_dict.values():
                            if hasattr(centroid, 'tolist'):
                                centers.append(centroid.tolist())
                            elif isinstance(centroid, dict):
                                # Handle dict format centroids from EnhancedAdaptiveDBSCAN
                                centers.append(list(centroid.values()) if centroid else [])
                            else:
                                centers.append(list(centroid) if centroid is not None else [])
                    else:
                        centers = []
                    
                elif method == 'ensemble':
                    clusterer = ConsensusClusteringEngine(**params)
                    labels = clusterer.fit_consensus_clustering(X).tolist()
                    centers = self._compute_centers(X, np.array(labels)).tolist()
                    
                else:
                    return jsonify({'error': f'Unknown method: {method}'}), 400
                    
                processing_time = time.time() - start_time
                
                # Calculate metrics
                metrics = self._calculate_batch_metrics(X, np.array(labels))
                
                return jsonify({
                    'labels': labels,
                    'cluster_centers': centers,
                    'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
                    'processing_time': processing_time,
                    'metrics': metrics
                })
                
            except Exception as e:
                logger.error(f"Batch clustering error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/cluster/streaming/start', methods=['POST'])
        def start_streaming():
            """Start a new streaming clustering session."""
            try:
                data = request.get_json() or {}
                
                # Generate session ID
                session_id = str(uuid.uuid4())
                
                # Parse configuration
                config_data = data.get('config', {})
                method = data.get('method', 'adaptive_dbscan')
                params = data.get('parameters', {})
                
                # Create streaming engine
                engine = create_streaming_engine(method=method, **config_data, **params)
                engine.start_streaming()
                
                # Store session
                with self.session_lock:
                    self.active_sessions[session_id] = engine
                    
                logger.info(f"Started streaming session {session_id}")
                
                return jsonify({
                    'session_id': session_id,
                    'status': 'started',
                    'method': method,
                    'config': config_data
                })
                
            except Exception as e:
                logger.error(f"Start streaming error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/cluster/streaming/data', methods=['POST'])
        def send_streaming_data():
            """Send data to streaming session."""
            try:
                data = request.get_json()
                
                session_id = data.get('session_id')
                if not session_id:
                    return jsonify({'error': 'Missing session_id'}), 400
                    
                # Get session
                with self.session_lock:
                    engine = self.active_sessions.get(session_id)
                    
                if not engine:
                    return jsonify({'error': 'Session not found'}), 404
                    
                # Process data points
                points_data = data.get('points', [])
                processed_count = 0
                
                for point_data in points_data:
                    if isinstance(point_data, list):
                        # Simple array format
                        point = StreamingDataPoint(
                            data=np.array(point_data),
                            timestamp=time.time(),
                            point_id=f"api_{processed_count}"
                        )
                    else:
                        # Object format with metadata
                        point = StreamingDataPoint(
                            data=np.array(point_data.get('data', [])),
                            timestamp=point_data.get('timestamp', time.time()),
                            metadata=point_data.get('metadata'),
                            point_id=point_data.get('point_id', f"api_{processed_count}")
                        )
                        
                    if engine.process_point(point):
                        processed_count += 1
                        
                return jsonify({
                    'session_id': session_id,
                    'points_processed': processed_count,
                    'total_processed': engine.points_processed
                })
                
            except Exception as e:
                logger.error(f"Streaming data error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/cluster/streaming/status/<session_id>', methods=['GET'])
        def get_streaming_status(session_id):
            """Get streaming session status."""
            try:
                with self.session_lock:
                    engine = self.active_sessions.get(session_id)
                    
                if not engine:
                    return jsonify({'error': 'Session not found'}), 404
                    
                state = engine.get_current_state()
                performance = engine.get_performance_summary()
                
                # Get current clustering results
                results = {
                    'session_id': session_id,
                    'state': state,
                    'performance': performance,
                    'current_labels': engine.current_labels.tolist() if engine.current_labels is not None else None,
                    'current_centers': engine.current_centers.tolist() if engine.current_centers is not None else None
                }
                
                return jsonify(results)
                
            except Exception as e:
                logger.error(f"Status error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/cluster/streaming/stop', methods=['POST'])
        def stop_streaming():
            """Stop streaming session."""
            try:
                data = request.get_json() or {}
                session_id = data.get('session_id')
                
                if not session_id:
                    return jsonify({'error': 'Missing session_id'}), 400
                    
                with self.session_lock:
                    engine = self.active_sessions.pop(session_id, None)
                    
                if engine:
                    engine.stop_streaming()
                    logger.info(f"Stopped streaming session {session_id}")
                    
                    final_state = engine.get_current_state()
                    final_performance = engine.get_performance_summary()
                    
                    return jsonify({
                        'session_id': session_id,
                        'status': 'stopped',
                        'final_state': final_state,
                        'final_performance': final_performance
                    })
                else:
                    return jsonify({'error': 'Session not found'}), 404
                    
            except Exception as e:
                logger.error(f"Stop streaming error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/optimize', methods=['POST'])
        def optimize_parameters():
            """Optimize clustering parameters."""
            try:
                data = request.get_json()
                
                if 'data' not in data:
                    return jsonify({'error': 'Missing data field'}), 400
                    
                X = np.array(data['data'])
                parameter_space = data.get('parameter_space', {})
                method = data.get('optimization_method', 'bayesian')
                n_iterations = data.get('n_iterations', 20)
                metric = data.get('metric', 'silhouette_score')
                
                # Create optimization engine
                tuning_engine = AdaptiveTuningEngine(
                    optimization_method=method,
                    n_iterations=n_iterations,
                    prediction_enabled=True,
                    meta_learning_enabled=True
                )
                
                # Optimize
                start_time = time.time()
                result = tuning_engine.optimize_parameters(X, parameter_space, metric)
                optimization_time = time.time() - start_time
                
                return jsonify({
                    'best_parameters': result.best_parameters,
                    'best_score': result.best_score,
                    'optimization_time': optimization_time,
                    'meta_learning_insights': result.meta_learning_insights if hasattr(result, 'meta_learning_insights') else {},
                    'optimization_history': getattr(result, 'optimization_history', [])
                })
                
            except Exception as e:
                logger.error(f"Optimization error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/api/dashboard')
        def dashboard():
            """Web dashboard for monitoring."""
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Enhanced Adaptive DBSCAN Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .container { max-width: 1200px; margin: 0 auto; }
                    .header { text-align: center; margin-bottom: 30px; }
                    .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px; padding: 10px; background: #f5f5f5; border-radius: 3px; }
                    .controls { margin: 20px 0; }
                    .control-group { margin: 10px 0; }
                    button { padding: 10px 20px; margin: 5px; cursor: pointer; }
                    .status { font-weight: bold; }
                    .error { color: red; }
                    .success { color: green; }
                    #plot { height: 500px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🧬 Enhanced Adaptive DBSCAN Dashboard</h1>
                        <p>Real-time clustering monitoring and control</p>
                    </div>
                    
                    <div class="section">
                        <h2>🎮 Streaming Controls</h2>
                        <div class="controls">
                            <div class="control-group">
                                <label>Method:</label>
                                <select id="method">
                                    <option value="adaptive_dbscan">Adaptive DBSCAN</option>
                                    <option value="ensemble">Ensemble Clustering</option>
                                </select>
                            </div>
                            <div class="control-group">
                                <button onclick="startStreaming()">Start Streaming</button>
                                <button onclick="stopStreaming()">Stop Streaming</button>
                                <button onclick="generateData()">Generate Test Data</button>
                            </div>
                        </div>
                        <div id="session-status" class="status">No active session</div>
                    </div>
                    
                    <div class="section">
                        <h2>📊 Performance Metrics</h2>
                        <div id="metrics">
                            <div class="metric">Points Processed: <span id="points-processed">0</span></div>
                            <div class="metric">Active Clusters: <span id="clusters">0</span></div>
                            <div class="metric">Processing Time: <span id="processing-time">0</span>ms</div>
                            <div class="metric">Quality Score: <span id="quality-score">N/A</span></div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>📈 Real-time Visualization</h2>
                        <div id="plot"></div>
                    </div>
                </div>
                
                <script>
                let currentSessionId = null;
                let updateInterval = null;
                
                function startStreaming() {
                    const method = document.getElementById('method').value;
                    
                    fetch('/api/cluster/streaming/start', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            method: method,
                            config: {
                                window_size: 500,
                                update_frequency: 50
                            }
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.session_id) {
                            currentSessionId = data.session_id;
                            document.getElementById('session-status').innerHTML = 
                                '<span class="success">Active Session: ' + currentSessionId + '</span>';
                            startMonitoring();
                        } else {
                            document.getElementById('session-status').innerHTML = 
                                '<span class="error">Error: ' + data.error + '</span>';
                        }
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        document.getElementById('session-status').innerHTML = 
                            '<span class="error">Connection error</span>';
                    });
                }
                
                function stopStreaming() {
                    if (!currentSessionId) return;
                    
                    fetch('/api/cluster/streaming/stop', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({session_id: currentSessionId})
                    })
                    .then(response => response.json())
                    .then(data => {
                        currentSessionId = null;
                        document.getElementById('session-status').innerHTML = 
                            '<span class="status">Session stopped</span>';
                        stopMonitoring();
                    });
                }
                
                function generateData() {
                    if (!currentSessionId) return;
                    
                    // Generate random test data
                    const points = [];
                    for (let i = 0; i < 10; i++) {
                        points.push([
                            Math.random() * 10 - 5,
                            Math.random() * 10 - 5
                        ]);
                    }
                    
                    fetch('/api/cluster/streaming/data', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            session_id: currentSessionId,
                            points: points
                        })
                    });
                }
                
                function startMonitoring() {
                    updateInterval = setInterval(updateStatus, 1000);
                }
                
                function stopMonitoring() {
                    if (updateInterval) {
                        clearInterval(updateInterval);
                        updateInterval = null;
                    }
                }
                
                function updateStatus() {
                    if (!currentSessionId) return;
                    
                    fetch('/api/cluster/streaming/status/' + currentSessionId)
                    .then(response => response.json())
                    .then(data => {
                        if (data.state) {
                            document.getElementById('points-processed').textContent = data.state.points_processed || 0;
                            document.getElementById('clusters').textContent = data.state.current_clusters || 0;
                        }
                        
                        if (data.performance) {
                            document.getElementById('processing-time').textContent = 
                                Math.round((data.performance.avg_processing_time || 0) * 1000);
                            document.getElementById('quality-score').textContent = 
                                (data.performance.avg_quality_score || 0).toFixed(3);
                        }
                        
                        // Update plot if we have data
                        if (data.current_labels && data.current_centers) {
                            updatePlot(data);
                        }
                    })
                    .catch(error => console.error('Status update error:', error));
                }
                
                function updatePlot(data) {
                    // Placeholder for real-time plot updates
                    // In a real implementation, you'd track data points and update the visualization
                    const plotData = [{
                        x: data.current_centers.map(center => center[0]),
                        y: data.current_centers.map(center => center[1]),
                        mode: 'markers',
                        type: 'scatter',
                        name: 'Cluster Centers',
                        marker: {size: 12, color: 'red'}
                    }];
                    
                    const layout = {
                        title: 'Current Cluster Centers',
                        xaxis: {title: 'Feature 1'},
                        yaxis: {title: 'Feature 2'}
                    };
                    
                    Plotly.newPlot('plot', plotData, layout);
                }
                
                // Initialize empty plot
                Plotly.newPlot('plot', [], {title: 'Waiting for data...'});
                </script>
            </body>
            </html>
            """
            return dashboard_html
            
    def _calculate_batch_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Calculate clustering metrics for batch processing."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        try:
            unique_labels = set(labels) - {-1}
            if len(unique_labels) > 1:
                # Filter out noise points
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
            logger.warning(f"Metrics calculation failed: {e}")
            metrics['error'] = str(e)
            
        return metrics
        
    def _compute_centers(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centers."""
        unique_labels = set(labels) - {-1}
        centers = []
        
        for label in unique_labels:
            mask = labels == label
            if np.any(mask):
                center = np.mean(X[mask], axis=0)
                centers.append(center)
                
        return np.array(centers) if centers else np.array([])
        
    def run(self):
        """Run the web API server."""
        logger.info(f"Starting Enhanced Adaptive DBSCAN API on {self.host}:{self.port}")
        self.app.run(host=self.host, port=self.port, debug=self.debug, threaded=True)

def create_api_server(**kwargs) -> ClusteringWebAPI:
    """Factory function to create API server."""
    return ClusteringWebAPI(**kwargs)

if __name__ == '__main__':
    # Example usage
    api = create_api_server(debug=True)
    api.run()

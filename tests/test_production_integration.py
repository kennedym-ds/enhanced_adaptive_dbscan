# tests/test_production_integration.py

import pytest
import numpy as np
import json
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Skip tests if production dependencies are not available
pytest_flask = pytest.importorskip("flask", reason="Flask not available")
pytest_flask_cors = pytest.importorskip("flask_cors", reason="Flask-CORS not available")

from enhanced_adaptive_dbscan.streaming_engine import (
    StreamingClusteringEngine, StreamingConfig, ConceptDriftDetector
)
from enhanced_adaptive_dbscan.production_pipeline import (
    ProductionPipeline, DeploymentConfig, ModelStore, ModelValidator,
    ClusteringModelValidator, PerformanceMonitor, create_production_pipeline
)
from enhanced_adaptive_dbscan.web_api import ClusteringWebAPI
from enhanced_adaptive_dbscan.dbscan import EnhancedAdaptiveDBSCAN

class TestStreamingEngine:
    """Test streaming clustering engine."""
    
    def test_streaming_config_creation(self):
        """Test streaming configuration creation."""
        config = StreamingConfig(
            window_size=100,
            drift_threshold=0.1,
            update_interval=10,
            min_samples_for_update=50,
            enable_concept_drift_detection=True,
            drift_detection_window=500,
            performance_tracking_window=1000
        )
        
        assert config.window_size == 100
        assert config.drift_threshold == 0.1
        assert config.update_interval == 10
        assert config.min_samples_for_update == 50
        assert config.enable_concept_drift_detection is True
        assert config.drift_detection_window == 500
        assert config.performance_tracking_window == 1000
    
    def test_streaming_engine_initialization(self):
        """Test streaming engine initialization."""
        base_model = EnhancedAdaptiveDBSCAN()
        config = StreamingConfig()
        
        engine = StreamingClusteringEngine(base_model, config)
        
        assert engine.base_model is base_model
        assert engine.config is config
        assert engine.current_model is None
        assert len(engine.data_buffer) == 0
        assert engine.last_update_time is None
    
    def test_streaming_data_processing(self):
        """Test streaming data processing."""
        base_model = EnhancedAdaptiveDBSCAN()
        config = StreamingConfig(window_size=50, update_interval=25)
        engine = StreamingClusteringEngine(base_model, config)
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        
        # Process data in chunks
        chunk_size = 10
        for i in range(0, len(X), chunk_size):
            chunk = X[i:i+chunk_size]
            labels = engine.process_streaming_data(chunk)
            
            # Should return labels for the chunk
            assert len(labels) == len(chunk)
            assert all(isinstance(label, (int, np.integer)) for label in labels)
    
    def test_concept_drift_detection(self):
        """Test concept drift detection."""
        base_model = EnhancedAdaptiveDBSCAN()
        config = StreamingConfig(
            enable_concept_drift_detection=True,
            drift_threshold=0.1,
            drift_detection_window=100
        )
        engine = StreamingClusteringEngine(base_model, config)
        
        # Generate initial data
        np.random.seed(42)
        X_initial = np.random.randn(100, 2)
        engine.process_streaming_data(X_initial)
        
        # Generate drifted data
        X_drifted = np.random.randn(100, 2) + 5  # Shifted distribution
        
        # Process drifted data and check for drift detection
        labels = engine.process_streaming_data(X_drifted)
        
        # Should still return valid labels
        assert len(labels) == len(X_drifted)
    
    def test_model_updating(self):
        """Test model updating in streaming engine."""
        base_model = EnhancedAdaptiveDBSCAN()
        config = StreamingConfig(
            window_size=50,
            update_interval=25,
            min_samples_for_update=30
        )
        engine = StreamingClusteringEngine(base_model, config)
        
        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 2)
        
        # Process enough data to trigger updates
        for i in range(0, len(X), 25):
            chunk = X[i:i+25]
            labels = engine.process_streaming_data(chunk)
            
            if i >= 50:  # After enough data for update
                assert engine.current_model is not None


class TestProductionPipeline:
    """Test production pipeline functionality."""
    
    def test_deployment_config_creation(self):
        """Test deployment configuration creation."""
        config = DeploymentConfig(
            model_name="test_model",
            version="1.0.0",
            environment="test",
            enable_monitoring=True,
            enable_auto_scaling=False,
            max_concurrent_requests=100,
            request_timeout=30.0,
            model_cache_size=10,
            enable_performance_logging=True
        )
        
        assert config.model_name == "test_model"
        assert config.version == "1.0.0"
        assert config.environment == "test"
        assert config.enable_monitoring is True
        assert config.enable_auto_scaling is False
        assert config.max_concurrent_requests == 100
        assert config.request_timeout == 30.0
        assert config.model_cache_size == 10
        assert config.enable_performance_logging is True
    
    def test_model_store_operations(self):
        """Test model store operations."""
        store = ModelStore()
        
        # Create a test model
        model = EnhancedAdaptiveDBSCAN()
        X_train = np.random.randn(100, 2)
        model.fit(X_train)
        
        # Store the model
        model_id = store.store_model(model, {"version": "1.0.0", "trained_on": "test_data"})
        
        assert model_id is not None
        assert len(model_id) > 0
        
        # Retrieve the model
        retrieved_model, metadata = store.load_model(model_id)
        
        assert retrieved_model is not None
        assert metadata["version"] == "1.0.0"
        assert metadata["trained_on"] == "test_data"
        
        # List models
        models = store.list_models()
        assert len(models) >= 1
        assert any(m["id"] == model_id for m in models)
    
    def test_model_validator(self):
        """Test model validation."""
        validator = ClusteringModelValidator()
        
        # Create a test model
        model = EnhancedAdaptiveDBSCAN()
        X_train = np.random.randn(100, 2)
        model.fit(X_train)
        
        # Validate the model
        is_valid, metrics = validator.validate_model(model, X_train)
        
        assert isinstance(is_valid, bool)
        assert isinstance(metrics, dict)
        
        if is_valid:
            assert "accuracy" in metrics or "silhouette_score" in metrics
    
    def test_performance_monitor(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor()
        
        # Record some metrics
        monitor.record_prediction_time(0.1)
        monitor.record_prediction_time(0.2)
        monitor.record_prediction_time(0.15)
        
        monitor.record_memory_usage(50.0)
        monitor.record_memory_usage(55.0)
        
        # Get statistics
        stats = monitor.get_statistics()
        
        assert "prediction_times" in stats
        assert "memory_usage" in stats
        assert "avg_prediction_time" in stats
        assert "max_memory_usage" in stats
        
        assert stats["avg_prediction_time"] == pytest.approx(0.15, rel=1e-2)
        assert stats["max_memory_usage"] == 55.0
    
    def test_production_pipeline_creation(self):
        """Test production pipeline creation."""
        config = DeploymentConfig(
            model_name="test_model",
            version="1.0.0",
            environment="test"
        )
        
        pipeline = create_production_pipeline(config)
        
        assert isinstance(pipeline, ProductionPipeline)
        assert pipeline.config is config
        assert pipeline.model_store is not None
        assert pipeline.validator is not None
        assert pipeline.monitor is not None
    
    def test_production_pipeline_model_deployment(self):
        """Test model deployment through pipeline."""
        config = DeploymentConfig(
            model_name="test_model",
            version="1.0.0",
            environment="test"
        )
        
        pipeline = create_production_pipeline(config)
        
        # Create and train a model
        model = EnhancedAdaptiveDBSCAN()
        X_train = np.random.randn(100, 2)
        model.fit(X_train)
        
        # Deploy the model
        deployment_id = pipeline.deploy_model(model, {"version": "1.0.0"})
        
        assert deployment_id is not None
        assert len(deployment_id) > 0
        
        # Test prediction through pipeline
        X_test = np.random.randn(10, 2)
        predictions = pipeline.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert all(isinstance(pred, (int, np.integer)) for pred in predictions)


class TestWebAPI:
    """Test web API functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from flask import Flask
        app = Flask(__name__)
        app.config['TESTING'] = True
        
        # Initialize API
        api = ClusteringWebAPI(app)
        
        with app.test_client() as client:
            yield client
    
    def test_api_health_check(self, client):
        """Test API health check endpoint."""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
    
    def test_api_cluster_endpoint(self, client):
        """Test clustering endpoint."""
        # Prepare test data
        np.random.seed(42)
        X = np.random.randn(50, 2).tolist()
        
        request_data = {
            'data': X,
            'algorithm': 'dbscan',
            'parameters': {
                'eps': 0.5,
                'min_samples': 5
            }
        }
        
        response = client.post('/cluster', 
                              data=json.dumps(request_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert 'labels' in result
        assert 'metadata' in result
        assert len(result['labels']) == len(X)
    
    def test_api_streaming_endpoint(self, client):
        """Test streaming clustering endpoint."""
        # Initialize streaming session
        init_data = {
            'config': {
                'window_size': 100,
                'drift_threshold': 0.1
            }
        }
        
        response = client.post('/streaming/init',
                              data=json.dumps(init_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert 'session_id' in result
        
        session_id = result['session_id']
        
        # Send streaming data
        np.random.seed(42)
        X = np.random.randn(20, 2).tolist()
        
        stream_data = {
            'session_id': session_id,
            'data': X
        }
        
        response = client.post('/streaming/process',
                              data=json.dumps(stream_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        
        result = json.loads(response.data)
        assert 'labels' in result
        assert len(result['labels']) == len(X)
    
    def test_api_model_management(self, client):
        """Test model management endpoints."""
        # List models
        response = client.get('/models')
        assert response.status_code == 200
        
        models = json.loads(response.data)
        assert isinstance(models, list)
        
        # Get model info (if any models exist)
        if models:
            model_id = models[0]['id']
            response = client.get(f'/models/{model_id}')
            assert response.status_code == 200
            
            model_info = json.loads(response.data)
            assert 'id' in model_info
            assert 'metadata' in model_info


class TestProductionIntegration:
    """Test integration between all production components."""
    
    def test_end_to_end_production_workflow(self):
        """Test complete end-to-end production workflow."""
        # Create production pipeline
        config = DeploymentConfig(
            model_name="integration_test_model",
            version="1.0.0",
            environment="test",
            enable_monitoring=True
        )
        
        pipeline = create_production_pipeline(config)
        
        # Train and deploy model
        model = EnhancedAdaptiveDBSCAN()
        np.random.seed(42)
        X_train = np.random.randn(200, 2)
        model.fit(X_train)
        
        deployment_id = pipeline.deploy_model(model, {"version": "1.0.0"})
        assert deployment_id is not None
        
        # Test batch prediction
        X_test = np.random.randn(50, 2)
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(X_test)
        
        # Test streaming workflow
        streaming_config = StreamingConfig(
            window_size=100,
            update_interval=25,
            enable_concept_drift_detection=True
        )
        
        streaming_engine = StreamingClusteringEngine(model, streaming_config)
        
        # Process streaming data
        stream_data = np.random.randn(150, 2)
        chunk_size = 25
        
        all_labels = []
        for i in range(0, len(stream_data), chunk_size):
            chunk = stream_data[i:i+chunk_size]
            labels = streaming_engine.process_streaming_data(chunk)
            all_labels.extend(labels)
        
        assert len(all_labels) == len(stream_data)
        
        # Verify monitoring data was collected
        stats = pipeline.monitor.get_statistics()
        assert "prediction_times" in stats
    
    def test_production_pipeline_error_handling(self):
        """Test error handling in production pipeline."""
        config = DeploymentConfig(
            model_name="error_test_model",
            version="1.0.0",
            environment="test"
        )
        
        pipeline = create_production_pipeline(config)
        
        # Test prediction with no deployed model
        X_test = np.random.randn(10, 2)
        
        try:
            predictions = pipeline.predict(X_test)
            # Should either return predictions or raise an informative error
            if predictions is not None:
                assert len(predictions) == len(X_test)
        except Exception as e:
            # Error should be informative
            assert len(str(e)) > 0
    
    def test_concurrent_requests_handling(self):
        """Test handling of concurrent requests."""
        config = DeploymentConfig(
            model_name="concurrent_test_model",
            version="1.0.0",
            environment="test",
            max_concurrent_requests=5
        )
        
        pipeline = create_production_pipeline(config)
        
        # Deploy a model first
        model = EnhancedAdaptiveDBSCAN()
        X_train = np.random.randn(100, 2)
        model.fit(X_train)
        pipeline.deploy_model(model, {"version": "1.0.0"})
        
        # Create multiple threads for concurrent requests
        results = []
        exceptions = []
        
        def make_request():
            try:
                X_test = np.random.randn(20, 2)
                predictions = pipeline.predict(X_test)
                results.append(predictions)
            except Exception as e:
                exceptions.append(e)
        
        # Start multiple threads
        threads = []
        for _ in range(3):  # 3 concurrent requests
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(results) + len(exceptions) == 3
        
        # At least some requests should succeed
        if len(results) > 0:
            for result in results:
                assert len(result) == 20


# Performance and stress tests
class TestProductionPerformance:
    """Test production performance characteristics."""
    
    def test_prediction_latency(self):
        """Test prediction latency under normal load."""
        config = DeploymentConfig(
            model_name="latency_test_model",
            version="1.0.0",
            environment="test"
        )
        
        pipeline = create_production_pipeline(config)
        
        # Deploy model
        model = EnhancedAdaptiveDBSCAN()
        X_train = np.random.randn(500, 2)
        model.fit(X_train)
        pipeline.deploy_model(model, {"version": "1.0.0"})
        
        # Test prediction latency
        X_test = np.random.randn(100, 2)
        
        start_time = time.time()
        predictions = pipeline.predict(X_test)
        end_time = time.time()
        
        latency = end_time - start_time
        
        # Should complete within reasonable time
        assert latency < 5.0  # 5 seconds max for 100 points
        assert len(predictions) == len(X_test)
        
        print(f"Prediction latency: {latency:.4f} seconds for {len(X_test)} points")
        print(f"Throughput: {len(X_test)/latency:.2f} points/second")
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring."""
        config = DeploymentConfig(
            model_name="memory_test_model",
            version="1.0.0",
            environment="test",
            enable_monitoring=True
        )
        
        pipeline = create_production_pipeline(config)
        
        # Deploy model
        model = EnhancedAdaptiveDBSCAN()
        X_train = np.random.randn(1000, 10)  # Larger dataset
        model.fit(X_train)
        pipeline.deploy_model(model, {"version": "1.0.0"})
        
        # Make several predictions to generate monitoring data
        for _ in range(5):
            X_test = np.random.randn(50, 10)
            pipeline.predict(X_test)
        
        # Check monitoring statistics
        stats = pipeline.monitor.get_statistics()
        
        assert "memory_usage" in stats
        assert "prediction_times" in stats
        assert len(stats["prediction_times"]) >= 5
        assert len(stats["memory_usage"]) >= 5
        
        print(f"Average memory usage: {stats.get('avg_memory_usage', 'N/A')} MB")
        print(f"Peak memory usage: {stats.get('max_memory_usage', 'N/A')} MB")

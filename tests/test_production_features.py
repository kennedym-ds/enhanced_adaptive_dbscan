# Production Features Integration Tests

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime
import tempfile
import shutil
import os

# Import production components
try:
    from enhanced_adaptive_dbscan.streaming_engine import (
        StreamingConfig, 
        StreamingClusteringEngine,
        ConceptDriftDetector, 
        StreamingDataPoint,
        StreamingClusterResult
    )
    from enhanced_adaptive_dbscan.production_pipeline import (
        ProductionPipeline, 
        DeploymentConfig, 
        ModelStore, 
        ModelMetadata,
        ClusteringModelValidator,
        PerformanceMonitor
    )
    from enhanced_adaptive_dbscan.web_api import ClusteringWebAPI
    from enhanced_adaptive_dbscan.dbscan import EnhancedAdaptiveDBSCAN
    PRODUCTION_FEATURES_AVAILABLE = True
except ImportError as e:
    PRODUCTION_FEATURES_AVAILABLE = False
    print(f"Production features not available: {e}")

# Skip all tests if production features are not available
pytestmark = pytest.mark.skipif(
    not PRODUCTION_FEATURES_AVAILABLE, 
    reason="Production features dependencies not available"
)


class TestStreamingEngine:
    """Test streaming clustering engine functionality."""
    
    def test_streaming_config_validation(self):
        """Test streaming configuration validation."""
        # Valid configuration
        config = StreamingConfig(
            window_size=100,
            drift_threshold=0.1,
            update_interval=10,
            min_samples_for_update=50
        )
        
        assert config.window_size == 100
        assert config.drift_threshold == 0.1
        assert config.update_interval == 10
        assert config.min_samples_for_update == 50
    
    def test_streaming_engine_basic_functionality(self):
        """Test basic streaming engine functionality."""
        # Create a simple model
        base_model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=5)
        
        # Create streaming configuration
        config = StreamingConfig(
            window_size=50,
            drift_threshold=0.2,
            update_interval=25
        )
        
        # Initialize streaming engine
        engine = StreamingClusteringEngine(base_model, config)
        
        # Verify initialization
        assert engine.base_model is base_model
        assert engine.config is config
        assert len(engine.data_buffer) == 0
    
    def test_streaming_data_processing_workflow(self):
        """Test complete streaming data processing workflow."""
        base_model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=3)
        config = StreamingConfig(window_size=30, update_interval=15)
        engine = StreamingClusteringEngine(base_model, config)
        
        # Generate test data
        np.random.seed(42)
        data_stream = np.random.randn(100, 2)
        
        # Process data in chunks
        chunk_size = 10
        all_results = []
        
        for i in range(0, len(data_stream), chunk_size):
            chunk = data_stream[i:i+chunk_size]
            
            # Process chunk
            labels = engine.process_streaming_data(chunk)
            all_results.extend(labels)
            
            # Verify results
            assert len(labels) == len(chunk)
            assert all(isinstance(label, (int, np.integer)) for label in labels)
        
        # Verify total results
        assert len(all_results) == len(data_stream)
    
    def test_concept_drift_detection_basic(self):
        """Test basic concept drift detection."""
        base_model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=3)
        config = StreamingConfig(
            window_size=50,
            enable_concept_drift_detection=True,
            drift_threshold=0.3,
            drift_detection_window=100
        )
        engine = StreamingClusteringEngine(base_model, config)
        
        # Process initial data
        np.random.seed(42)
        initial_data = np.random.randn(60, 2)
        engine.process_streaming_data(initial_data)
        
        # Process potentially drifted data
        drifted_data = np.random.randn(40, 2) + 3  # Shifted distribution
        labels = engine.process_streaming_data(drifted_data)
        
        # Should still return valid labels
        assert len(labels) == len(drifted_data)
        assert all(isinstance(label, (int, np.integer)) for label in labels)


class TestProductionPipeline:
    """Test production pipeline functionality."""
    
    def test_deployment_config_creation(self):
        """Test deployment configuration creation."""
        config = DeploymentConfig(
            model_name="test_model",
            version="1.0.0",
            environment="test",
            enable_monitoring=True
        )
        
        assert config.model_name == "test_model"
        assert config.version == "1.0.0"
        assert config.environment == "test"
        assert config.enable_monitoring is True
    
    def test_model_store_basic_operations(self):
        """Test basic model store operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ModelStore(base_path=temp_dir)
            
            # Create and train a test model
            model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=5)
            X_train = np.random.randn(100, 2)
            model.fit(X_train)
            
            # Store the model
            metadata = ModelMetadata(
                model_name="test_model",
                version="1.0.0",
                created_at=datetime.now(),
                model_type="EnhancedAdaptiveDBSCAN",
                parameters={"eps": 0.5, "min_samples": 5},
                training_data_info={"samples": 100, "features": 2}
            )
            
            model_id = store.store_model(model, metadata)
            assert model_id is not None
            assert len(model_id) > 0
            
            # Load the model
            loaded_model, loaded_metadata = store.load_model(model_id)
            assert loaded_model is not None
            assert loaded_metadata.model_name == "test_model"
            assert loaded_metadata.version == "1.0.0"
    
    def test_model_validation(self):
        """Test model validation functionality."""
        validator = ClusteringModelValidator()
        
        # Create and train a model
        model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=5)
        X_train = np.random.randn(100, 2)
        model.fit(X_train)
        
        # Validate the model
        is_valid, metrics = validator.validate_model(model, X_train)
        
        assert isinstance(is_valid, bool)
        assert isinstance(metrics, dict)
        
        # Should have some validation metrics
        if is_valid:
            assert len(metrics) > 0
    
    def test_performance_monitoring(self):
        """Test performance monitoring."""
        monitor = PerformanceMonitor()
        
        # Record some performance metrics
        monitor.record_prediction_time(0.1)
        monitor.record_prediction_time(0.15)
        monitor.record_prediction_time(0.12)
        
        monitor.record_memory_usage(45.5)
        monitor.record_memory_usage(48.2)
        
        # Get statistics
        stats = monitor.get_statistics()
        
        assert "prediction_times" in stats
        assert "memory_usage" in stats
        assert len(stats["prediction_times"]) == 3
        assert len(stats["memory_usage"]) == 2
        
        # Check calculated statistics
        if "avg_prediction_time" in stats:
            expected_avg = (0.1 + 0.15 + 0.12) / 3
            assert abs(stats["avg_prediction_time"] - expected_avg) < 0.01


class TestProductionWorkflow:
    """Test end-to-end production workflow."""
    
    def test_complete_production_workflow(self):
        """Test complete production workflow from training to deployment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create deployment configuration
            config = DeploymentConfig(
                model_name="workflow_test_model",
                version="1.0.0",
                environment="test",
                enable_monitoring=True
            )
            
            # Create production pipeline
            pipeline = ProductionPipeline(
                config=config,
                model_store=ModelStore(base_path=temp_dir),
                validator=ClusteringModelValidator(),
                monitor=PerformanceMonitor()
            )
            
            # Train a model
            model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=5)
            np.random.seed(42)
            X_train = np.random.randn(200, 2)
            model.fit(X_train)
            
            # Deploy the model
            metadata = ModelMetadata(
                model_name="workflow_test_model",
                version="1.0.0",
                created_at=datetime.now(),
                model_type="EnhancedAdaptiveDBSCAN",
                parameters={"eps": 0.5, "min_samples": 5},
                training_data_info={"samples": 200, "features": 2}
            )
            
            deployment_id = pipeline.deploy_model(model, metadata)
            assert deployment_id is not None
            
            # Test prediction through pipeline
            X_test = np.random.randn(50, 2)
            predictions = pipeline.predict(X_test, model_id=deployment_id)
            
            assert len(predictions) == len(X_test)
            assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
            
            # Check monitoring data
            stats = pipeline.monitor.get_statistics()
            assert "prediction_times" in stats
            assert len(stats["prediction_times"]) >= 1


class TestProductionIntegration:
    """Test integration between production components."""
    
    def test_streaming_with_production_pipeline(self):
        """Test streaming engine integration with production pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create base model
            base_model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=3)
            
            # Create streaming configuration
            streaming_config = StreamingConfig(
                window_size=50,
                update_interval=25,
                enable_concept_drift_detection=True
            )
            
            # Create streaming engine
            streaming_engine = StreamingClusteringEngine(base_model, streaming_config)
            
            # Create production pipeline
            pipeline_config = DeploymentConfig(
                model_name="streaming_test_model",
                version="1.0.0",
                environment="test"
            )
            
            pipeline = ProductionPipeline(
                config=pipeline_config,
                model_store=ModelStore(base_path=temp_dir),
                validator=ClusteringModelValidator(),
                monitor=PerformanceMonitor()
            )
            
            # Process some initial data through streaming
            np.random.seed(42)
            initial_data = np.random.randn(100, 2)
            
            chunk_size = 25
            for i in range(0, len(initial_data), chunk_size):
                chunk = initial_data[i:i+chunk_size]
                labels = streaming_engine.process_streaming_data(chunk)
                assert len(labels) == len(chunk)
            
            # Deploy the current model from streaming engine
            if streaming_engine.current_model is not None:
                metadata = ModelMetadata(
                    model_name="streaming_model",
                    version="1.0.0",
                    created_at=datetime.now(),
                    model_type="EnhancedAdaptiveDBSCAN",
                    parameters={},
                    training_data_info={"samples": len(initial_data), "features": 2}
                )
                
                deployment_id = pipeline.deploy_model(streaming_engine.current_model, metadata)
                assert deployment_id is not None
    
    def test_performance_under_load(self):
        """Test performance characteristics under load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and deploy model
            config = DeploymentConfig(
                model_name="load_test_model",
                version="1.0.0",
                environment="test",
                enable_monitoring=True
            )
            
            pipeline = ProductionPipeline(
                config=config,
                model_store=ModelStore(base_path=temp_dir),
                validator=ClusteringModelValidator(),
                monitor=PerformanceMonitor()
            )
            
            # Train and deploy model
            model = EnhancedAdaptiveDBSCAN(eps=0.5, min_samples=5)
            X_train = np.random.randn(500, 2)
            model.fit(X_train)
            
            metadata = ModelMetadata(
                model_name="load_test_model",
                version="1.0.0",
                created_at=datetime.now(),
                model_type="EnhancedAdaptiveDBSCAN",
                parameters={"eps": 0.5, "min_samples": 5},
                training_data_info={"samples": 500, "features": 2}
            )
            
            deployment_id = pipeline.deploy_model(model, metadata)
            
            # Test multiple predictions
            total_start_time = time.time()
            num_requests = 10
            
            for i in range(num_requests):
                X_test = np.random.randn(20, 2)
                predictions = pipeline.predict(X_test, model_id=deployment_id)
                assert len(predictions) == 20
            
            total_time = time.time() - total_start_time
            
            # Should complete all requests in reasonable time
            assert total_time < 10.0  # 10 seconds for 10 requests
            
            # Check monitoring statistics
            stats = pipeline.monitor.get_statistics()
            assert len(stats["prediction_times"]) == num_requests
            
            print(f"Load test completed: {num_requests} requests in {total_time:.2f} seconds")
            print(f"Average request time: {total_time/num_requests:.3f} seconds")

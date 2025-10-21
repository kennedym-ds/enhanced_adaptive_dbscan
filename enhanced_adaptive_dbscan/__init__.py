import logging
from .dbscan import EnhancedAdaptiveDBSCAN
from .density_engine import (
    MultiScaleDensityEngine, 
    RelativeDensityComputer, 
    DynamicBoundaryManager,
    DensityRegion,
    DensityAnalysis
)
from .utils import (
    compute_kdist_graph,
    find_kdist_elbow,
    suggest_dbscan_parameters
)

# Phase 4: Production Pipeline & Enterprise Integration
try:
    from .streaming_engine import StreamingClusteringEngine, StreamingConfig, ConceptDriftDetector
    from .production_pipeline import (
        ProductionPipeline, DeploymentConfig, ModelStore, 
        ClusteringModelValidator, PerformanceMonitor,
        create_production_pipeline, create_deployment_config
    )
    from .web_api import ClusteringWebAPI
    
    PHASE4_AVAILABLE = True
    
    __all__ = [
        'EnhancedAdaptiveDBSCAN',
        'MultiScaleDensityEngine',
        'RelativeDensityComputer', 
        'DynamicBoundaryManager',
        'DensityRegion',
        'DensityAnalysis',
        'compute_kdist_graph',
        'find_kdist_elbow',
        'suggest_dbscan_parameters',
        # Phase 4 components
        'StreamingClusteringEngine',
        'StreamingConfig',
        'ConceptDriftDetector',
        'ProductionPipeline',
        'DeploymentConfig',
        'ModelStore',
        'ClusteringModelValidator',
        'PerformanceMonitor',
        'ClusteringWebAPI',
        'create_production_pipeline',
        'create_deployment_config'
    ]
    
except ImportError as e:
    # Phase 4 dependencies not available
    PHASE4_AVAILABLE = False
    
    __all__ = [
        'EnhancedAdaptiveDBSCAN',
        'MultiScaleDensityEngine',
        'RelativeDensityComputer', 
        'DynamicBoundaryManager',
        'DensityRegion',
        'DensityAnalysis',
        'compute_kdist_graph',
        'find_kdist_elbow',
        'suggest_dbscan_parameters'
    ]
    
    import warnings
    warnings.warn(
        f"Phase 4 Production Pipeline features not available. "
        f"Install flask, flask-cors, and pyyaml to enable. Error: {e}",
        ImportWarning
    )

# Prevent "No handler found" warnings for library users who don't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

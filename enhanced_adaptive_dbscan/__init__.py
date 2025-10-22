import logging
from .dbscan import EnhancedAdaptiveDBSCAN
from .density_engine import (
    MultiScaleDensityEngine, 
    RelativeDensityComputer, 
    DynamicBoundaryManager,
    DensityRegion,
    DensityAnalysis
)

# Phase 5: Advanced Clustering Enhancements
try:
    from .deep_clustering import (
        DeepClusteringEngine,
        DeepClusteringResult,
        HybridDeepDBSCAN,
        TORCH_AVAILABLE
    )
    DEEP_CLUSTERING_AVAILABLE = True
except ImportError:
    DEEP_CLUSTERING_AVAILABLE = False

from .scalable_indexing import (
    ScalableIndexManager,
    ScalableDBSCAN,
    IndexConfig,
    ChunkedProcessor,
    DistributedClusteringCoordinator,
    ANNOY_AVAILABLE,
    FAISS_AVAILABLE
)

from .hdbscan_clustering import (
    HDBSCANClusterer,
    MinimumSpanningTree,
    HierarchicalClusterTree,
    CondensedTree,
    StabilityBasedSelector
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
        'create_deployment_config',
        # Phase 5 components
        'ScalableIndexManager',
        'ScalableDBSCAN',
        'IndexConfig',
        'ChunkedProcessor',
        'DistributedClusteringCoordinator',
        'HDBSCANClusterer',
        'MinimumSpanningTree',
        'HierarchicalClusterTree',
        'CondensedTree',
        'StabilityBasedSelector',
    ]
    
    if DEEP_CLUSTERING_AVAILABLE:
        __all__.extend([
            'DeepClusteringEngine',
            'DeepClusteringResult',
            'HybridDeepDBSCAN',
        ])
    
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
        # Phase 5 components (always available)
        'ScalableIndexManager',
        'ScalableDBSCAN',
        'IndexConfig',
        'ChunkedProcessor',
        'DistributedClusteringCoordinator',
        'HDBSCANClusterer',
        'MinimumSpanningTree',
        'HierarchicalClusterTree',
        'CondensedTree',
        'StabilityBasedSelector',
    ]
    
    if DEEP_CLUSTERING_AVAILABLE:
        __all__.extend([
            'DeepClusteringEngine',
            'DeepClusteringResult',
            'HybridDeepDBSCAN',
        ])
    
    import warnings
    warnings.warn(
        f"Phase 4 Production Pipeline features not available. "
        f"Install flask, flask-cors, and pyyaml to enable. Error: {e}",
        ImportWarning
    )

# Prevent "No handler found" warnings for library users who don't configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

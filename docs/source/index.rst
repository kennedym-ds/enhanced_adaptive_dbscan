.. Enhanced Adaptive DBSCAN documentation master file

Enhanced Adaptive DBSCAN Framework
==================================

A comprehensive, production-ready clustering framework that extends DBSCAN with advanced adaptive algorithms, 
multi-density processing, ensemble methods, and streaming capabilities.

**Key Features:**

* **Adaptive Parameter Optimization**: Automatic parameter tuning using Bayesian and genetic algorithms
* **Multi-Density Clustering**: Handle datasets with varying density regions  
* **Ensemble Methods**: Consensus clustering for improved stability
* **Production Pipeline**: Streaming, deployment, and monitoring capabilities
* **Comprehensive API**: Scikit-learn compatible with extensive visualization

Quick Start
-----------

.. code-block:: python

   from enhanced_adaptive_dbscan import EnhancedAdaptiveDBSCAN
   import numpy as np
   
   # Generate sample data
   X = np.random.rand(100, 2)
   
   # Create and fit clustering model
   clustering = EnhancedAdaptiveDBSCAN()
   labels = clustering.fit_predict(X)
   
   # Visualize results
   clustering.plot()

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   overview
   getting_started
   usage_examples
   advanced_features

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/core
   api/algorithms
   api/optimization
   api/production

.. toctree::
   :maxdepth: 1
   :caption: Advanced Topics
   :hidden:

   algorithms/adaptive_dbscan
   algorithms/multi_density
   algorithms/ensemble_clustering
   streaming/production_pipeline

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   contributing
   changelog
   testing


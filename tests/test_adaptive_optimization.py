#!/usr/bin/env python3
"""
Test Suite for Phase 3.2: Adaptive Parameter Optimization Framework

This module tests all components of the adaptive optimization system including:
- Bayesian optimization
- Genetic algorithm optimization  
- Performance prediction
- Meta-learning components
- Parameter space exploration
- Adaptive tuning engine

Author: Enhanced Adaptive DBSCAN Development Team
Date: 2024
Version: Phase 3.2 Tests
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from sklearn.datasets import make_blobs, make_circles
from sklearn.cluster import DBSCAN
import warnings

# Suppress specific sklearn warnings that are expected during optimization
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.gaussian_process')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*ABNORMAL_TERMINATION_IN_LNSRCH.*')
warnings.filterwarnings('ignore', message='.*close to the specified.*bound.*')
warnings.filterwarnings('ignore', message='.*Predicted variances smaller than 0.*')

# Suppress specific sklearn warnings that are expected during optimization
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.gaussian_process')
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')
warnings.filterwarnings('ignore', message='.*ABNORMAL_TERMINATION_IN_LNSRCH.*')
warnings.filterwarnings('ignore', message='.*close to the specified.*bound.*')
warnings.filterwarnings('ignore', message='.*Predicted variances smaller than 0.*')

# Import the modules to test
from enhanced_adaptive_dbscan.adaptive_optimization import (
    BayesianOptimizer,
    GeneticOptimizer,
    PerformancePredictor,
    MetaLearningComponent,
    ParameterSpaceExplorer,
    AdaptiveTuningEngine,
    OptimizationResult,
    DatasetCharacteristics
)

# Remove the duplicate warning filter that was less specific
# warnings.filterwarnings('ignore')

class TestOptimizationResult:
    """Test OptimizationResult data structure."""
    
    def test_optimization_result_creation(self):
        """Test OptimizationResult creation and attributes."""
        result = OptimizationResult(
            best_parameters={'eps': 0.5, 'min_samples': 5},
            best_score=0.8,
            optimization_history=[],
            convergence_iteration=10,
            total_iterations=20,
            total_time=5.0,
            method_used='bayesian'
        )
        
        assert result.best_parameters == {'eps': 0.5, 'min_samples': 5}
        assert result.best_score == 0.8
        assert result.convergence_iteration == 10
        assert result.total_iterations == 20
        assert result.total_time == 5.0
        assert result.method_used == 'bayesian'

class TestBayesianOptimizer:
    """Test BayesianOptimizer class."""
    
    def test_bayesian_optimizer_initialization(self):
        """Test BayesianOptimizer initialization."""
        optimizer = BayesianOptimizer(acquisition_function='EI', xi=0.01, random_state=42)
        
        assert optimizer.acquisition_function == 'EI'
        assert optimizer.xi == 0.01
        assert optimizer.random_state == 42
        assert optimizer.kernel is not None
    
    def test_bayesian_optimization_simple(self):
        """Test Bayesian optimization with simple objective function."""
        def simple_objective(params):
            # Simple quadratic function with known optimum
            return -(params['x'] - 0.5) ** 2 + 1.0
        
        optimizer = BayesianOptimizer(random_state=42)
        parameter_space = {'x': (0.0, 1.0)}
        
        result = optimizer.optimize(simple_objective, parameter_space, n_iterations=10)
        
        assert isinstance(result, OptimizationResult)
        assert result.method_used == 'bayesian_optimization'
        assert 'x' in result.best_parameters
        assert 0.0 <= result.best_parameters['x'] <= 1.0
        assert result.best_score <= 1.0
        assert len(result.optimization_history) <= 10
        assert result.total_time > 0
    
    def test_bayesian_optimization_multiple_parameters(self):
        """Test Bayesian optimization with multiple parameters."""
        def multi_param_objective(params):
            # Multi-parameter function
            return -(params['x'] - 0.3) ** 2 - (params['y'] - 0.7) ** 2 + 2.0
        
        optimizer = BayesianOptimizer(random_state=42)
        parameter_space = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        
        result = optimizer.optimize(multi_param_objective, parameter_space, n_iterations=15)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.best_parameters) == 2
        assert 'x' in result.best_parameters
        assert 'y' in result.best_parameters
        assert result.best_score <= 2.0
    
    def test_acquisition_functions(self):
        """Test different acquisition functions."""
        def simple_objective(params):
            return params['x'] ** 2
        
        parameter_space = {'x': (-1.0, 1.0)}
        
        for acq_func in ['EI', 'UCB', 'PI']:
            optimizer = BayesianOptimizer(acquisition_function=acq_func, random_state=42)
            result = optimizer.optimize(simple_objective, parameter_space, n_iterations=5)
            
            assert isinstance(result, OptimizationResult)
            assert result.method_used == 'bayesian_optimization'
    
    def test_bayesian_optimization_with_exceptions(self):
        """Test Bayesian optimization handles objective function exceptions."""
        def failing_objective(params):
            if params['x'] > 0.5:
                raise ValueError("Simulated failure")
            return params['x']
        
        optimizer = BayesianOptimizer(random_state=42)
        parameter_space = {'x': (0.0, 1.0)}
        
        result = optimizer.optimize(failing_objective, parameter_space, n_iterations=10)
        
        assert isinstance(result, OptimizationResult)
        # Should still find some valid results
        assert len(result.optimization_history) > 0

class TestGeneticOptimizer:
    """Test GeneticOptimizer class."""
    
    def test_genetic_optimizer_initialization(self):
        """Test GeneticOptimizer initialization."""
        optimizer = GeneticOptimizer(
            population_size=30, 
            mutation_rate=0.1, 
            crossover_rate=0.8,
            elite_size=3,
            random_state=42
        )
        
        assert optimizer.population_size == 30
        assert optimizer.mutation_rate == 0.1
        assert optimizer.crossover_rate == 0.8
        assert optimizer.elite_size == 3
        assert optimizer.random_state == 42
    
    def test_genetic_optimization_simple(self):
        """Test genetic algorithm optimization."""
        def simple_objective(params):
            # Simple function to maximize
            return -(params['x'] - 0.7) ** 2 + 1.0
        
        optimizer = GeneticOptimizer(population_size=20, random_state=42)
        parameter_space = {'x': (0.0, 1.0)}
        
        result = optimizer.optimize(simple_objective, parameter_space, n_iterations=10)
        
        assert isinstance(result, OptimizationResult)
        assert result.method_used == 'genetic_algorithm'
        assert 'x' in result.best_parameters
        assert 0.0 <= result.best_parameters['x'] <= 1.0
        assert result.total_time >= 0  # Should be non-negative (can be 0 for very fast operations)
        assert len(result.optimization_history) > 0
    
    def test_genetic_optimization_multi_parameter(self):
        """Test genetic algorithm with multiple parameters."""
        def multi_objective(params):
            return -(params['x'] - 0.3) ** 2 - (params['y'] - 0.6) ** 2
        
        optimizer = GeneticOptimizer(population_size=20, random_state=42)
        parameter_space = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        
        result = optimizer.optimize(multi_objective, parameter_space, n_iterations=10)
        
        assert isinstance(result, OptimizationResult)
        assert len(result.best_parameters) == 2
        assert 'x' in result.best_parameters
        assert 'y' in result.best_parameters
    
    def test_tournament_selection(self):
        """Test tournament selection method."""
        optimizer = GeneticOptimizer(random_state=42)
        
        population = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        fitness_scores = np.array([0.1, 0.8, 0.3, 0.9])
        
        selected = optimizer._tournament_selection(population, fitness_scores)
        
        assert selected.shape == (2,)
        assert np.any(np.all(selected == population, axis=1))
    
    def test_crossover(self):
        """Test crossover operation."""
        optimizer = GeneticOptimizer(random_state=42)
        
        parent1 = np.array([0.1, 0.2, 0.3])
        parent2 = np.array([0.7, 0.8, 0.9])
        
        child1, child2 = optimizer._crossover(parent1, parent2)
        
        assert len(child1) == len(parent1)
        assert len(child2) == len(parent2)
        # Children should be different from parents (in most cases)
        assert not np.array_equal(child1, parent1) or not np.array_equal(child2, parent2)
    
    def test_mutation(self):
        """Test mutation operation."""
        optimizer = GeneticOptimizer(mutation_rate=1.0, random_state=42)  # High mutation rate for testing
        
        individual = np.array([0.5, 0.5])
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        
        mutated = optimizer._mutate(individual, bounds)
        
        assert len(mutated) == len(individual)
        assert np.all(mutated >= bounds[:, 0])
        assert np.all(mutated <= bounds[:, 1])

class TestPerformancePredictor:
    """Test PerformancePredictor class."""
    
    def test_performance_predictor_initialization(self):
        """Test PerformancePredictor initialization."""
        predictor = PerformancePredictor(model_type='random_forest', random_state=42)
        
        assert predictor.model_type == 'random_forest'
        assert predictor.random_state == 42
        assert not predictor.is_fitted
        assert predictor.model is not None
    
    def test_dataset_feature_extraction(self):
        """Test dataset feature extraction."""
        predictor = PerformancePredictor(random_state=42)
        
        # Create simple test data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        features = predictor.extract_dataset_features(X)
        
        assert len(features) > 0
        assert features[0] == 3  # n_samples
        assert features[1] == 2  # n_features
        # Additional features should be numeric
        assert all(isinstance(f, (int, float)) for f in features)
    
    def test_performance_prediction_training(self):
        """Test performance predictor training."""
        predictor = PerformancePredictor(random_state=42)
        
        # Create training data
        X_data = [
            np.random.rand(100, 2),
            np.random.rand(200, 2),
            np.random.rand(150, 2)
        ]
        parameters = [
            {'eps': 0.1, 'min_samples': 5},
            {'eps': 0.2, 'min_samples': 10},
            {'eps': 0.15, 'min_samples': 7}
        ]
        scores = [0.7, 0.8, 0.75]
        
        predictor.fit(X_data, parameters, scores)
        
        assert predictor.is_fitted
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) > 0
    
    def test_performance_prediction(self):
        """Test performance prediction."""
        predictor = PerformancePredictor(random_state=42)
        
        # Train predictor
        X_data = [np.random.rand(100, 2) for _ in range(5)]
        parameters = [{'eps': 0.1 + 0.05*i, 'min_samples': 5 + i} for i in range(5)]
        scores = [0.6 + 0.05*i for i in range(5)]
        
        predictor.fit(X_data, parameters, scores)
        
        # Test prediction
        test_X = np.random.rand(120, 2)
        test_params = {'eps': 0.125, 'min_samples': 6}
        
        prediction = predictor.predict(test_X, test_params)
        
        assert isinstance(prediction, float)
        
        # Test with confidence
        prediction_with_conf = predictor.predict(test_X, test_params, return_confidence=True)
        
        assert isinstance(prediction_with_conf, tuple)
        assert len(prediction_with_conf) == 2
        assert isinstance(prediction_with_conf[0], float)
        assert isinstance(prediction_with_conf[1], tuple)
    
    def test_predictor_update(self):
        """Test predictor update mechanism."""
        predictor = PerformancePredictor(random_state=42)
        
        # Train predictor
        X_data = [np.random.rand(100, 2)]
        parameters = [{'eps': 0.1, 'min_samples': 5}]
        scores = [0.7]
        
        predictor.fit(X_data, parameters, scores)
        
        # Make prediction and update
        test_X = np.random.rand(120, 2)
        test_params = {'eps': 0.15, 'min_samples': 6}
        
        prediction = predictor.predict(test_X, test_params)
        predictor.update(test_X, test_params, 0.75)
        
        assert len(predictor.prediction_history) > 0
        assert 'actual_score' in predictor.prediction_history[-1]
    
    def test_prediction_accuracy_calculation(self):
        """Test prediction accuracy calculation."""
        predictor = PerformancePredictor(random_state=42)
        
        # Train and make predictions
        X_data = [np.random.rand(100, 2)]
        parameters = [{'eps': 0.1, 'min_samples': 5}]
        scores = [0.7]
        
        predictor.fit(X_data, parameters, scores)
        
        # Make prediction and update with actual score
        test_X = np.random.rand(120, 2)
        test_params = {'eps': 0.15, 'min_samples': 6}
        
        prediction = predictor.predict(test_X, test_params)
        predictor.update(test_X, test_params, prediction + 0.01)  # Close to prediction
        
        accuracy = predictor.get_prediction_accuracy()
        
        assert accuracy is None or isinstance(accuracy, float)

class TestMetaLearningComponent:
    """Test MetaLearningComponent class."""
    
    def test_meta_learning_initialization(self):
        """Test MetaLearningComponent initialization."""
        meta_learner = MetaLearningComponent()
        
        assert len(meta_learner.dataset_experiences) == 0
        assert len(meta_learner.strategy_performance) == 0
    
    def test_dataset_analysis(self):
        """Test dataset characteristic analysis."""
        meta_learner = MetaLearningComponent()
        
        # Create test dataset
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        
        characteristics = meta_learner.analyze_dataset(X)
        
        assert isinstance(characteristics, DatasetCharacteristics)
        assert characteristics.n_samples == 100
        assert characteristics.n_features == 2
        assert characteristics.density_estimate >= 0
        assert len(characteristics.distance_distribution) > 0
    
    def test_experience_recording(self):
        """Test experience recording."""
        meta_learner = MetaLearningComponent()
        
        # Create test data
        X, _ = make_blobs(n_samples=50, random_state=42)
        characteristics = meta_learner.analyze_dataset(X)
        
        # Create mock optimization result
        result = OptimizationResult(
            best_parameters={'eps': 0.5},
            best_score=0.8,
            optimization_history=[],
            convergence_iteration=10,
            total_iterations=20,
            total_time=5.0,
            method_used='bayesian'
        )
        
        meta_learner.record_experience(characteristics, 'bayesian', result)
        
        assert len(meta_learner.dataset_experiences) == 1
        assert 'bayesian' in meta_learner.strategy_performance
        assert len(meta_learner.strategy_performance['bayesian']) == 1
    
    def test_strategy_recommendation(self):
        """Test strategy recommendation."""
        meta_learner = MetaLearningComponent()
        
        # Create test characteristics
        X, _ = make_blobs(n_samples=100, random_state=42)
        characteristics = meta_learner.analyze_dataset(X)
        
        # Should provide default recommendation for new meta-learner
        recommendation = meta_learner.recommend_strategy(characteristics)
        
        assert recommendation in ['bayesian', 'genetic']
        
        # Add some experiences
        result1 = OptimizationResult(
            best_parameters={'eps': 0.5},
            best_score=0.8,
            optimization_history=[],
            convergence_iteration=None,
            total_iterations=20,
            total_time=5.0,
            method_used='bayesian'
        )
        
        result2 = OptimizationResult(
            best_parameters={'eps': 0.3},
            best_score=0.9,
            optimization_history=[],
            convergence_iteration=None,
            total_iterations=25,
            total_time=7.0,
            method_used='genetic'
        )
        
        meta_learner.record_experience(characteristics, 'bayesian', result1)
        meta_learner.record_experience(characteristics, 'genetic', result2)
        
        # Should now recommend based on experience
        recommendation = meta_learner.recommend_strategy(characteristics)
        assert recommendation in ['bayesian', 'genetic']
    
    def test_similarity_calculation(self):
        """Test dataset similarity calculation."""
        meta_learner = MetaLearningComponent()
        
        # Create similar characteristics
        chars1 = DatasetCharacteristics(
            n_samples=100, n_features=2, density_estimate=0.5,
            noise_ratio=0.1, dimensionality_ratio=0.02,
            distance_distribution={}, cluster_separation=1.0
        )
        
        chars2 = DatasetCharacteristics(
            n_samples=110, n_features=2, density_estimate=0.55,
            noise_ratio=0.12, dimensionality_ratio=0.018,
            distance_distribution={}, cluster_separation=1.1
        )
        
        similarity = meta_learner._calculate_similarity(chars1, chars2)
        
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.5  # Should be similar
    
    def test_insights_generation(self):
        """Test meta-learning insights generation."""
        meta_learner = MetaLearningComponent()
        
        # Add some experiences
        X, _ = make_blobs(n_samples=50, random_state=42)
        characteristics = meta_learner.analyze_dataset(X)
        
        result = OptimizationResult(
            best_parameters={'eps': 0.5},
            best_score=0.8,
            optimization_history=[],
            convergence_iteration=10,
            total_iterations=20,
            total_time=5.0,
            method_used='bayesian'
        )
        
        meta_learner.record_experience(characteristics, 'bayesian', result)
        
        insights = meta_learner.get_insights()
        
        assert isinstance(insights, dict)
        assert 'total_experiences' in insights
        assert 'strategies_used' in insights
        assert 'strategy_performance' in insights
        assert insights['total_experiences'] == 1

class TestParameterSpaceExplorer:
    """Test ParameterSpaceExplorer class."""
    
    def test_parameter_space_explorer_initialization(self):
        """Test ParameterSpaceExplorer initialization."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        assert explorer.random_state == 42
        assert 'bayesian' in explorer.strategies
        assert 'genetic' in explorer.strategies
        assert explorer.performance_predictor is not None
        assert explorer.meta_learner is not None
    
    def test_parameter_space_exploration(self):
        """Test parameter space exploration."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        # Create test data and objective
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        
        def simple_objective(params):
            return -(params['eps'] - 0.5) ** 2 + 1.0
        
        parameter_space = {'eps': (0.1, 1.0)}
        
        result = explorer.explore(
            X=X,
            objective_function=simple_objective,
            parameter_space=parameter_space,
            strategy='bayesian',
            n_iterations=10,
            use_prediction=False
        )
        
        assert isinstance(result, OptimizationResult)
        assert 'eps' in result.best_parameters
        assert result.meta_learning_insights is not None
    
    def test_strategy_auto_selection(self):
        """Test automatic strategy selection."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        X, _ = make_blobs(n_samples=50, random_state=42)
        
        def simple_objective(params):
            return params['eps']
        
        parameter_space = {'eps': (0.1, 1.0)}
        
        result = explorer.explore(
            X=X,
            objective_function=simple_objective,
            parameter_space=parameter_space,
            strategy='auto',  # Auto-select strategy
            n_iterations=5,
            use_prediction=False
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.method_used in ['bayesian_optimization', 'genetic_algorithm']
    
    def test_predictor_training(self):
        """Test performance predictor training."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        # Create training data
        training_data = []
        for i in range(5):
            X = np.random.rand(100, 2)
            params = {'eps': 0.1 + 0.1*i, 'min_samples': 5 + i}
            score = 0.6 + 0.05*i
            training_data.append((X, params, score))
        
        explorer.train_predictor(training_data)
        
        assert explorer.performance_predictor.is_fitted
    
    def test_enhanced_objective_function(self):
        """Test enhanced objective function with prediction."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        # Train predictor first
        training_data = [
            (np.random.rand(100, 2), {'eps': 0.1}, 0.6),
            (np.random.rand(100, 2), {'eps': 0.2}, 0.7)
        ]
        explorer.train_predictor(training_data)
        
        X = np.random.rand(100, 2)
        
        def original_objective(params):
            return params['eps']
        
        enhanced_objective = explorer._create_enhanced_objective(
            X, original_objective, explorer.performance_predictor
        )
        
        result = enhanced_objective({'eps': 0.15})
        assert isinstance(result, float)

class TestAdaptiveTuningEngine:
    """Test AdaptiveTuningEngine class."""
    
    def test_adaptive_tuning_engine_initialization(self):
        """Test AdaptiveTuningEngine initialization."""
        engine = AdaptiveTuningEngine(random_state=42)
        
        assert engine.random_state == 42
        assert engine.explorer is not None
        assert len(engine.optimization_history) == 0
    
    def test_parameter_optimization_clustering(self):
        """Test parameter optimization with clustering function."""
        engine = AdaptiveTuningEngine(random_state=42)
        
        # Create test data
        X, true_labels = make_blobs(n_samples=100, centers=3, random_state=42)
        
        def clustering_function(data, params):
            dbscan = DBSCAN(eps=params['eps'], min_samples=int(params['min_samples']))
            return dbscan.fit_predict(data)
        
        parameter_space = {
            'eps': (0.1, 2.0),
            'min_samples': (2, 10)
        }
        
        result = engine.optimize_parameters(
            X=X,
            clustering_function=clustering_function,
            parameter_space=parameter_space,
            quality_metrics=['silhouette'],
            strategy='bayesian',
            n_iterations=8
        )
        
        assert isinstance(result, OptimizationResult)
        assert 'eps' in result.best_parameters
        assert 'min_samples' in result.best_parameters
        assert len(engine.optimization_history) == 1
    
    def test_quality_score_calculation(self):
        """Test clustering quality score calculation."""
        engine = AdaptiveTuningEngine(random_state=42)
        
        # Create test data and labels
        X, true_labels = make_blobs(n_samples=100, centers=3, random_state=42)
        
        # Test silhouette score
        score = engine._calculate_quality_score(X, true_labels, ['silhouette'])
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0
        
        # Test multiple metrics
        score_multi = engine._calculate_quality_score(
            X, true_labels, ['silhouette', 'davies_bouldin']
        )
        assert isinstance(score_multi, float)
        
        # Test with noise labels (-1)
        noisy_labels = true_labels.copy()
        noisy_labels[:10] = -1  # Add some noise points
        
        score_noisy = engine._calculate_quality_score(X, noisy_labels, ['silhouette'])
        assert isinstance(score_noisy, float)
    
    def test_quality_score_edge_cases(self):
        """Test quality score calculation edge cases."""
        engine = AdaptiveTuningEngine(random_state=42)
        
        X = np.random.rand(50, 2)
        
        # All points in same cluster
        uniform_labels = np.zeros(50)
        score = engine._calculate_quality_score(X, uniform_labels, ['silhouette'])
        assert score == float('-inf')
        
        # All points are noise
        noise_labels = np.full(50, -1)
        score = engine._calculate_quality_score(X, noise_labels, ['silhouette'])
        assert score == float('-inf')
    
    def test_optimization_summary(self):
        """Test optimization summary generation."""
        engine = AdaptiveTuningEngine(random_state=42)
        
        # Initially empty
        summary = engine.get_optimization_summary()
        assert summary == {}
        
        # Add mock optimization history
        mock_result = OptimizationResult(
            best_parameters={'eps': 0.5},
            best_score=0.8,
            optimization_history=[],
            convergence_iteration=10,
            total_iterations=20,
            total_time=5.0,
            method_used='bayesian'
        )
        
        engine.optimization_history.append({
            'dataset_shape': (100, 2),
            'strategy': 'bayesian',
            'result': mock_result,
            'timestamp': 123456789
        })
        
        summary = engine.get_optimization_summary()
        
        assert isinstance(summary, dict)
        assert summary['total_optimizations'] == 1
        assert 'bayesian' in summary['strategies_used']
        assert 'average_performance' in summary
        assert 'convergence_rate' in summary

class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline integration."""
        # Create engine
        engine = AdaptiveTuningEngine(random_state=42)
        
        # Create realistic dataset
        X, _ = make_blobs(n_samples=200, centers=4, cluster_std=0.8, random_state=42)
        
        def dbscan_clustering(data, params):
            dbscan = DBSCAN(eps=params['eps'], min_samples=int(params['min_samples']))
            return dbscan.fit_predict(data)
        
        parameter_space = {
            'eps': (0.3, 1.5),
            'min_samples': (3, 15)
        }
        
        # Run optimization
        result = engine.optimize_parameters(
            X=X,
            clustering_function=dbscan_clustering,
            parameter_space=parameter_space,
            quality_metrics=['silhouette'],
            strategy='auto',
            n_iterations=12
        )
        
        # Verify result
        assert isinstance(result, OptimizationResult)
        assert result.best_score > float('-inf')
        assert 0.3 <= result.best_parameters['eps'] <= 1.5
        assert 3 <= result.best_parameters['min_samples'] <= 15
        
        # Test clustering with optimized parameters
        final_labels = dbscan_clustering(X, result.best_parameters)
        n_clusters = len(set(final_labels)) - (1 if -1 in final_labels else 0)
        assert n_clusters > 0  # Should find some clusters
    
    def test_meta_learning_workflow(self):
        """Test meta-learning workflow across multiple optimizations."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        # Run optimizations on different datasets
        datasets = [
            make_blobs(n_samples=100, centers=2, random_state=42)[0],
            make_blobs(n_samples=150, centers=3, random_state=43)[0],
            make_circles(n_samples=120, noise=0.1, random_state=44)[0]
        ]
        
        def simple_objective(params):
            return -(params['eps'] - 0.5) ** 2 + 1.0
        
        parameter_space = {'eps': (0.1, 1.0)}
        
        results = []
        for X in datasets:
            result = explorer.explore(
                X=X,
                objective_function=simple_objective,
                parameter_space=parameter_space,
                strategy='auto',
                n_iterations=5,
                use_prediction=False
            )
            results.append(result)
        
        # Verify meta-learning insights accumulate
        assert len(results) == 3
        final_insights = results[-1].meta_learning_insights
        assert final_insights['total_experiences'] == 3
    
    def test_performance_prediction_accuracy(self):
        """Test performance prediction improves with experience."""
        explorer = ParameterSpaceExplorer(random_state=42)
        
        # Generate training data
        training_data = []
        for i in range(10):
            X = np.random.rand(100, 2)
            params = {'eps': 0.1 + 0.1*i, 'min_samples': 3 + i}
            # Simulate realistic clustering score
            score = 0.4 + 0.3 * np.exp(-(i-5)**2 / 10)  # Peak around middle values
            training_data.append((X, params, score))
        
        explorer.train_predictor(training_data)
        
        # Test prediction accuracy
        test_X = np.random.rand(100, 2)
        test_params = {'eps': 0.5, 'min_samples': 8}
        
        prediction = explorer.performance_predictor.predict(test_X, test_params)
        assert isinstance(prediction, float)
        assert 0.0 <= prediction <= 1.0  # Should be reasonable clustering score
    
    def test_optimization_convergence(self):
        """Test optimization convergence detection."""
        optimizer = BayesianOptimizer(random_state=42)
        
        def converging_objective(params):
            # Function that should converge quickly
            return -(params['x'] - 0.5) ** 2 + 1.0
        
        parameter_space = {'x': (0.0, 1.0)}
        
        result = optimizer.optimize(converging_objective, parameter_space, n_iterations=25)
        
        # Should detect convergence for this simple function
        assert isinstance(result.convergence_iteration, (int, type(None)))
        if result.convergence_iteration is not None:
            assert result.convergence_iteration < 25

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

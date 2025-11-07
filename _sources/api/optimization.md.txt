# Optimization Reference

Comprehensive documentation of optimization algorithms and strategies in the Enhanced Adaptive DBSCAN framework.

## Overview

The optimization module provides automatic parameter tuning capabilities to find optimal clustering parameters for different datasets and use cases. The framework supports multiple optimization strategies, from simple grid search to advanced Bayesian optimization.

## Parameter Optimization

### Core Parameters

The framework optimizes the following key parameters:

- **eps**: Neighborhood radius for density calculation
- **min_samples**: Minimum points required to form a cluster
- **algorithm**: Clustering algorithm variant
- **metric**: Distance metric for neighborhood calculation
- **adaptive_eps**: Enable/disable adaptive epsilon
- **multi_density**: Multi-density clustering parameters

### Parameter Spaces

Different parameter types require different optimization approaches:

```python
parameter_spaces = {
    'eps': {
        'type': 'continuous',
        'bounds': (0.1, 2.0),
        'log_scale': False
    },
    'min_samples': {
        'type': 'discrete',
        'values': [3, 5, 10, 15, 20, 30]
    },
    'metric': {
        'type': 'categorical',
        'values': ['euclidean', 'manhattan', 'cosine']
    }
}
```

## Optimization Algorithms

### Grid Search Optimization

Systematic exploration of parameter space through exhaustive search.

**Algorithm:**
```python
def grid_search_optimization(X, param_grid, scoring_func):
    """
    Grid search parameter optimization
    
    Args:
        X: Input data
        param_grid: Dictionary of parameter ranges
        scoring_func: Objective function to maximize
        
    Returns:
        best_params: Optimal parameter combination
        best_score: Best achieved score
        results: Full grid search results
    """
    best_score = -np.inf
    best_params = None
    results = []
    
    # Generate all parameter combinations
    param_combinations = generate_param_combinations(param_grid)
    
    for params in param_combinations:
        # Evaluate clustering with current parameters
        clusterer = EnhancedAdaptiveDBSCAN(**params)
        labels = clusterer.fit_predict(X)
        score = scoring_func(X, labels)
        
        results.append({
            'params': params,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
    
    return best_params, best_score, results
```

**Advantages:**
- Guaranteed to find global optimum (within grid)
- Simple to implement and understand
- Provides complete exploration results

**Disadvantages:**
- Exponential complexity with parameter count
- May miss optimal values between grid points
- Computationally expensive for fine grids

### Bayesian Optimization

Probabilistic optimization using Gaussian Process models to efficiently explore parameter space.

**Core Components:**

1. **Gaussian Process Model**: Models objective function
2. **Acquisition Function**: Guides parameter selection
3. **Optimization Loop**: Iteratively improves model

**Implementation:**
```python
class BayesianOptimizer:
    """Bayesian optimization for parameter tuning"""
    
    def __init__(self, parameter_space, acquisition='ei', 
                 n_initial=10, n_iterations=50):
        self.parameter_space = parameter_space
        self.acquisition = acquisition
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []
    
    def optimize(self, objective_function):
        """Run Bayesian optimization"""
        # Phase 1: Random initialization
        self._initialize_random(objective_function)
        
        # Phase 2: Bayesian optimization loop
        for i in range(self.n_iterations):
            # Fit GP model to observations
            self._fit_gp_model()
            
            # Find next point to evaluate
            next_params = self._select_next_point()
            
            # Evaluate objective function
            score = objective_function(next_params)
            
            # Update observations
            self._update_observations(next_params, score)
        
        # Return best parameters found
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]
    
    def _fit_gp_model(self):
        """Fit Gaussian Process model to observations"""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        kernel = Matern(length_scale=1.0, nu=2.5)
        self.gp_model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            random_state=42
        )
        
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp_model.fit(X, y)
    
    def _select_next_point(self):
        """Select next point using acquisition function"""
        if self.acquisition == 'ei':
            return self._expected_improvement()
        elif self.acquisition == 'ucb':
            return self._upper_confidence_bound()
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")
    
    def _expected_improvement(self):
        """Expected Improvement acquisition function"""
        # Generate candidate points
        candidates = self._generate_candidates(1000)
        
        # Calculate EI for each candidate
        ei_values = []
        current_best = np.max(self.y_observed)
        
        for candidate in candidates:
            mean, std = self.gp_model.predict([candidate], return_std=True)
            
            if std > 0:
                z = (mean - current_best) / std
                ei = (mean - current_best) * norm.cdf(z) + std * norm.pdf(z)
            else:
                ei = 0
            
            ei_values.append(ei)
        
        # Select candidate with highest EI
        best_idx = np.argmax(ei_values)
        return candidates[best_idx]
```

**Acquisition Functions:**

1. **Expected Improvement (EI)**:
   ```
   EI(x) = (μ(x) - f_best) * Φ(z) + σ(x) * φ(z)
   where z = (μ(x) - f_best) / σ(x)
   ```

2. **Upper Confidence Bound (UCB)**:
   ```
   UCB(x) = μ(x) + κ * σ(x)
   where κ controls exploration vs exploitation
   ```

3. **Probability of Improvement (PI)**:
   ```
   PI(x) = Φ((μ(x) - f_best) / σ(x))
   ```

### Genetic Algorithm Optimization

Evolutionary optimization inspired by natural selection and genetics.

**Genetic Operations:**

```python
class GeneticOptimizer:
    """Genetic algorithm for parameter optimization"""
    
    def __init__(self, parameter_space, population_size=50, 
                 generations=30, mutation_rate=0.1, crossover_rate=0.8):
        self.parameter_space = parameter_space
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
    
    def optimize(self, objective_function):
        """Run genetic algorithm optimization"""
        # Initialize population
        population = self._initialize_population()
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = [
                objective_function(individual) 
                for individual in population
            ]
            
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self._crossover(parents[i], parents[i+1])
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    offspring.extend([child1, child2])
            
            # Combine and select next generation
            population = self._survivor_selection(
                population + offspring, 
                objective_function
            )
        
        # Return best individual
        final_fitness = [objective_function(ind) for ind in population]
        best_idx = np.argmax(final_fitness)
        return population[best_idx], final_fitness[best_idx]
    
    def _crossover(self, parent1, parent2):
        """Uniform crossover operation"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if np.random.random() < self.crossover_rate:
            for key in parent1.keys():
                if np.random.random() < 0.5:
                    child1[key], child2[key] = child2[key], child1[key]
        
        return child1, child2
    
    def _mutate(self, individual):
        """Gaussian mutation for continuous parameters"""
        mutated = individual.copy()
        
        for key, value in individual.items():
            if np.random.random() < self.mutation_rate:
                param_info = self.parameter_space[key]
                
                if param_info['type'] == 'continuous':
                    # Gaussian mutation
                    bounds = param_info['bounds']
                    std = (bounds[1] - bounds[0]) * 0.1
                    mutated[key] = np.clip(
                        value + np.random.normal(0, std),
                        bounds[0], bounds[1]
                    )
                elif param_info['type'] == 'discrete':
                    # Random selection from valid values
                    mutated[key] = np.random.choice(param_info['values'])
        
        return mutated
```

**Selection Strategies:**

1. **Tournament Selection**: Compare random subsets
2. **Roulette Wheel**: Probability proportional to fitness
3. **Rank Selection**: Based on fitness ranking
4. **Elitism**: Preserve best individuals

## Multi-Objective Optimization

For scenarios with multiple competing objectives (e.g., clustering quality vs computational cost).

### Pareto Optimization

```python
class ParetoOptimizer:
    """Multi-objective optimization using Pareto dominance"""
    
    def __init__(self, objectives, weights=None):
        self.objectives = objectives
        self.weights = weights or [1.0] * len(objectives)
    
    def is_dominated(self, solution1, solution2):
        """Check if solution1 is dominated by solution2"""
        scores1 = [obj(solution1) for obj in self.objectives]
        scores2 = [obj(solution2) for obj in self.objectives]
        
        # Weighted comparison
        weighted_scores1 = [s * w for s, w in zip(scores1, self.weights)]
        weighted_scores2 = [s * w for s, w in zip(scores2, self.weights)]
        
        # Check dominance
        better_or_equal = all(s2 >= s1 for s1, s2 in 
                             zip(weighted_scores1, weighted_scores2))
        strictly_better = any(s2 > s1 for s1, s2 in 
                             zip(weighted_scores1, weighted_scores2))
        
        return better_or_equal and strictly_better
    
    def find_pareto_front(self, solutions):
        """Find Pareto optimal solutions"""
        pareto_front = []
        
        for candidate in solutions:
            is_dominated = False
            for other in solutions:
                if candidate != other and self.is_dominated(candidate, other):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
```

## Adaptive Optimization Strategies

### Meta-Learning Approach

Learn from previous optimization experiences to accelerate future optimizations.

```python
class MetaLearningOptimizer:
    """Meta-learning for parameter optimization"""
    
    def __init__(self, feature_extractor, base_optimizer):
        self.feature_extractor = feature_extractor
        self.base_optimizer = base_optimizer
        self.experience_database = []
    
    def optimize(self, X, y=None):
        """Optimize using meta-learning"""
        # Extract dataset features
        features = self.feature_extractor.extract(X, y)
        
        # Find similar datasets from experience
        similar_cases = self._find_similar_datasets(features)
        
        # Initialize optimization with prior knowledge
        if similar_cases:
            initial_params = self._aggregate_prior_knowledge(similar_cases)
            self.base_optimizer.set_initial_parameters(initial_params)
        
        # Run optimization
        best_params, best_score = self.base_optimizer.optimize(X)
        
        # Store experience
        self._store_experience(features, best_params, best_score)
        
        return best_params, best_score
    
    def _find_similar_datasets(self, features, k=5):
        """Find k most similar datasets from experience"""
        similarities = []
        
        for exp in self.experience_database:
            similarity = self._calculate_similarity(features, exp['features'])
            similarities.append((similarity, exp))
        
        # Return top k similar cases
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in similarities[:k]]
```

### Warm-Start Strategies

Initialize optimization with good starting points based on data characteristics.

```python
def warm_start_initialization(X, n_candidates=10):
    """Generate good initial parameter candidates"""
    candidates = []
    
    # Strategy 1: K-distance analysis
    k_distances = calculate_k_distances(X, k=4)
    eps_candidates = [
        np.percentile(k_distances, p) for p in [10, 25, 50, 75, 90]
    ]
    
    # Strategy 2: Nearest neighbor analysis
    nn_analysis = analyze_nearest_neighbors(X)
    min_samples_candidates = nn_analysis['suggested_min_samples']
    
    # Strategy 3: Density estimation
    density_regions = estimate_density_regions(X)
    adaptive_candidates = density_regions['suggested_adaptive']
    
    # Combine strategies
    for eps in eps_candidates:
        for min_samples in min_samples_candidates:
            for adaptive in adaptive_candidates:
                candidates.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'adaptive_eps': adaptive
                })
    
    return candidates[:n_candidates]
```

## Objective Functions

### Standard Clustering Metrics

```python
def silhouette_objective(X, labels):
    """Silhouette score objective (higher is better)"""
    if len(set(labels)) < 2:
        return -1.0
    return silhouette_score(X, labels)

def davies_bouldin_objective(X, labels):
    """Davies-Bouldin index objective (lower is better, so negate)"""
    if len(set(labels)) < 2:
        return -np.inf
    return -davies_bouldin_score(X, labels)

def calinski_harabasz_objective(X, labels):
    """Calinski-Harabasz index objective (higher is better)"""
    if len(set(labels)) < 2:
        return 0.0
    return calinski_harabasz_score(X, labels)
```

### Composite Objectives

```python
def composite_objective(X, labels, weights=None):
    """Weighted combination of multiple objectives"""
    if weights is None:
        weights = {'silhouette': 0.4, 'davies_bouldin': 0.3, 'calinski': 0.3}
    
    scores = {
        'silhouette': silhouette_objective(X, labels),
        'davies_bouldin': davies_bouldin_objective(X, labels),
        'calinski': calinski_harabasz_objective(X, labels) / 1000  # Normalize
    }
    
    # Weighted combination
    total_score = sum(
        weight * scores[metric] 
        for metric, weight in weights.items()
    )
    
    return total_score
```

### Custom Domain Objectives

```python
def anomaly_detection_objective(X, labels, known_anomalies=None):
    """Objective function for anomaly detection scenarios"""
    base_score = silhouette_objective(X, labels)
    
    if known_anomalies is not None:
        # Bonus for correctly identifying known anomalies as noise
        noise_points = (labels == -1)
        anomaly_detection_rate = np.mean(noise_points[known_anomalies])
        base_score += 0.5 * anomaly_detection_rate
    
    # Penalty for too many noise points
    noise_ratio = np.mean(labels == -1)
    if noise_ratio > 0.1:  # More than 10% noise is penalized
        base_score -= (noise_ratio - 0.1) * 2.0
    
    return base_score
```

## Optimization Pipelines

### Hierarchical Optimization

```python
class HierarchicalOptimizer:
    """Optimize parameters in multiple stages"""
    
    def __init__(self, stages):
        self.stages = stages
    
    def optimize(self, X):
        """Run hierarchical optimization"""
        best_params = {}
        
        for stage_name, stage_config in self.stages.items():
            print(f"Optimizing stage: {stage_name}")
            
            # Extract stage-specific parameters
            stage_params = stage_config['parameters']
            optimizer = stage_config['optimizer']
            
            # Use previous results as constraints
            if best_params:
                optimizer.set_constraints(best_params)
            
            # Optimize current stage
            stage_best = optimizer.optimize(X, stage_params)
            
            # Update best parameters
            best_params.update(stage_best)
        
        return best_params

# Example usage
stages = {
    'coarse': {
        'parameters': ['eps', 'min_samples'],
        'optimizer': GridSearchOptimizer(resolution='coarse')
    },
    'fine': {
        'parameters': ['eps', 'min_samples'],
        'optimizer': BayesianOptimizer(n_iterations=20)
    },
    'advanced': {
        'parameters': ['adaptive_eps', 'multi_density'],
        'optimizer': GeneticOptimizer(generations=15)
    }
}
```

### Ensemble Optimization

```python
class EnsembleOptimizer:
    """Combine multiple optimization strategies"""
    
    def __init__(self, optimizers, voting_strategy='weighted'):
        self.optimizers = optimizers
        self.voting_strategy = voting_strategy
    
    def optimize(self, X):
        """Run ensemble optimization"""
        results = []
        
        # Run each optimizer
        for name, optimizer in self.optimizers.items():
            best_params, best_score = optimizer.optimize(X)
            results.append({
                'name': name,
                'params': best_params,
                'score': best_score
            })
        
        # Combine results
        if self.voting_strategy == 'best':
            return self._select_best(results)
        elif self.voting_strategy == 'weighted':
            return self._weighted_combination(results)
        else:
            return self._consensus_voting(results)
```

## Performance Considerations

### Optimization Budget Management

```python
class BudgetManager:
    """Manage computational budget for optimization"""
    
    def __init__(self, max_evaluations=100, max_time=3600):
        self.max_evaluations = max_evaluations
        self.max_time = max_time
        self.evaluations_used = 0
        self.start_time = None
    
    def start_optimization(self):
        """Start budget tracking"""
        self.start_time = time.time()
        self.evaluations_used = 0
    
    def can_continue(self):
        """Check if optimization can continue"""
        if self.evaluations_used >= self.max_evaluations:
            return False
        
        if self.start_time and (time.time() - self.start_time) > self.max_time:
            return False
        
        return True
    
    def record_evaluation(self):
        """Record an evaluation"""
        self.evaluations_used += 1
```

### Early Stopping

```python
class EarlyStopping:
    """Early stopping for optimization convergence"""
    
    def __init__(self, patience=10, min_improvement=1e-4):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score = -np.inf
        self.wait = 0
    
    def should_stop(self, current_score):
        """Check if optimization should stop early"""
        if current_score > self.best_score + self.min_improvement:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
        
        return self.wait >= self.patience
```

## References

1. Brochu, E., et al. "A tutorial on Bayesian optimization of expensive cost functions." arXiv preprint arXiv:1012.2599 (2010).
2. Bergstra, J., Bengio, Y. "Random search for hyper-parameter optimization." JMLR 13 (2012).
3. Snoek, J., et al. "Practical Bayesian optimization of machine learning algorithms." NIPS 2012.
4. Deb, K., et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." IEEE Trans. Evolutionary Computation 6.2 (2002).

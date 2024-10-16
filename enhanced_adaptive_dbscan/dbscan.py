# enhanced_adaptive_dbscan/dbscan.py

import numpy as np
from sklearn.neighbors import KDTree
import plotly.express as px
from collections import deque, defaultdict
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import logging
import warnings

from .utils import count_neighbors

warnings.filterwarnings("ignore")

class EnhancedAdaptiveDBSCAN:
    def __init__(self, wafer_shape='circular', wafer_size=100, k=20, density_scaling=1.0,
                 buffer_ratio=0.1, min_scaling=5, max_scaling=10, n_jobs=-1,
                 max_points=100000, subsample_ratio=0.1, random_state=42,
                 additional_features=None, feature_weights=None, stability_threshold=0.5):
        """
        Initialize the Enhanced Adaptive DBSCAN with Stability-Based Clustering.

        Parameters:
        - wafer_shape (str): Shape of the wafer ('circular' or 'square').
        - wafer_size (float): Size of the wafer (radius for circular, side length for square).
        - k (int): Number of neighbors for density estimation.
        - density_scaling (float): Scaling factor for adaptive ε.
        - buffer_ratio (float): Fraction of wafer size to create a buffer zone near boundaries.
        - min_scaling (int): Minimum scaling factor for adaptive MinPts.
        - max_scaling (int): Maximum scaling factor for adaptive MinPts.
        - n_jobs (int): Number of parallel jobs for multiprocessing. -1 uses all available cores.
        - max_points (int): Threshold for maximum number of points before subsampling is applied.
        - subsample_ratio (float): Ratio of data to subsample when max_points is exceeded.
        - random_state (int): Seed for reproducibility in subsampling.
        - additional_features (list or None): List of additional feature indices to include in clustering.
        - feature_weights (list or None): Weights for additional features to balance their influence.
        - stability_threshold (float): Minimum stability score to retain a cluster.
        """
        self.wafer_shape = wafer_shape
        self.wafer_size = wafer_size
        self.k = k
        self.density_scaling = density_scaling
        self.buffer_ratio = buffer_ratio
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling
        self.n_jobs = n_jobs
        self.max_points = max_points
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state
        self.additional_features = additional_features
        self.feature_weights = feature_weights
        self.stability_threshold = stability_threshold

        # Initialize logging
        self.logger = logging.getLogger('EnhancedAdaptiveDBSCAN')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

        # Initialize other attributes
        self.labels_ = None
        self.scaler_ = None
        self.cluster_stability_ = {}    # {cluster_label: stability_score}
        self.cluster_centers_ = {}      # {cluster_label: centroid}
        self.cluster_sizes_ = defaultdict(int)  # {cluster_label: size}

        # Define wafer boundary
        self.define_boundary()

    def define_boundary(self):
        """Define wafer boundary based on its shape."""
        if self.wafer_shape == 'circular':
            self.center = np.array([0, 0])
            self.radius = self.wafer_size / 2
        elif self.wafer_shape == 'square':
            self.half_size = self.wafer_size / 2
        else:
            raise ValueError("Unsupported wafer shape. Choose 'circular' or 'square'.")

    def is_near_boundary(self, point):
        """Check if a point is near the wafer boundary."""
        if self.wafer_shape == 'circular':
            distance = np.linalg.norm(point - self.center)
            return distance >= (self.radius * (1 - self.buffer_ratio))
        elif self.wafer_shape == 'square':
            x, y = point
            return (abs(x) >= (self.half_size * (1 - self.buffer_ratio))) or \
                   (abs(y) >= (self.half_size * (1 - self.buffer_ratio)))

    def preprocess_features(self, X):
        """
        Incorporate additional features into the feature set.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)

        Returns:
        - X_processed (ndarray): Shape (n_samples, n_processed_features)
        """
        if self.additional_features:
            X_additional = X[:, self.additional_features]
            if self.feature_weights:
                weights = np.array(self.feature_weights)
                X_additional = X_additional * weights
            X_processed = np.hstack((X[:, :2], X_additional))
        else:
            X_processed = X[:, :2]  # Only spatial features

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_processed)
        return X_scaled

    def compute_local_density(self, X):
        """
        Compute local density using k-NN distances with KDTree for efficiency.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)

        Returns:
        - local_density (ndarray): Shape (n_samples,)
        """
        tree = KDTree(X)
        distances, _ = tree.query(X, k=self.k + 1)  # k+1 because the first neighbor is the point itself
        mean_distance = np.mean(distances[:, 1:], axis=1)  # Exclude the point itself
        local_density = 1 / (mean_distance + 1e-5)  # Avoid division by zero
        return local_density

    def adjust_density_near_boundary(self, X, local_density):
        """
        Adjust local density estimates near wafer boundaries.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - local_density (ndarray): Shape (n_samples,)

        Returns:
        - adjusted_density (ndarray): Shape (n_samples,)
        """
        near_boundary = np.array([self.is_near_boundary(point) for point in X[:, :2]])
        adjustment_factor = 1.2  # Parameterize if necessary
        local_density[near_boundary] *= adjustment_factor
        return local_density

    def compute_adaptive_parameters(self, local_density, additional_attributes=None):
        """
        Compute adaptive ε and MinPts based on local density and additional attributes.

        Parameters:
        - local_density (ndarray): Shape (n_samples,)
        - additional_attributes (ndarray or None): Shape (n_samples, n_additional_features)

        Returns:
        - epsilon (ndarray): Shape (n_samples,)
        - min_pts (ndarray): Shape (n_samples,)
        """
        # Adaptive ε
        epsilon = self.density_scaling / (local_density + 1e-5)
        # Clip ε to prevent extreme values
        median_epsilon = np.median(epsilon)
        epsilon = np.clip(epsilon, a_min=0.5 * median_epsilon, a_max=2 * median_epsilon)

        # Adaptive MinPts
        norm_density = (local_density - np.min(local_density)) / (np.max(local_density) - np.min(local_density) + 1e-5)
        min_pts = self.min_scaling + (self.max_scaling - self.min_scaling) * norm_density

        if additional_attributes is not None and self.feature_weights:
            # Example: Increase min_pts for higher severity defects
            # Assuming the first additional attribute is severity (scale 1-10)
            severity = additional_attributes[:, 0]
            severity_norm = (severity - np.min(severity)) / (np.max(severity) - np.min(severity) + 1e-5)
            min_pts += severity_norm * 2  # Parameterize if necessary
            min_pts = np.clip(min_pts, self.min_scaling, self.max_scaling)

        min_pts = np.round(min_pts).astype(int)
        return epsilon, min_pts

    def identify_core_points(self, X, epsilon, min_pts):
        """
        Identify core points based on adaptive ε and MinPts.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - epsilon (ndarray): Shape (n_samples,)
        - min_pts (ndarray): Shape (n_samples,)

        Returns:
        - core_points (ndarray): Boolean array indicating core points.
        """
        tree = KDTree(X)
        # Parallel computation using joblib
        counts = Parallel(n_jobs=self.n_jobs)(
            delayed(count_neighbors)(tree, X, epsilon, i) for i in range(len(X))
        )
        counts = np.array(counts)
        core_points = counts >= min_pts
        return core_points

    def form_clusters(self, X, epsilon, min_pts, core_points):
        """
        Form clusters using core points and adaptive parameters.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - epsilon (ndarray): Shape (n_samples,)
        - min_pts (ndarray): Shape (n_samples,)
        - core_points (ndarray): Boolean array indicating core points.

        Returns:
        - labels (ndarray): Cluster labels for each point.
        """
        n_points = X.shape[0]
        labels = np.full(n_points, -1, dtype=int)  # Initialize all labels to -1 (noise)
        cluster_id = 0

        tree = KDTree(X)

        # Precompute neighbors for all points
        neighbors_list = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda i: tree.query_radius(X[i].reshape(1, -1), r=epsilon[i])[0])(i) for i in range(n_points)
        )

        for i in range(n_points):
            if labels[i] != -1 or not core_points[i]:
                continue
            # Start a new cluster
            labels[i] = cluster_id
            queue = deque()
            queue.extend(neighbors_list[i])
            while queue:
                j = queue.popleft()
                if labels[j] == -1:
                    labels[j] = cluster_id
                    if core_points[j]:
                        queue.extend(neighbors_list[j])
            cluster_id += 1

        return labels

    def build_hierarchy(self, X, epsilon_levels, min_pts_levels):
        """
        Build a cluster hierarchy across multiple density levels and calculate stability scores.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - epsilon_levels (list of ndarrays): Each ndarray represents ε at a density level.
        - min_pts_levels (list of ndarrays): Each ndarray represents MinPts at a density level.

        Returns:
        - hierarchy (dict): Mapping levels to cluster labels.
        """
        hierarchy = {}
        cluster_presence = defaultdict(int)  # To track cluster persistence

        for level, (eps, min_p) in enumerate(zip(epsilon_levels, min_pts_levels)):
            core_points = self.identify_core_points(X, eps, min_p)
            labels = self.form_clusters(X, eps, min_p, core_points)
            hierarchy[level] = labels

            # Update cluster presence
            unique_clusters = set(labels)
            for cluster in unique_clusters:
                if cluster != -1:
                    cluster_presence[cluster] += 1

        self.cluster_hierarchy_ = hierarchy

        # Calculate stability scores
        for cluster, count in cluster_presence.items():
            stability = count / len(epsilon_levels)
            self.cluster_stability_[cluster] = stability

        return hierarchy

    def select_stable_clusters(self):
        """
        Select clusters based on their stability scores.
        Clusters with stability scores above the threshold are retained.
        """
        stable_clusters = {cluster for cluster, stability in self.cluster_stability_.items()
                           if stability >= self.stability_threshold}

        # Create a mapping from old labels to new labels
        label_mapping = {cluster: idx for idx, cluster in enumerate(sorted(stable_clusters))}

        # Assign new labels based on stability
        final_labels = np.full_like(self.labels_, -1)
        for i, label in enumerate(self.labels_):
            if label in label_mapping:
                final_labels[i] = label_mapping[label]

        self.labels_ = final_labels
        return self.labels_

    def subsample_data(self, X, stratify=None):
        """
        Subsample the dataset if it exceeds max_points using stratified subsampling.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - stratify (ndarray or None): Labels to stratify by.

        Returns:
        - subsampled_X (ndarray): Subsampled data.
        - subsample_indices (ndarray): Indices of the subsampled data.
        """
        n_samples = X.shape[0]
        if n_samples <= self.max_points:
            return X, np.arange(n_samples)
        else:
            n_subsampled = int(n_samples * self.subsample_ratio)
            np.random.seed(self.random_state)
            if stratify is not None:
                unique, counts = np.unique(stratify, return_counts=True)
                proportions = counts / counts.sum()
                n_subsampled_per_class = (proportions * n_subsampled).astype(int)
                subsample_indices = []
                for cls, n_cls in zip(unique, n_subsampled_per_class):
                    cls_indices = np.where(stratify == cls)[0]
                    if len(cls_indices) < n_cls:
                        subsample_indices.extend(cls_indices)
                    else:
                        subsample_indices.extend(np.random.choice(cls_indices, size=n_cls, replace=False))
                subsample_indices = np.array(subsample_indices)
            else:
                # Random subsampling
                subsample_indices = np.random.choice(n_samples, size=n_subsampled, replace=False)
            subsampled_X = X[subsample_indices]
            self.subsample_indices_ = subsample_indices
            return subsampled_X, subsample_indices

    def assign_full_data_clusters(self, X_full, X_sub, labels_sub):
        """
        Assign cluster labels to the full dataset based on the subsampled clusters.

        Parameters:
        - X_full (ndarray): Shape (n_samples_full, n_features)
        - X_sub (ndarray): Shape (n_samples_sub, n_features)
        - labels_sub (ndarray): Cluster labels for subsampled data.

        Returns:
        - labels_full (ndarray): Cluster labels for the full dataset.
        """
        # Remove noise points from subsampled data
        mask = labels_sub != -1
        X_sub_core = X_sub[mask]
        labels_sub_core = labels_sub[mask]

        if len(X_sub_core) == 0:
            # If no core points in subsample, label all as noise
            return np.full(X_full.shape[0], -1, dtype=int)

        # Build KD-Tree for subsampled core points
        tree = KDTree(X_sub_core)

        # Query the nearest cluster center for each full data point
        distances, indices = tree.query(X_full, k=1)
        nearest_labels = labels_sub_core[indices.flatten()]

        # Assign labels
        labels_full = nearest_labels

        return labels_full

    def update_cluster_centers(self, new_points, cluster_label):
        """
        Update the centroid of a cluster with new points.

        Parameters:
        - new_points (ndarray): Shape (n_new_points, n_features)
        - cluster_label (int): Label of the cluster to update.
        """
        if cluster_label not in self.cluster_centers_:
            # Initialize centroid and size
            self.cluster_centers_[cluster_label] = new_points.mean(axis=0)
            self.cluster_sizes_[cluster_label] = len(new_points)
        else:
            # Update centroid incrementally
            current_centroid = self.cluster_centers_[cluster_label]
            current_size = self.cluster_sizes_[cluster_label]
            new_size = current_size + len(new_points)
            new_centroid = (current_centroid * current_size + new_points.sum(axis=0)) / new_size
            self.cluster_centers_[cluster_label] = new_centroid
            self.cluster_sizes_[cluster_label] = new_size

    def partial_reclustering(self, affected_clusters, X, epsilon, min_pts):
        """
        Perform partial reclustering on affected clusters.

        Parameters:
        - affected_clusters (set): Set of cluster labels that are affected.
        - X (ndarray): Shape (n_samples, n_features)
        - epsilon (ndarray): Shape (n_samples,)
        - min_pts (ndarray): Shape (n_samples,)
        """
        for cluster in affected_clusters:
            # Extract points belonging to the cluster
            cluster_mask = self.labels_ == cluster
            cluster_points = X[cluster_mask]

            if len(cluster_points) < self.min_scaling:
                # If cluster is too small, mark as noise
                self.labels_[cluster_mask] = -1
                del self.cluster_centers_[cluster]
                del self.cluster_sizes_[cluster]
                continue

            # Recompute local density for cluster points
            local_density = self.compute_local_density(cluster_points)
            # Adjust density near boundaries
            local_density = self.adjust_density_near_boundary(cluster_points, local_density)
            # Recompute adaptive parameters
            epsilon_cluster, min_pts_cluster = self.compute_adaptive_parameters(local_density, 
                                                                                additional_attributes=None if self.additional_features is None else cluster_points[:, 2].reshape(-1, 1))
            # Identify core points within the cluster
            core_points_cluster = self.identify_core_points(cluster_points, epsilon_cluster, min_pts_cluster)
            # Form new sub-clusters within the affected cluster
            labels_cluster = self.form_clusters(cluster_points, epsilon_cluster, min_pts_cluster, core_points_cluster)

            # Update cluster labels
            unique_sub_clusters = set(labels_cluster)
            for sub_cluster in unique_sub_clusters:
                if sub_cluster == -1:
                    continue  # Ignore noise
                # Assign a new unique label
                new_label = self.labels_.max() + 1
                self.labels_[cluster_mask][labels_cluster == sub_cluster] = new_label
                # Update cluster centers
                new_points = cluster_points[labels_cluster == sub_cluster]
                self.update_cluster_centers(new_points, new_label)

            # Remove the old cluster center and size
            del self.cluster_centers_[cluster]
            del self.cluster_sizes_[cluster]

    def fit(self, X, additional_attributes=None):
        """
        Fit the Enhanced Adaptive DBSCAN algorithm on data X.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - additional_attributes (ndarray or None): Shape (n_samples, n_additional_features)

        Returns:
        - self
        """
        X = np.array(X)
        n_points = X.shape[0]

        # Incorporate additional features if provided
        if self.additional_features:
            if additional_attributes is None:
                raise ValueError("additional_attributes must be provided when additional_features are used.")
            X_processed = self.preprocess_features(X)
        else:
            X_processed = X[:, :2]
            if self.scaler_ is None:
                self.scaler_ = StandardScaler()
                X_processed = self.scaler_.fit_transform(X_processed)
            else:
                X_processed = self.scaler_.transform(X_processed)

        # Step 1: Subsampling if necessary
        if self.additional_features and additional_attributes is not None:
            stratify_labels = additional_attributes[:, 0]
        else:
            stratify_labels = None

        X_sub, subsample_indices = self.subsample_data(X_processed, stratify=stratify_labels)

        if self.additional_features and additional_attributes is not None:
            additional_attrs_sub = additional_attributes[subsample_indices]
        else:
            additional_attrs_sub = None

        # Step 2: Local Density Estimation on Subsample
        local_density = self.compute_local_density(X_sub)

        # Step 3: Adjust density near boundaries on Subsample
        local_density = self.adjust_density_near_boundary(X_sub, local_density)

        # Step 4: Compute adaptive ε and MinPts on Subsample
        epsilon, min_pts = self.compute_adaptive_parameters(local_density, additional_attributes=additional_attrs_sub)

        # Step 5: Identify core points on Subsample
        core_points = self.identify_core_points(X_sub, epsilon, min_pts)

        # Step 6: Form clusters on Subsample
        labels_sub = self.form_clusters(X_sub, epsilon, min_pts, core_points)

        # **Assign labels_sub to self.labels_**
        self.labels_ = labels_sub

        # Step 7: Multi-Scale Clustering on Subsample
        scaling_factors = np.linspace(0.8, 1.2, num=5)
        epsilon_levels = [epsilon * scale for scale in scaling_factors]
        min_pts_levels = [np.round(min_pts * scale).astype(int) for scale in scaling_factors]

        hierarchy = self.build_hierarchy(X_sub, epsilon_levels, min_pts_levels)

        # Step 8: Select stable clusters based on stability scores
        self.select_stable_clusters()

        # Step 9: Initialize cluster centers based on stable clusters
        stable_labels = set(self.labels_)
        stable_labels.discard(-1)  # Remove noise

        for label in stable_labels:
            points_in_cluster = X_sub[self.labels_ == label]
            if len(points_in_cluster) == 0:
                continue
            centroid = points_in_cluster.mean(axis=0)
            self.cluster_centers_[label] = centroid
            self.cluster_sizes_[label] = len(points_in_cluster)

        # Step 10: Assign clusters to full dataset
        if n_points > self.max_points:
            X_full = X_processed
            X_sub_core = X_sub[self.labels_ != -1]
            labels_sub_core = self.labels_[self.labels_ != -1]

            if len(X_sub_core) == 0:
                # If no core points in subsample, label all as noise
                labels_full = np.full(n_points, -1, dtype=int)
            else:
                # Assign clusters based on nearest subsampled core point
                labels_full = self.assign_full_data_clusters(X_full, X_sub_core, labels_sub_core)

            self.labels_ = labels_full
        else:
            # Use subsampled clusters as full clusters
            self.labels_ = labels_sub

        self.logger.info("Fitting completed.")
        return self

    def fit_incremental(self, X_new, additional_attributes_new=None):
        """
        Incrementally fit the model with new data points.

        Parameters:
        - X_new (ndarray): Shape (n_new_samples, n_features)
        - additional_attributes_new (ndarray or None): Shape (n_new_samples, n_additional_features)

        Returns:
        - self
        """
        if self.scaler_ is None:
            raise ValueError("The model must be fitted before calling fit_incremental.")

        X_new = np.array(X_new)
        n_new = X_new.shape[0]

        # Preprocess new data points
        if self.additional_features:
            if additional_attributes_new is None:
                raise ValueError("additional_attributes_new must be provided when additional_features are used.")
            X_new_processed = self.preprocess_features(X_new)
        else:
            X_new_processed = X_new[:, :2]
            X_new_processed = self.scaler_.transform(X_new_processed)

        # Extract current cluster centers and labels
        if not self.cluster_centers_:
            # If no clusters exist, perform initial clustering
            self.fit(X_new, additional_attributes_new)
            return self

        current_labels = list(self.cluster_centers_.keys())
        current_centroids = np.array([self.cluster_centers_[label] for label in current_labels])
        tree_centers = KDTree(current_centroids)

        # Assign new points to nearest cluster centers
        distances, indices = tree_centers.query(X_new_processed, k=1)
        assigned_labels = [current_labels[idx] for idx in indices.flatten()]

        # Define a threshold for assignment (could be based on cluster-specific ε or a global threshold)
        # Here, we use a global threshold based on median epsilon
        if len(self.cluster_centers_) > 0:
            # Calculate a global epsilon as the median of current epsilon values
            # Alternatively, use cluster-specific epsilon if stored
            epsilon_threshold = self.density_scaling / (self.k)  # Example heuristic
        else:
            epsilon_threshold = self.density_scaling / (self.k)

        within_epsilon = distances.flatten() <= epsilon_threshold

        # Initialize labels list if necessary
        if self.labels_ is None:
            self.labels_ = np.array([])

        # Assign labels accordingly
        new_labels = []
        affected_clusters = set()
        for i, (label, within) in enumerate(zip(assigned_labels, within_epsilon)):
            if within:
                new_labels.append(label)
                affected_clusters.add(label)
                # Update cluster center
                self.update_cluster_centers(X_new_processed[i].reshape(1, -1), label)
                self.cluster_sizes_[label] += 1
            else:
                # Assign as noise initially
                new_labels.append(-1)

        # Append new labels to existing labels
        self.labels_ = np.concatenate([self.labels_, new_labels])

        # Identify new core points from the new assignments
        noise_mask = np.array(new_labels) == -1
        if np.any(noise_mask):
            noise_points = X_new_processed[noise_mask]
            n_noise = noise_points.shape[0]

            if n_noise == 0:
                return self

            tree_noise = KDTree(noise_points)
            local_density_noise = self.compute_local_density(noise_points)
            local_density_noise = self.adjust_density_near_boundary(noise_points, local_density_noise)
            epsilon_noise, min_pts_noise = self.compute_adaptive_parameters(local_density_noise, 
                                                                            additional_attributes=additional_attributes_new[noise_mask] if self.additional_features else None)
            core_points_noise = self.identify_core_points(noise_points, epsilon_noise, min_pts_noise)
            labels_noise = self.form_clusters(noise_points, epsilon_noise, min_pts_noise, core_points_noise)

            for i, label in enumerate(labels_noise):
                if label != -1:
                    # Assign a new unique label
                    new_label = self.labels_.max() + 1
                    self.labels_[-n_noise + i] = new_label
                    # Update cluster centers
                    self.update_cluster_centers(noise_points[i].reshape(1, -1), new_label)

        # Perform partial reclustering on affected clusters
        if affected_clusters:
            self.partial_reclustering(affected_clusters, self.scaler_.transform(X_new), 
                                      epsilon_threshold, 
                                      self.min_scaling)

        self.logger.info("Incremental fitting completed.")
        return self

    def plot_clusters(self, X, plot_all=False, title='Enhanced Adaptive DBSCAN Clustering'):
        """
        Plot the clustering results using Plotly for interactive visualization.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - plot_all (bool): If True, plots all data points. Otherwise, plots a subset for clarity.
        - title (str): Plot title.
        """
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")

        if not plot_all and X.shape[0] > 5000:
            # Plot a random subset of 5000 points for clarity
            np.random.seed(self.random_state)
            plot_indices = np.random.choice(X.shape[0], size=5000, replace=False)
            X_plot = X[plot_indices]
            labels_plot = self.labels_[plot_indices]
        else:
            X_plot = X
            labels_plot = self.labels_

        # Create a DataFrame for Plotly
        import pandas as pd
        df = pd.DataFrame({
            'X': X_plot[:, 0],
            'Y': X_plot[:, 1],
            'Cluster': labels_plot.astype(str)  # Convert to string for better color handling
        })

        # Define color palette
        unique_labels = df['Cluster'].unique()
        if '-1' in unique_labels:
            # Remove '-1' from unique_labels using boolean indexing
            unique_labels = unique_labels[unique_labels != '-1']
            # Alternatively, convert to list and use list comprehension
            # unique_labels = [label for label in unique_labels if label != '-1']
            
            color_discrete_map = {label: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)] 
                                for i, label in enumerate(sorted(unique_labels))}
            color_discrete_map['-1'] = 'black'  # Noise
        else:
            color_discrete_map = {label: px.colors.qualitative.Dark24[i % len(px.colors.qualitative.Dark24)] 
                                for i, label in enumerate(sorted(unique_labels))}

        fig = px.scatter(df, x='X', y='Y', color='Cluster',
                        title=title,
                        color_discrete_map=color_discrete_map,
                        opacity=0.6,
                        hover_data=['Cluster'])

        fig.update_layout(showlegend=True)
        fig.show()


    def evaluate_clustering(self, X, labels):
        """
        Evaluate clustering using internal metrics.

        Parameters:
        - X (ndarray): Shape (n_samples, n_features)
        - labels (ndarray): Cluster labels.

        Prints:
        - Silhouette Score
        - Davies-Bouldin Index
        """
        mask = labels != -1
        if len(set(labels[mask])) < 2:
            print("Not enough clusters to compute silhouette and Davies-Bouldin scores.")
            return
        silhouette = silhouette_score(X[mask], labels[mask])
        davies_bouldin = davies_bouldin_score(X[mask], labels[mask])
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f}")

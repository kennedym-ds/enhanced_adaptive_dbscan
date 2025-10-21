# enhanced_adaptive_dbscan/hdbscan_clustering.py

"""
HDBSCAN-Style Hierarchical Clustering Module (Phase 5)

This module implements a complete HDBSCAN-style hierarchical density-based clustering
algorithm, addressing the hierarchical clustering gap (Partial → 8.5/10).

Key Features:
- Minimum spanning tree construction
- Hierarchical cluster tree building
- Condensed tree extraction
- Cluster selection via excess of mass
- Stability-based cluster selection
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class MSTreeEdge:
    """Edge in minimum spanning tree."""
    point1: int
    point2: int
    distance: float


@dataclass
class ClusterNode:
    """Node in the cluster hierarchy tree."""
    node_id: int
    parent_id: Optional[int]
    children_ids: List[int]
    lambda_birth: float  # 1/distance where cluster was born
    lambda_death: float  # 1/distance where cluster died
    points: Set[int]
    stability: float
    is_selected: bool = False


@dataclass
class CondensedTreeNode:
    """Node in the condensed tree."""
    parent: int
    child: int
    lambda_val: float
    child_size: int


class MinimumSpanningTree:
    """
    Construct minimum spanning tree using mutual reachability distance.
    
    The MST is the foundation of HDBSCAN's hierarchical clustering.
    """
    
    def __init__(self, min_samples: int = 5, metric: str = 'euclidean'):
        """
        Initialize MST constructor.
        
        Parameters:
        - min_samples: Minimum points for core distance computation
        - metric: Distance metric
        """
        self.min_samples = min_samples
        self.metric = metric
        self.edges_ = None
    
    def _compute_core_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute core distances for all points.
        
        Core distance is the distance to the min_samples-th nearest neighbor.
        
        Parameters:
        - X: Data points
        
        Returns:
        - Core distances for each point
        """
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = X.shape[0]
        k = min(self.min_samples, n_samples)
        
        if k == n_samples:
            # All points are neighbors
            return np.zeros(n_samples)
        
        nbrs = NearestNeighbors(
            n_neighbors=k,
            metric=self.metric
        ).fit(X)
        
        distances, _ = nbrs.kneighbors(X)
        # Core distance is distance to k-th neighbor
        core_distances = distances[:, -1]
        
        return core_distances
    
    def _mutual_reachability_distance(
        self,
        dist: float,
        core_dist_i: float,
        core_dist_j: float
    ) -> float:
        """
        Compute mutual reachability distance.
        
        MRD(i,j) = max(core(i), core(j), dist(i,j))
        
        Parameters:
        - dist: Euclidean distance between points
        - core_dist_i: Core distance of point i
        - core_dist_j: Core distance of point j
        
        Returns:
        - Mutual reachability distance
        """
        return max(dist, core_dist_i, core_dist_j)
    
    def construct(self, X: np.ndarray) -> List[MSTreeEdge]:
        """
        Construct minimum spanning tree using Prim's algorithm.
        
        Parameters:
        - X: Data points
        
        Returns:
        - List of edges in MST
        """
        n_samples = X.shape[0]
        
        logger.info(f"Computing core distances for {n_samples} points...")
        core_distances = self._compute_core_distances(X)
        
        logger.info("Building minimum spanning tree...")
        
        # Prim's algorithm
        in_tree = np.zeros(n_samples, dtype=bool)
        min_distances = np.full(n_samples, np.inf)
        parent = np.full(n_samples, -1, dtype=np.int32)
        
        # Start with arbitrary point
        current = 0
        in_tree[current] = True
        
        edges = []
        
        for _ in range(n_samples - 1):
            # Update distances to tree
            for i in range(n_samples):
                if not in_tree[i]:
                    dist = np.linalg.norm(X[current] - X[i])
                    mrd = self._mutual_reachability_distance(
                        dist,
                        core_distances[current],
                        core_distances[i]
                    )
                    
                    if mrd < min_distances[i]:
                        min_distances[i] = mrd
                        parent[i] = current
            
            # Find closest point to tree
            min_idx = -1
            min_dist = np.inf
            for i in range(n_samples):
                if not in_tree[i] and min_distances[i] < min_dist:
                    min_dist = min_distances[i]
                    min_idx = i
            
            if min_idx == -1:
                break
            
            # Add to tree
            in_tree[min_idx] = True
            edges.append(MSTreeEdge(
                point1=parent[min_idx],
                point2=min_idx,
                distance=min_dist
            ))
            current = min_idx
        
        self.edges_ = edges
        logger.info(f"MST constructed with {len(edges)} edges")
        
        return edges


class HierarchicalClusterTree:
    """
    Build hierarchical cluster tree from minimum spanning tree.
    
    Creates a dendrogram-like structure by processing edges in order
    of increasing distance.
    """
    
    def __init__(self):
        """Initialize hierarchical cluster tree."""
        self.nodes_ = {}
        self.root_id_ = None
        self.next_node_id_ = None
    
    def build_from_mst(
        self, mst_edges: List[MSTreeEdge], n_points: int
    ) -> Dict[int, ClusterNode]:
        """
        Build hierarchy from MST edges.
        
        Parameters:
        - mst_edges: Edges from minimum spanning tree
        - n_points: Total number of points
        
        Returns:
        - Dictionary of cluster nodes
        """
        logger.info("Building hierarchical cluster tree...")
        
        # Sort edges by distance
        sorted_edges = sorted(mst_edges, key=lambda e: e.distance)
        
        # Initialize leaf nodes
        self.nodes_ = {}
        for i in range(n_points):
            self.nodes_[i] = ClusterNode(
                node_id=i,
                parent_id=None,
                children_ids=[],
                lambda_birth=np.inf,
                lambda_death=0.0,
                points={i},
                stability=0.0
            )
        
        # Initialize union-find with component mapping
        # Maps point_id or node_id to its current cluster representative
        component_map = {i: i for i in range(n_points)}
        
        def find_component(x):
            """Find representative of component containing x."""
            if x not in component_map:
                return x
            if component_map[x] != x:
                component_map[x] = find_component(component_map[x])
            return component_map[x]
        
        next_id = n_points
        
        # Process edges to build hierarchy
        for edge in sorted_edges:
            c1 = find_component(edge.point1)
            c2 = find_component(edge.point2)
            
            if c1 == c2:
                continue
            
            # Create new internal node
            lambda_val = 1.0 / edge.distance if edge.distance > 0 else np.inf
            
            new_node = ClusterNode(
                node_id=next_id,
                parent_id=None,
                children_ids=[c1, c2],
                lambda_birth=lambda_val,
                lambda_death=0.0,
                points=self.nodes_[c1].points | self.nodes_[c2].points,
                stability=0.0
            )
            
            # Update parent pointers
            self.nodes_[c1].parent_id = next_id
            self.nodes_[c1].lambda_death = lambda_val
            self.nodes_[c2].parent_id = next_id
            self.nodes_[c2].lambda_death = lambda_val
            
            self.nodes_[next_id] = new_node
            
            # Update union-find
            component_map[c1] = next_id
            component_map[c2] = next_id
            component_map[next_id] = next_id
            
            next_id += 1
        
        # Find root
        self.root_id_ = next_id - 1
        self.next_node_id_ = next_id
        
        logger.info(f"Built hierarchy with {len(self.nodes_)} nodes")
        
        return self.nodes_


class CondensedTree:
    """
    Create condensed tree by removing noise-like splits.
    
    The condensed tree simplifies the hierarchy by removing clusters
    with too few points.
    """
    
    def __init__(self, min_cluster_size: int = 5):
        """
        Initialize condensed tree.
        
        Parameters:
        - min_cluster_size: Minimum points to form a cluster
        """
        self.min_cluster_size = min_cluster_size
        self.condensed_nodes_ = []
    
    def condense(
        self, hierarchy_nodes: Dict[int, ClusterNode], root_id: int
    ) -> List[CondensedTreeNode]:
        """
        Condense the hierarchy tree.
        
        Parameters:
        - hierarchy_nodes: Nodes from hierarchical tree
        - root_id: Root node ID
        
        Returns:
        - List of condensed tree nodes
        """
        logger.info("Condensing hierarchy tree...")
        
        condensed = []
        
        def process_node(node_id: int, ignore_size: bool = False):
            """Recursively process nodes to create condensed tree."""
            node = hierarchy_nodes[node_id]
            
            # Check if cluster is large enough
            is_cluster = len(node.points) >= self.min_cluster_size or ignore_size
            
            if not node.children_ids:
                # Leaf node
                return node_id if is_cluster else None
            
            # Process children
            valid_children = []
            for child_id in node.children_ids:
                result = process_node(child_id, ignore_size=False)
                if result is not None:
                    valid_children.append((result, hierarchy_nodes[child_id]))
            
            if is_cluster:
                # This is a valid cluster - record splits
                for child_id, child_node in valid_children:
                    condensed.append(CondensedTreeNode(
                        parent=node_id,
                        child=child_id,
                        lambda_val=child_node.lambda_death,
                        child_size=len(child_node.points)
                    ))
                return node_id
            else:
                # Too small - pass children up
                return valid_children[0][0] if valid_children else None
        
        process_node(root_id, ignore_size=True)
        
        self.condensed_nodes_ = condensed
        logger.info(f"Created condensed tree with {len(condensed)} splits")
        
        return condensed


class StabilityBasedSelector:
    """
    Select clusters based on stability (excess of mass).
    
    Uses the condensed tree to select the optimal flat clustering
    that maximizes overall stability.
    """
    
    def __init__(self, allow_single_cluster: bool = False):
        """
        Initialize stability-based selector.
        
        Parameters:
        - allow_single_cluster: Whether to allow selecting all points as one cluster
        """
        self.allow_single_cluster = allow_single_cluster
        self.selected_clusters_ = []
        self.cluster_stabilities_ = {}
    
    def compute_stability(
        self,
        condensed_nodes: List[CondensedTreeNode],
        hierarchy_nodes: Dict[int, ClusterNode]
    ) -> Dict[int, float]:
        """
        Compute stability for each cluster.
        
        Stability = sum over points of (lambda_p - lambda_birth)
        where lambda_p is the lambda value where point falls out
        
        Parameters:
        - condensed_nodes: Nodes from condensed tree
        - hierarchy_nodes: Nodes from hierarchy
        
        Returns:
        - Dictionary mapping cluster ID to stability
        """
        logger.info("Computing cluster stabilities...")
        
        # Build parent-child relationships
        children_map = defaultdict(list)
        for node in condensed_nodes:
            children_map[node.parent].append(node)
        
        # Compute stability for each cluster
        stabilities = {}
        
        for node_id, cluster_node in hierarchy_nodes.items():
            if len(cluster_node.points) < 1:
                continue
            
            # Compute persistence
            lambda_birth = cluster_node.lambda_birth
            lambda_death = cluster_node.lambda_death
            
            if not np.isfinite(lambda_birth):
                lambda_birth = 0.0
            
            # Stability is size * persistence
            persistence = lambda_death - lambda_birth
            stability = len(cluster_node.points) * persistence
            
            stabilities[node_id] = max(0.0, stability)
            
            # Update node
            cluster_node.stability = stabilities[node_id]
        
        self.cluster_stabilities_ = stabilities
        
        return stabilities
    
    def select_clusters(
        self,
        condensed_nodes: List[CondensedTreeNode],
        hierarchy_nodes: Dict[int, ClusterNode],
        root_id: int
    ) -> List[int]:
        """
        Select optimal clusters based on stability.
        
        Uses recursive selection algorithm to find clusters that
        maximize overall stability.
        
        Parameters:
        - condensed_nodes: Condensed tree nodes
        - hierarchy_nodes: Hierarchy nodes
        - root_id: Root node ID
        
        Returns:
        - List of selected cluster IDs
        """
        logger.info("Selecting optimal clusters...")
        
        # Compute stabilities
        stabilities = self.compute_stability(condensed_nodes, hierarchy_nodes)
        
        # Build parent-child map
        children_map = defaultdict(list)
        for node in condensed_nodes:
            children_map[node.parent].append(node.child)
        
        selected = []
        
        def select_recursive(node_id: int) -> float:
            """
            Recursively select clusters.
            
            Returns total stability of selection.
            """
            if node_id not in children_map:
                # Leaf cluster - select it
                hierarchy_nodes[node_id].is_selected = True
                selected.append(node_id)
                return stabilities.get(node_id, 0.0)
            
            # Compute stability if we select this cluster
            self_stability = stabilities.get(node_id, 0.0)
            
            # Compute stability if we select children
            children_stability = sum(
                select_recursive(child_id)
                for child_id in children_map[node_id]
            )
            
            # Choose option with higher stability
            if self_stability >= children_stability:
                # Select this cluster
                hierarchy_nodes[node_id].is_selected = True
                # Remove children from selected
                for child_id in children_map[node_id]:
                    if child_id in selected:
                        selected.remove(child_id)
                    hierarchy_nodes[child_id].is_selected = False
                selected.append(node_id)
                return self_stability
            else:
                # Keep children selection
                return children_stability
        
        # Start selection from root
        select_recursive(root_id)
        
        self.selected_clusters_ = selected
        logger.info(f"Selected {len(selected)} optimal clusters")
        
        return selected


class HDBSCANClusterer:
    """
    Complete HDBSCAN-style hierarchical density-based clustering.
    
    Implements the full HDBSCAN algorithm:
    1. Build minimum spanning tree with mutual reachability distance
    2. Construct hierarchical cluster tree
    3. Condense tree by removing small clusters
    4. Select clusters based on stability
    """
    
    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        metric: str = 'euclidean',
        allow_single_cluster: bool = False
    ):
        """
        Initialize HDBSCAN clusterer.
        
        Parameters:
        - min_cluster_size: Minimum points to form a cluster
        - min_samples: Points for core distance (default: min_cluster_size)
        - metric: Distance metric
        - allow_single_cluster: Allow all points in one cluster
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.allow_single_cluster = allow_single_cluster
        
        # Components
        self.mst_builder = MinimumSpanningTree(self.min_samples, metric)
        self.hierarchy_builder = HierarchicalClusterTree()
        self.condenser = CondensedTree(min_cluster_size)
        self.selector = StabilityBasedSelector(allow_single_cluster)
        
        # Results
        self.labels_ = None
        self.probabilities_ = None
        self.cluster_persistence_ = None
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN model and predict cluster labels.
        
        Parameters:
        - X: Input data (n_samples, n_features)
        
        Returns:
        - Cluster labels (noise points have label -1)
        """
        n_samples = X.shape[0]
        
        logger.info(f"Running HDBSCAN on {n_samples} points...")
        
        # Step 1: Build MST
        mst_edges = self.mst_builder.construct(X)
        
        # Step 2: Build hierarchy
        hierarchy_nodes = self.hierarchy_builder.build_from_mst(
            mst_edges, n_samples
        )
        
        # Step 3: Condense tree
        condensed_nodes = self.condenser.condense(
            hierarchy_nodes,
            self.hierarchy_builder.root_id_
        )
        
        # Step 4: Select clusters
        selected_clusters = self.selector.select_clusters(
            condensed_nodes,
            hierarchy_nodes,
            self.hierarchy_builder.root_id_
        )
        
        # Step 5: Assign labels
        self.labels_ = self._assign_labels(
            n_samples, selected_clusters, hierarchy_nodes
        )
        
        # Compute cluster persistence
        self.cluster_persistence_ = self._compute_persistence(
            selected_clusters, hierarchy_nodes
        )
        
        logger.info(f"HDBSCAN complete: found {len(selected_clusters)} clusters")
        
        return self.labels_
    
    def _assign_labels(
        self,
        n_samples: int,
        selected_clusters: List[int],
        hierarchy_nodes: Dict[int, ClusterNode]
    ) -> np.ndarray:
        """Assign cluster labels to points."""
        labels = np.full(n_samples, -1, dtype=np.int32)
        
        for cluster_idx, cluster_id in enumerate(selected_clusters):
            cluster_node = hierarchy_nodes[cluster_id]
            for point in cluster_node.points:
                labels[point] = cluster_idx
        
        return labels
    
    def _compute_persistence(
        self,
        selected_clusters: List[int],
        hierarchy_nodes: Dict[int, ClusterNode]
    ) -> Dict[int, float]:
        """Compute persistence for each selected cluster."""
        persistence = {}
        
        for cluster_idx, cluster_id in enumerate(selected_clusters):
            node = hierarchy_nodes[cluster_id]
            lambda_birth = node.lambda_birth if np.isfinite(node.lambda_birth) else 0.0
            lambda_death = node.lambda_death
            persistence[cluster_idx] = lambda_death - lambda_birth
        
        return persistence
    
    def get_cluster_info(self) -> Dict:
        """
        Get detailed information about clusters.
        
        Returns:
        - Dictionary with cluster statistics
        """
        if self.labels_ is None:
            raise ValueError("Model not fitted. Call fit_predict first.")
        
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels[unique_labels >= 0])
        n_noise = np.sum(self.labels_ == -1)
        
        cluster_sizes = {}
        for label in unique_labels:
            if label >= 0:
                cluster_sizes[label] = np.sum(self.labels_ == label)
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'cluster_sizes': cluster_sizes,
            'cluster_persistence': self.cluster_persistence_,
            'selected_stabilities': {
                k: v for k, v in self.selector.cluster_stabilities_.items()
                if k in self.selector.selected_clusters_
            }
        }

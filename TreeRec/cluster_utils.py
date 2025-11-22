import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node
# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)
# Set numpy random seed for reproducibility (alternative to random_state parameter)
np.random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    # Ensure dim is at least 1
    dim = max(1, dim)
    
    # UMAP requires at least 3 samples (n_neighbors > 1 and n_neighbors < len(embeddings))
    if len(embeddings) <= 2:
        # If too few samples, return original embeddings or pad to minimum dimension
        # Ensure we return at least 1 dimension
        actual_dim = min(dim, embeddings.shape[1])
        if actual_dim <= 0:
            actual_dim = 1
        if embeddings.shape[1] <= actual_dim:
            return embeddings
        # If original dimension is larger, use PCA or just return first dim dimensions
        return embeddings[:, :actual_dim]
    
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    # UMAP requires n_neighbors > 1 and n_neighbors < len(embeddings)
    n_neighbors = max(2, min(n_neighbors, len(embeddings) - 1))
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    # Ensure dim is at least 1
    dim = max(1, dim)
    
    # UMAP requires at least 3 samples (n_neighbors > 1 and n_neighbors < len(embeddings))
    if len(embeddings) <= 2:
        # If too few samples, return original embeddings or pad to minimum dimension
        # Ensure we return at least 1 dimension
        actual_dim = min(dim, embeddings.shape[1])
        if actual_dim <= 0:
            actual_dim = 1
        if embeddings.shape[1] <= actual_dim:
            return embeddings
        # If original dimension is larger, use PCA or just return first dim dimensions
        return embeddings[:, :actual_dim]
    
    # UMAP requires n_neighbors > 1 and n_neighbors < len(embeddings)
    num_neighbors = max(2, min(num_neighbors, len(embeddings) - 1))
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    # Ensure embeddings are float64 for better numerical stability
    embeddings = embeddings.astype(np.float64)
    
    # Handle edge cases
    if len(embeddings) <= 1:
        return 1
    
    # Note: random_state parameter is kept for API compatibility but not used
    # to avoid RecursionError in sklearn parameter validation
    # Randomness is controlled via np.random.seed() at module level
    
    max_clusters = min(max_clusters, len(embeddings))
    if max_clusters < 2:
        return max_clusters
    
    n_clusters = np.arange(1, max_clusters)
    bics = []
    
    for n in n_clusters:
        # Start with a small reg_covar and increase if needed for each iteration
        reg_covar = 1e-6
        success = False
        
        # Avoid using random_state parameter to prevent RecursionError in sklearn
        # Use numpy random seed instead for reproducibility
        try:
            gm = GaussianMixture(
                n_components=int(n), 
                reg_covar=reg_covar,
                covariance_type='full'
            )
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
            success = True
        except (ValueError, np.linalg.LinAlgError) as e:
            # If fitting fails, try with larger reg_covar
            try:
                reg_covar = 1e-4
                gm = GaussianMixture(
                    n_components=int(n), 
                    reg_covar=reg_covar,
                    covariance_type='full'
                )
                gm.fit(embeddings)
                bics.append(gm.bic(embeddings))
                success = True
            except (ValueError, np.linalg.LinAlgError):
                # If still fails, skip this cluster number
                logging.warning(f"Failed to fit GMM with {n} components, skipping")
                continue
    
    if len(bics) == 0:
        # If all failed, return a conservative default
        logging.warning("All GMM fits failed, using default of 2 clusters")
        return 2
    
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    # Ensure embeddings are float64 for better numerical stability
    embeddings = embeddings.astype(np.float64)
    
    # Handle edge cases
    if len(embeddings) <= 1:
        return [np.array([0])], 1
    
    # Note: random_state parameter is kept for API compatibility but not used
    # to avoid RecursionError in sklearn parameter validation
    # Randomness is controlled via np.random.seed() at module level
    
    n_clusters = get_optimal_clusters(embeddings)
    
    # Ensure n_clusters is valid
    if n_clusters < 1:
        n_clusters = 1
    if n_clusters > len(embeddings):
        n_clusters = len(embeddings)
    
    # Start with a small reg_covar and increase if needed
    # Avoid using random_state parameter to prevent RecursionError in sklearn
    # Use numpy random seed instead for reproducibility
    reg_covar = 1e-6
    
    try:
        gm = GaussianMixture(
            n_components=int(n_clusters), 
            reg_covar=reg_covar,
            covariance_type='full'
        )
        gm.fit(embeddings)
    except (ValueError, np.linalg.LinAlgError):
        # If fitting fails, try with larger reg_covar
        reg_covar = 1e-4
        try:
            gm = GaussianMixture(
                n_components=int(n_clusters), 
                reg_covar=reg_covar,
                covariance_type='full'
            )
            gm.fit(embeddings)
        except (ValueError, np.linalg.LinAlgError):
            # If still fails, try with even larger reg_covar
            reg_covar = 1e-2
            gm = GaussianMixture(
                n_components=int(n_clusters), 
                reg_covar=reg_covar,
                covariance_type='full'
            )
            gm.fit(embeddings)
    
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    # Ensure dim is at least 1 to avoid empty arrays
    target_dim = max(1, min(dim, len(embeddings) - 2, embeddings.shape[1]))
    reduced_embeddings_global = global_cluster_embeddings(embeddings, target_dim)
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        # embedding_model: BaseEmbeddingModel,
        max_length_in_cluster: int = 3500,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embedding for node in nodes])

        # Perform the clustering
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate total token length using combined name+desc
            total_length = 0
            for node in cluster_nodes:
                node_text = (node.name or "") + (": " if node.name and node.desc else "") + (node.desc or "")
                # Disable special token check to handle text containing special tokens (e.g., <|endoftext|>)
                total_length += len(tokenizer.encode(node_text, disallowed_special=()))

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, max_length_in_cluster=max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
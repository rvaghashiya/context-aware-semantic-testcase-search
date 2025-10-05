"""
Clustering algorithms for test case grouping.
"""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

class ClusteringEngine:
    """Clustering engine for test case semantic grouping."""

    def __init__(self):
        self.clusterer = None
        self.cluster_labels = None
        self.cluster_centers = None

    def apply_kmeans(self, embeddings: np.ndarray, n_clusters: int = 10, 
                    random_state: int = 42) -> np.ndarray:
        """Apply K-means clustering."""
        print(f"Applying K-means clustering with {n_clusters} clusters")

        self.clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_labels = self.clusterer.fit_predict(embeddings)
        self.cluster_centers = self.clusterer.cluster_centers_

        print(f"K-means completed. Found {len(np.unique(self.cluster_labels))} clusters")
        return self.cluster_labels

    def apply_hierarchical(self, embeddings: np.ndarray, n_clusters: int = 10,
                          linkage: str = 'ward') -> np.ndarray:
        """Apply hierarchical clustering."""
        print(f"Applying hierarchical clustering with {n_clusters} clusters")

        self.clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.cluster_labels = self.clusterer.fit_predict(embeddings)

        print(f"Hierarchical clustering completed")
        return self.cluster_labels

    def apply_dbscan(self, embeddings: np.ndarray, eps: float = 0.5,
                    min_samples: int = 5) -> np.ndarray:
        """Apply DBSCAN clustering."""
        print(f"Applying DBSCAN with eps={eps}, min_samples={min_samples}")

        self.clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        self.cluster_labels = self.clusterer.fit_predict(embeddings)

        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)

        print(f"DBSCAN completed. Found {n_clusters} clusters and {n_noise} noise points")
        return self.cluster_labels

    def cluster_embeddings(self, embeddings: np.ndarray, method: str = 'kmeans', **kwargs) -> np.ndarray:
        """Apply clustering with specified method."""
        method = method.lower()

        if method == 'kmeans':
            return self.apply_kmeans(embeddings, **kwargs)
        elif method == 'hierarchical':
            return self.apply_hierarchical(embeddings, **kwargs)
        elif method == 'dbscan':
            return self.apply_dbscan(embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def get_cluster_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Get statistics about clustering results."""
        if self.cluster_labels is None:
            raise ValueError("No clustering results available")

        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        stats = {
            'n_clusters': n_clusters,
            'n_samples': len(self.cluster_labels),
            'cluster_sizes': {}
        }

        # Compute cluster sizes
        for label in unique_labels:
            if label != -1:
                size = np.sum(self.cluster_labels == label)
                stats['cluster_sizes'][f'cluster_{label}'] = size

        # Compute clustering metrics if valid
        if n_clusters > 1:
            try:
                stats['silhouette_score'] = silhouette_score(embeddings, self.cluster_labels)
                stats['davies_bouldin_score'] = davies_bouldin_score(embeddings, self.cluster_labels)
            except Exception as e:
                print(f"Could not compute clustering metrics: {e}")

        return stats

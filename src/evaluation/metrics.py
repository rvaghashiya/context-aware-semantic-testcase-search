"""
Evaluation metrics for clustering and semantic search.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

class ClusteringMetrics:
    """Clustering evaluation metrics."""

    @staticmethod
    def silhouette_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score."""
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(silhouette_score(embeddings, labels))

    @staticmethod
    def davies_bouldin_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Davies-Bouldin score."""
        if len(np.unique(labels)) < 2:
            return float('inf')
        return float(davies_bouldin_score(embeddings, labels))

    @staticmethod
    def calinski_harabasz_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Calinski-Harabasz score."""
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(calinski_harabasz_score(embeddings, labels))

    @classmethod
    def compute_all_metrics(cls, embeddings: np.ndarray, labels: np.ndarray,
                           true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute all clustering metrics."""
        metrics = {}

        try:
            metrics['silhouette_score'] = cls.silhouette_score(embeddings, labels)
        except Exception:
            metrics['silhouette_score'] = 0.0

        try:
            metrics['davies_bouldin_score'] = cls.davies_bouldin_score(embeddings, labels)
        except Exception:
            metrics['davies_bouldin_score'] = float('inf')

        try:
            metrics['calinski_harabasz_score'] = cls.calinski_harabasz_score(embeddings, labels)
        except Exception:
            metrics['calinski_harabasz_score'] = 0.0

        if true_labels is not None:
            try:
                metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, labels)
                metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(true_labels, labels)
            except Exception:
                pass

        return metrics

class SearchMetrics:
    """Semantic search evaluation metrics."""

    @staticmethod
    def precision_at_k(relevant_items: List[int], retrieved_items: List[int], k: int) -> float:
        """Calculate Precision@K."""
        if k <= 0:
            return 0.0

        top_k = retrieved_items[:k]
        relevant_set = set(relevant_items)

        num_relevant_retrieved = len([item for item in top_k if item in relevant_set])
        return num_relevant_retrieved / k

    @staticmethod
    def recall_at_k(relevant_items: List[int], retrieved_items: List[int], k: int) -> float:
        """Calculate Recall@K."""
        if k <= 0 or len(relevant_items) == 0:
            return 0.0

        top_k = retrieved_items[:k]
        relevant_set = set(relevant_items)

        num_relevant_retrieved = len([item for item in top_k if item in relevant_set])
        return num_relevant_retrieved / len(relevant_items)

    @staticmethod
    def mean_reciprocal_rank(relevant_items_list: List[List[int]], 
                            retrieved_items_list: List[List[int]]) -> float:
        """Calculate Mean Reciprocal Rank (MRR)."""
        if len(relevant_items_list) != len(retrieved_items_list):
            raise ValueError("Number of queries must match")

        reciprocal_ranks = []

        for relevant_items, retrieved_items in zip(relevant_items_list, retrieved_items_list):
            relevant_set = set(relevant_items)

            for rank, item in enumerate(retrieved_items, 1):
                if item in relevant_set:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

class EvaluationSuite:
    """Complete evaluation suite for clustering and search."""

    def __init__(self):
        self.clustering_metrics = ClusteringMetrics()
        self.search_metrics = SearchMetrics()

    def evaluate_clustering(self, embeddings: np.ndarray, predicted_labels: np.ndarray,
                          true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Evaluate clustering results."""
        results = {
            'n_samples': len(predicted_labels),
            'n_clusters': len(np.unique(predicted_labels)),
            'metrics': self.clustering_metrics.compute_all_metrics(
                embeddings, predicted_labels, true_labels
            )
        }

        # Add cluster size statistics
        unique_labels, counts = np.unique(predicted_labels, return_counts=True)
        results['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        results['cluster_size_stats'] = {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts))
        }

        return results

def create_clustering_report(embeddings: np.ndarray, labels: np.ndarray,
                           text_ids: List[str], method_name: str = "clustering") -> str:
    """Create a formatted clustering evaluation report."""
    evaluator = EvaluationSuite()
    results = evaluator.evaluate_clustering(embeddings, labels)

    report = f"\n=== {method_name.upper()} CLUSTERING REPORT ===\n"
    report += f"Number of samples: {results['n_samples']}\n"
    report += f"Number of clusters: {results['n_clusters']}\n"

    report += "\n--- Quality Metrics ---\n"
    for metric, value in results['metrics'].items():
        if isinstance(value, float) and not np.isinf(value):
            report += f"{metric.replace('_', ' ').title()}: {value:.4f}\n"

    report += "\n--- Cluster Size Statistics ---\n"
    stats = results['cluster_size_stats']
    report += f"Mean cluster size: {stats['mean']:.1f}\n"
    report += f"Std deviation: {stats['std']:.1f}\n"
    report += f"Smallest cluster: {stats['min']} samples\n"
    report += f"Largest cluster: {stats['max']} samples\n"

    return report

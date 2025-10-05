"""
Static visualization using matplotlib and seaborn.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

def plot_cluster_scatter(coordinates: np.ndarray, labels: np.ndarray,
                        title: str = "Test Case Clusters",
                        figsize: Tuple[int, int] = (10, 8),
                        save_path: Optional[str] = None):
    """Create static scatter plot of clusters."""
    if not MPL_AVAILABLE:
        print("Matplotlib not available")
        return

    plt.figure(figsize=figsize)

    if coordinates.shape[1] >= 2:
        scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7, s=50)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')

    plt.title(title)
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

def plot_cluster_sizes(labels: np.ndarray, title: str = "Cluster Sizes",
                      figsize: Tuple[int, int] = (10, 6),
                      save_path: Optional[str] = None):
    """Plot cluster size distribution."""
    if not MPL_AVAILABLE:
        return

    unique_labels, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(counts)), counts, color='skyblue', alpha=0.7)

    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Test Cases')
    plt.title(title)
    plt.xticks(range(len(unique_labels)), unique_labels)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_embedding_metrics(metrics_dict: dict, title: str = "Model Comparison",
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None):
    """Plot comparison of different embedding models."""
    if not MPL_AVAILABLE:
        return

    models = list(metrics_dict.keys())
    metric_names = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for i, metric in enumerate(metric_names):
        values = []
        for model in models:
            if metric in metrics_dict[model].get('metrics', {}):
                val = metrics_dict[model]['metrics'][metric]
                if not np.isinf(val):
                    values.append(val)
                else:
                    values.append(0)
            else:
                values.append(0)

        axes[i].bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel('Score')

        # Add value labels
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values) * 0.01, f'{v:.3f}',
                        ha='center', va='bottom')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_similarity_distribution(similarity_matrix: np.ndarray,
                                title: str = "Similarity Distribution",
                                figsize: Tuple[int, int] = (10, 6),
                                save_path: Optional[str] = None):
    """Plot distribution of similarity scores."""
    if not MPL_AVAILABLE:
        return

    # Get upper triangle (excluding diagonal)
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
    similarities = similarity_matrix[mask]

    plt.figure(figsize=figsize)
    plt.hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(similarities), color='red', linestyle='--', 
               label=f'Mean: {np.mean(similarities):.3f}')

    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def create_comparison_plot(results_dict: dict, metric: str = 'silhouette_score',
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """Create comparison plot for different models/methods."""
    if not MPL_AVAILABLE:
        return

    if title is None:
        title = f"Comparison: {metric.replace('_', ' ').title()}"

    models = list(results_dict.keys())
    values = [results_dict[model].get('metrics', {}).get(metric, 0) for model in models]

    plt.figure(figsize=figsize)
    bars = plt.bar(models, values, color='lightblue', alpha=0.8)

    plt.title(title)
    plt.ylabel(metric.replace('_', ' ').title())
    plt.xticks(rotation=45)

    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

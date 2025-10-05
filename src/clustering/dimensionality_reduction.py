"""
Dimensionality reduction utilities for visualization.
"""
import numpy as np
from typing import Dict, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

class DimensionalityReducer:
    """Dimensionality reduction for visualization."""

    def __init__(self):
        self.pca_model = None
        self.tsne_model = None
        self.umap_model = None

    def apply_pca(self, embeddings: np.ndarray, n_components: int = 50, 
                  random_state: int = 42) -> np.ndarray:
        """Apply PCA for initial dimensionality reduction."""
        print(f"Applying PCA: {embeddings.shape} -> {n_components} components")

        self.pca_model = PCA(n_components=n_components, random_state=random_state)
        reduced = self.pca_model.fit_transform(embeddings)

        explained_var = self.pca_model.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.3f}")

        return reduced

    def apply_tsne(self, embeddings: np.ndarray, n_components: int = 2,
                   perplexity: float = 30.0, n_iter: int = 1000,
                   random_state: int = 42, use_pca_first: bool = True) -> np.ndarray:
        """Apply t-SNE for visualization."""
        print(f"Applying t-SNE with perplexity={perplexity}")

        # Apply PCA first if high-dimensional
        if use_pca_first and embeddings.shape[1] > 50:
            embeddings = self.apply_pca(embeddings, n_components=50)

        self.tsne_model = TSNE(n_components=n_components, perplexity=perplexity,
                              n_iter=n_iter, random_state=random_state)

        reduced = self.tsne_model.fit_transform(embeddings)
        print(f"t-SNE completed: {reduced.shape}")
        return reduced

    def apply_umap(self, embeddings: np.ndarray, n_components: int = 2,
                   n_neighbors: int = 15, min_dist: float = 0.1,
                   random_state: int = 42) -> np.ndarray:
        """Apply UMAP for visualization."""
        if not UMAP_AVAILABLE:
            print("UMAP not available, using t-SNE instead")
            return self.apply_tsne(embeddings, n_components=n_components)

        print(f"Applying UMAP with n_neighbors={n_neighbors}")

        self.umap_model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                   min_dist=min_dist, random_state=random_state)

        reduced = self.umap_model.fit_transform(embeddings)
        print(f"UMAP completed: {reduced.shape}")
        return reduced

    def reduce_embeddings(self, embeddings: np.ndarray, method: str = 'tsne',
                         n_components: int = 2, **kwargs) -> np.ndarray:
        """Apply dimensionality reduction with specified method."""
        method = method.lower()

        if method == 'pca':
            return self.apply_pca(embeddings, n_components, **kwargs)
        elif method == 'tsne':
            return self.apply_tsne(embeddings, n_components, **kwargs)
        elif method == 'umap':
            return self.apply_umap(embeddings, n_components, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

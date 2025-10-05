"""
Semantic similarity and search engine for test cases.
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    """Semantic similarity and search engine."""

    def __init__(self):
        self.embeddings = None
        self.text_ids = None
        self.texts = None
        self.similarity_matrix = None

    def load_embeddings(self, embeddings: np.ndarray, text_ids: List[str], texts: List[str]):
        """Load embeddings and associated data."""
        self.embeddings = embeddings
        self.text_ids = text_ids
        self.texts = texts
        print(f"Loaded embeddings: {embeddings.shape}")

    def compute_similarity_matrix(self) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        print("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.embeddings)
        return self.similarity_matrix

    def find_similar_texts(self, query_index: int, top_k: int = 10) -> List[Tuple[int, str, float]]:
        """Find most similar texts to a given text by index."""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        similarities = self.similarity_matrix[query_index]

        # Exclude self-similarity
        similarities_copy = similarities.copy()
        similarities_copy[query_index] = -1
        top_indices = np.argsort(similarities_copy)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] == -1:
                continue
            results.append((int(idx), self.text_ids[idx], float(similarities[idx])))

        return results

    def search_by_text(self, query_text: str, embedder, top_k: int = 10) -> List[Tuple[int, str, float]]:
        """Search for similar texts using a query string."""
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        # Embed the query text
        query_embedding = embedder.embed_single_text(query_text)
        query_embedding = query_embedding.reshape(1, -1)

        # Compute similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((int(idx), self.text_ids[idx], float(similarities[idx])))

        return results

    def get_similarity_statistics(self) -> Dict[str, float]:
        """Get statistics about similarity distribution."""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # Exclude diagonal
        mask = np.eye(self.similarity_matrix.shape[0], dtype=bool)
        similarities = self.similarity_matrix[~mask]

        return {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities))
        }

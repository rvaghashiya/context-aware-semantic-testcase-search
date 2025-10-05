"""
BERT embeddings implementation.
"""
import numpy as np
from typing import List, Dict, Any
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from src.models.base_embedder import BaseEmbedder

# using all-MiniLM-L6-v2 instead of BERT for CPU-inference
class BERTEmbedder(BaseEmbedder):
    """BERT embeddings using Sentence Transformers."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name_or_path = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.model = None

    def initialize(self) -> None:
        """Initialize BERT model."""
        if self.is_initialized:
            return

        if not ST_AVAILABLE:
            print("Sentence Transformers not available. Using random embeddings for demo.")
            self.model = None
            self.embedding_dim = 384
            self.is_initialized = True
            return

        try:
            self.model = SentenceTransformer(self.model_name_or_path)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.is_initialized = True
            print(f"BERT model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Could not load BERT model: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = 384
            self.is_initialized = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings."""
        if not self.is_initialized:
            self.initialize()

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        if self.model is None:
            # Return random embeddings for demo
            np.random.seed(43)
            return np.random.randn(len(texts), self.embedding_dim)

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error generating BERT embeddings: {e}")
            np.random.seed(43)
            return np.random.randn(len(texts), self.embedding_dim)

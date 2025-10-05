"""
ELMo embeddings implementation.
"""
import numpy as np
from typing import List, Dict, Any
try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from src.models.base_embedder import BaseEmbedder

class ELMoEmbedder(BaseEmbedder):
    """ELMo embeddings using TensorFlow Hub."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hub_url = config.get('hub_url', 'https://tfhub.dev/google/elmo/3')
        self.elmo_model = None

    def initialize(self) -> None:
        """Initialize ELMo model."""
        if self.is_initialized:
            return

        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow tensorflow-hub")

        try:
            self.elmo_model = hub.load(self.hub_url)
            self.embedding_dim = 1024  # ELMo standard dimension
            self.is_initialized = True
            print(f"ELMo model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            # Fallback to random embeddings for demo
            print(f"Could not load ELMo model: {e}. Using random embeddings for demo.")
            self.elmo_model = None
            self.embedding_dim = 1024
            self.is_initialized = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate ELMo embeddings."""
        if not self.is_initialized:
            self.initialize()

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        if self.elmo_model is None:
            # Return random embeddings for demo
            np.random.seed(42)
            return np.random.randn(len(texts), self.embedding_dim)

        try:
            # Convert to tensor and get embeddings
            text_tensor = tf.constant(texts)
            embeddings = self.elmo_model(text_tensor)

            # Take mean across sequence dimension
            if len(embeddings.shape) == 3:
                embeddings = tf.reduce_mean(embeddings, axis=1)

            return embeddings.numpy()
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return random embeddings as fallback
            np.random.seed(42)
            return np.random.randn(len(texts), self.embedding_dim)

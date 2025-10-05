"""
Abstract base class for text embedding models.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any

class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('name', 'unknown')
        self.embedding_dim = config.get('embedding_dim', 512)
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the embedding model."""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass

    def embed_single_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        embeddings = self.embed_texts([text])
        return embeddings[0]

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess texts before embedding."""
        return [' '.join(text.split()) for text in texts]

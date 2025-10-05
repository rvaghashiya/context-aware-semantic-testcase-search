"""
T5 embeddings implementation.
"""
import numpy as np
from typing import List, Dict, Any
try:
    from transformers import T5EncoderModel, T5Tokenizer
    import torch
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False

from src.models.base_embedder import BaseEmbedder

class T5Embedder(BaseEmbedder):
    """T5 embeddings using Transformers library."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 't5-small')
        self.model = None
        self.tokenizer = None

    def initialize(self) -> None:
        """Initialize T5 model."""
        if self.is_initialized:
            return

        if not T5_AVAILABLE:
            print("Transformers/PyTorch not available. Using random embeddings for demo.")
            self.model = None
            self.embedding_dim = 512
            self.is_initialized = True
            return

        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            self.model.eval()
            self.embedding_dim = self.model.config.d_model
            self.is_initialized = True
            print(f"T5 model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            print(f"Could not load T5 model: {e}. Using random embeddings.")
            self.model = None
            self.embedding_dim = 512
            self.is_initialized = True

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate T5 embeddings."""
        if not self.is_initialized:
            self.initialize()

        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)

        if self.model is None:
            # Return random embeddings for demo
            np.random.seed(44)
            return np.random.randn(len(texts), self.embedding_dim)

        try:
            embeddings = []
            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                    outputs = self.model(**inputs)
                    # Mean pooling
                    pooled = outputs.last_hidden_state.mean(dim=1)
                    embeddings.append(pooled.numpy())

            return np.vstack(embeddings)
        except Exception as e:
            print(f"Error generating T5 embeddings: {e}")
            np.random.seed(44)
            return np.random.randn(len(texts), self.embedding_dim)

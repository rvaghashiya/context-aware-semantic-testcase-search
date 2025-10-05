"""
Factory for creating embedding models.
"""
from typing import Dict, Any
from src.models.base_embedder import BaseEmbedder
from src.models.elmo_embedder import ELMoEmbedder
from src.models.bert_embedder import BERTEmbedder
from src.models.t5_embedder import T5Embedder

class ModelFactory:
    """Factory for creating embedding models."""

    SUPPORTED_MODELS = {
        'elmo': ELMoEmbedder,
        'bert': BERTEmbedder,
        't5': T5Embedder
    }

    @classmethod
    def create_embedder(cls, model_name: str, config: Dict[str, Any]) -> BaseEmbedder:
        """Create embedder instance."""
        model_name = model_name.lower()

        if model_name not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")

        embedder_class = cls.SUPPORTED_MODELS[model_name]
        return embedder_class(config)

    @classmethod
    def get_supported_models(cls):
        """Get supported model names."""
        return list(cls.SUPPORTED_MODELS.keys())

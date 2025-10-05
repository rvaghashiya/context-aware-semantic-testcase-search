"""
Tests for embedding models.
"""
import pytest
import numpy as np
from src.models.model_factory import ModelFactory

class TestModelFactory:
    """Test model factory."""

    def test_supported_models(self):
        """Test getting supported models."""
        models = ModelFactory.get_supported_models()

        assert isinstance(models, list)
        assert 'bert' in models
        assert 'elmo' in models
        assert 't5' in models

    def test_create_bert_embedder(self):
        """Test BERT embedder creation."""
        config = {'name': 'bert', 'embedding_dim': 384}
        embedder = ModelFactory.create_embedder('bert', config)

        assert embedder is not None
        assert embedder.model_name == 'bert'

    def test_invalid_model(self):
        """Test invalid model handling."""
        with pytest.raises(ValueError):
            ModelFactory.create_embedder('invalid_model', {})

class TestEmbedders:
    """Test embedding generation."""

    def test_bert_embedder(self):
        """Test BERT embedder."""
        config = {'name': 'bert', 'embedding_dim': 384}
        embedder = ModelFactory.create_embedder('bert', config)
        embedder.initialize()

        texts = ["test user login", "verify payment process"]
        embeddings = embedder.embed_texts(texts)

        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] > 0  # Some embedding dimension
        assert not np.isnan(embeddings).any()

    def test_empty_texts(self):
        """Test empty text list handling."""
        config = {'name': 'bert', 'embedding_dim': 384}
        embedder = ModelFactory.create_embedder('bert', config)
        embedder.initialize()

        embeddings = embedder.embed_texts([])
        assert embeddings.shape == (0, embedder.embedding_dim)

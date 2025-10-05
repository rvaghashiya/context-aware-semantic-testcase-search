"""
Tests for data loading and preprocessing.
"""
import pytest
import pandas as pd
from src.data.dataloader import SyntheticTestCaseGenerator, load_demo_data
from src.data.preprocessor import TextPreprocessor, create_default_preprocessor

class TestSyntheticTestCaseGenerator:
    """Test synthetic data generation."""

    def test_generate_sample_data(self):
        """Test sample data generation."""
        generator = SyntheticTestCaseGenerator()
        df = generator.generate_sample_data(n_samples=100)

        assert len(df) == 100
        assert 'test_case_id' in df.columns
        assert 'test_description' in df.columns
        assert all(isinstance(text, str) for text in df['test_description'])

    def test_load_demo_data(self):
        """Test demo data loading."""
        df = load_demo_data(max_samples=50)

        assert len(df) <= 50
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

class TestTextPreprocessor:
    """Test text preprocessing."""

    def test_basic_preprocessing(self):
        """Test basic text cleaning."""
        preprocessor = TextPreprocessor()

        text = "Test_User_Login_Success!@#"
        cleaned = preprocessor.clean_text(text)

        assert cleaned == "test user login success"

    def test_empty_text(self):
        """Test empty text handling."""
        preprocessor = TextPreprocessor()

        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text(None) == ""

    def test_preprocess_texts_list(self):
        """Test batch text preprocessing."""
        preprocessor = create_default_preprocessor()

        texts = ["Test_One", "Test_Two", "Test_Three"]
        processed = preprocessor.preprocess_texts(texts)

        assert len(processed) == len(texts)
        assert all(isinstance(text, str) for text in processed)

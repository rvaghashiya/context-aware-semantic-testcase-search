"""
Text preprocessing utilities.
"""
import re
from typing import List

class TextPreprocessor:
    """Text preprocessor for test cases."""

    def __init__(self, lowercase: bool = True, remove_punctuation: bool = True):
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # Remove special characters
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        return text.strip()

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts."""
        return [self.clean_text(text) for text in texts]

def create_default_preprocessor() -> TextPreprocessor:
    """Create default preprocessor."""
    return TextPreprocessor(lowercase=True, remove_punctuation=True)

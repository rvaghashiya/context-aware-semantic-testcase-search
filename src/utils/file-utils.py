"""
File I/O utilities for the semantic clustering project.
"""
import os
import json
import pickle
import numpy as np
from typing import Any, Dict, List
from pathlib import Path

def ensure_dir(filepath: str) -> None:
    """Ensure directory exists for the given filepath."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save JSON file
    """
    ensure_dir(filepath)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str) -> Any:
    """Load data from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(obj: Any, filepath: str) -> None:
    """Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save pickle file
    """
    ensure_dir(filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_embeddings(embeddings: np.ndarray, filepath: str) -> None:
    """Save embeddings to numpy file.
    
    Args:
        embeddings: Numpy array of embeddings
        filepath: Path to save embeddings
    """
    ensure_dir(filepath)
    np.save(filepath, embeddings)

def load_embeddings(filepath: str) -> np.ndarray:
    """Load embeddings from numpy file.
    
    Args:
        filepath: Path to embeddings file
        
    Returns:
        Loaded embeddings array
    """
    return np.load(filepath)

def save_text(text: str, filepath: str) -> None:
    """Save text to file.
    
    Args:
        text: Text content to save
        filepath: Path to save text file
    """
    ensure_dir(filepath)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

def load_text(filepath: str) -> str:
    """Load text from file.
    
    Args:
        filepath: Path to text file
        
    Returns:
        Text content
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def list_files(directory: str, extension: str = None) -> List[str]:
    """List files in directory with optional extension filter.
    
    Args:
        directory: Directory to search
        extension: File extension filter (e.g., '.json')
        
    Returns:
        List of file paths
    """
    path = Path(directory)
    
    if not path.exists():
        return []
    
    if extension:
        pattern = f"*{extension}"
        files = list(path.glob(pattern))
    else:
        files = [f for f in path.iterdir() if f.is_file()]
    
    return [str(f) for f in files]

def get_file_size(filepath: str) -> int:
    """Get file size in bytes.
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
    """
    return os.path.getsize(filepath)

def file_exists(filepath: str) -> bool:
    """Check if file exists.
    
    Args:
        filepath: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    return os.path.exists(filepath)
"""
Configuration management utilities for the semantic clustering project.
"""
import yaml
import os
from typing import Dict, Any, Optional

class Config:
    """Configuration loader and manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = {}
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as file:
                self.config = yaml.safe_load(file) or {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.
        
        Args:
            key: Configuration key (supports dot notation like 'model.name')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('model', {})
    
    def get_clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration section."""
        return self.config.get('clustering', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration section."""
        return self.config.get('visualization', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration section."""
        return self.config.get('evaluation', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration section."""
        return self.config.get('data', {})
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self.config.update(updates)
    
    def save(self, filepath: str) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False, indent=2)

def load_model_config(model_name: str) -> Config:
    """Load configuration for specific model.
    
    Args:
        model_name: Name of the model ('elmo', 'bert', 't5')
        
    Returns:
        Config object loaded with model configuration
    """
    config_path = f"configs/{model_name}_config.yaml"
    return Config(config_path)
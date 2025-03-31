"""
Configuration handling utilities for the PINNs package.
Loads and merges configuration files, makes them accessible throughout the application.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Config:
    """Configuration manager for PINNs models."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_dir = Path(config_path).parent if isinstance(config_path, str) else config_path.parent
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file, resolving inheritance if needed.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing the configuration
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Handle inheritance
        if 'inherit' in config:
            parent_path = self.config_dir / config['inherit']
            with open(parent_path, 'r') as f:
                parent_config = yaml.safe_load(f)
            
            # Merge configurations (parent is overwritten by child)
            merged_config = self._deep_merge(parent_config, config)
            del merged_config['inherit']  # Remove the inherit key
            return merged_config
        
        return config
    
    def _deep_merge(self, parent: Dict[str, Any], child: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries, with child values taking precedence.
        
        Args:
            parent: Parent dictionary
            child: Child dictionary that overwrites parent values
            
        Returns:
            Merged dictionary
        """
        merged = parent.copy()
        
        for key, value in child.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
                
        return merged
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value by key, with optional default.
        Supports dot notation for nested dictionaries (e.g., 'model.layers').
        
        Args:
            key: Configuration key
            default: Default value if key is not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config values."""
        return self.get(key)
    
    def set_derived_paths(self) -> None:
        """
        Set derived paths based on the configuration.
        Creates output directories if they don't exist.
        """
        # Create output directories if they don't exist
        output_dirs = [
            self.get('output.save_dir', 'outputs'),
            self.get('output.model_dir', 'outputs/models'),
            self.get('output.figure_dir', 'outputs/figures'),
            self.get('output.metrics_dir', 'outputs/metrics'),
            self.get('logging.save_dir', 'outputs/logs')
        ]
        
        for directory in output_dirs:
            os.makedirs(directory, exist_ok=True)


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load a configuration file and return a Config object.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object
    """
    config = Config(config_path)
    config.set_derived_paths()
    return config
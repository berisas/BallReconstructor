"""
Configuration management system for research experiments.
Loads, validates, and provides access to configs.
"""
import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and manage experiment configurations."""
    
    def __init__(self, config_path: str = "configs/default_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        return self.config.get(section, {})
    
    def override(self, overrides: Dict[str, Any]):
        """Override config values."""
        for key, value in overrides.items():
            keys = key.split('.')
            current = self.config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
    
    def create_from_template(self, name: str, output_path: str):
        """Create a new config file from current config."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Config saved to {output_path}")
    
    def print_config(self):
        """Print configuration in readable format."""
        print("\n" + "="*80)
        print("CONFIGURATION")
        print("="*80)
        print(yaml.dump(self.config, default_flow_style=False))
        print("="*80 + "\n")
    
    def __str__(self) -> str:
        return yaml.dump(self.config, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary."""
        return self.config.copy()


class ConfigBuilder:
    """Build configurations programmatically."""
    
    def __init__(self):
        self.config = {}
    
    def set_experiment(self, name: str, method: str, description: str = "") -> 'ConfigBuilder':
        """Set experiment settings."""
        self.config['experiment'] = {
            'name': name,
            'method': method,
            'description': description,
            'timestamp': True
        }
        return self
    
    def set_dataset(self, mesh_file: str) -> 'ConfigBuilder':
        """Set dataset."""
        self.config['dataset'] = {
            'mesh_file': mesh_file,
            'normalize': True,
            'center': True
        }
        return self
    
    def set_training(self, epochs: int = 150, lr: float = 0.001) -> 'ConfigBuilder':
        """Set training parameters."""
        self.config['training'] = {
            'epochs': epochs,
            'batch_size': 1,
            'learning_rate': lr,
            'lr_decay_rate': 0.95,
            'lr_decay_steps': 100,
            'gradient_clip': 1.0
        }
        return self
    
    def set_model(self, hidden_dim: int = 512) -> 'ConfigBuilder':
        """Set model parameters."""
        self.config['model'] = {
            'architecture': 'progressive',
            'hidden_dim': hidden_dim,
            'dropout_rate': 0.1,
            'feature_scale_init': 1.0,
            'displacement_scale_init': 0.05
        }
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return config."""
        return self.config
    
    def to_yaml(self, path: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        print(f"Config saved to {path}")

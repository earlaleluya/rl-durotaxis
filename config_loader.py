"""
Configuration Loader for Durotaxis RL Training

This module provides utilities to load and validate configuration from YAML files,
with support for default values and parameter overrides.
"""

import yaml
import os
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and manage configuration from YAML files"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize config loader
        
        Parameters
        ----------
        config_path : str
            Path to the configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Empty or invalid configuration file: {self.config_path}")
        
        return config
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a configuration section
        
        Parameters
        ----------
        section_name : str
            Name of the configuration section
            
        Returns
        -------
        Dict[str, Any]
            Configuration section as dictionary
        """
        if section_name not in self.config:
            raise KeyError(f"Configuration section '{section_name}' not found")
        
        return self.config[section_name].copy()
    
    def get_trainer_config(self) -> Dict[str, Any]:
        """Get trainer configuration with flattened nested sections"""
        trainer_config = self.get_section('trainer')
        
        # Flatten nested configurations
        if 'random_substrate' in trainer_config:
            random_substrate = trainer_config.pop('random_substrate')
            for key, value in random_substrate.items():
                trainer_config[key] = value
        
        if 'component_weights' in trainer_config:
            trainer_config['component_weights'] = trainer_config['component_weights']
            
        if 'policy_loss_weights' in trainer_config:
            trainer_config['policy_loss_weights'] = trainer_config['policy_loss_weights']
            
        if 'adaptive_scaling' in trainer_config:
            adaptive_scaling = trainer_config.pop('adaptive_scaling')
            for key, value in adaptive_scaling.items():
                trainer_config[key] = value
        
        return trainer_config
    
    def get_environment_config(self) -> Dict[str, Any]:
        """Get environment configuration with reward structures"""
        return self.get_section('environment')
    
    def get_encoder_config(self) -> Dict[str, Any]:
        """Get encoder configuration"""
        return self.get_section('encoder')
    
    def get_actor_critic_config(self) -> Dict[str, Any]:
        """Get actor-critic configuration"""
        return self.get_section('actor_critic')
    
    def get_algorithm_config(self) -> Dict[str, Any]:
        """Get algorithm configuration"""
        return self.get_section('algorithm')
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return self.get_section('system')
    
    def override_params(self, section_name: str, **kwargs) -> Dict[str, Any]:
        """
        Get configuration section with parameter overrides
        
        Parameters
        ----------
        section_name : str
            Name of the configuration section
        **kwargs
            Parameters to override
            
        Returns
        -------
        Dict[str, Any]
            Configuration with overrides applied
        """
        config = self.get_section(section_name)
        
        # Apply overrides
        for key, value in kwargs.items():
            if value is not None:  # Only override if value is explicitly provided
                config[key] = value
        
        return config
    
    def merge_configs(self, *section_names: str, **overrides) -> Dict[str, Any]:
        """
        Merge multiple configuration sections with optional overrides
        
        Parameters
        ----------
        *section_names : str
            Names of configuration sections to merge
        **overrides
            Parameter overrides
            
        Returns
        -------
        Dict[str, Any]
            Merged configuration
        """
        merged_config = {}
        
        for section_name in section_names:
            section_config = self.get_section(section_name)
            merged_config.update(section_config)
        
        # Apply overrides
        for key, value in overrides.items():
            if value is not None:
                merged_config[key] = value
        
        return merged_config


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Convenience function to load configuration
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    ConfigLoader
        Loaded configuration object
    """
    return ConfigLoader(config_path)


def get_default_config_path() -> str:
    """Get default configuration file path"""
    # Look for config.yaml in current directory, then in script directory
    current_dir_config = "config.yaml"
    script_dir_config = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    if os.path.exists(current_dir_config):
        return current_dir_config
    elif os.path.exists(script_dir_config):
        return script_dir_config
    else:
        raise FileNotFoundError("No config.yaml found in current or script directory")
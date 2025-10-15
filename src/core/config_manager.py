"""
Centralized configuration management for LeadScore AI system.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Centralized configuration management with environment support.
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (dev, prod, test)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv('LEADSCORE_ENV', 'dev')
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from files."""
        try:
            # Load base configuration
            base_config_path = self.config_dir / "base_config.yaml"
            if base_config_path.exists():
                with open(base_config_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
            
            # Load environment-specific configuration
            env_config_path = self.config_dir / f"{self.environment}_config.yaml"
            if env_config_path.exists():
                with open(env_config_path, 'r') as f:
                    env_config = yaml.safe_load(f) or {}
                    self._deep_merge(self._config, env_config)
            
            # Load model configuration if exists
            model_config_path = self.config_dir / "model_config.yaml"
            if model_config_path.exists():
                with open(model_config_path, 'r') as f:
                    model_config = yaml.safe_load(f) or {}
                    self._config['model'] = model_config
            
            # Override with environment variables
            self._load_env_overrides()
            
            logger.info(f"Configuration loaded for environment: {self.environment}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration: {e}")
            self._config = self._get_default_config()
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables."""
        env_mappings = {
            'LEADSCORE_DATA_PATH': ['data', 'path'],
            'LEADSCORE_MODELS_DIR': ['model', 'models_dir'],
            'LEADSCORE_LOG_LEVEL': ['logging', 'level'],
            'LEADSCORE_API_PORT': ['api', 'port'],
            'LEADSCORE_WEBHOOK_PORT': ['webhook', 'port']
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(self._config, config_path, value)
    
    def _set_nested_value(self, config: Dict, path: list, value: Any):
        """Set a nested configuration value."""
        for key in path[:-1]:
            config = config.setdefault(key, {})
        config[path[-1]] = value
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'data': {
                'path': 'data/raw/',
                'file_name': 'data.xlsx',
                'sheet_name': 1
            },
            'model': {
                'models_dir': 'models',
                'test_size': 0.2,
                'random_state': 42,
                'cv_folds': 5,
                'outlier_multiplier': 1.5
            },
            'feature_engineering': {
                'engagement_weights': {
                    'email_weight': 0.2,
                    'meeting_weight': 0.6,
                    'demo_weight': 0.2
                }
            },
            'api': {
                'host': 'localhost',
                'port': 8000,
                'debug': False
            },
            'webhook': {
                'host': 'localhost',
                'port': 8001
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            'output': {
                'plots_dir': 'outputs/plots',
                'tables_dir': 'outputs/tables',
                'formats': ['png', 'svg']
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'model.test_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data-related configuration."""
        return self.get('data', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-related configuration."""
        return self.get('model', {})
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API-related configuration."""
        return self.get('api', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output generation configuration."""
        return self.get('output', {})
    
    def validate_config(self) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        required_keys = [
            'data.path',
            'model.models_dir',
            'api.port',
            'webhook.port'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Missing required configuration: {key}")
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()
    
    def save_config(self, file_path: str):
        """
        Save current configuration to file.
        
        Args:
            file_path: Path to save configuration
        """
        with open(file_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)


# Global configuration instance
_config_manager = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_config(config_dir: str = "config", environment: str = None) -> ConfigManager:
    """
    Initialize global configuration manager.
    
    Args:
        config_dir: Configuration directory
        environment: Environment name
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    _config_manager = ConfigManager(config_dir, environment)
    return _config_manager

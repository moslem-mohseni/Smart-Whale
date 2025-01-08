# ai/models/__init__.py
"""
AI Models Package
---------------
This package serves as the central hub for all AI models in the system.
It coordinates between different types of models (NLP, Financial, etc.)
and provides a unified interface for model management.

The models package includes:
1. NLP Models: For multilingual text processing
2. Financial Models: For market analysis and prediction
3. Training Infrastructure: For model training and evaluation

Usage:
    from ai.models import ModelManager
    manager = ModelManager()
    model = manager.get_model('sentiment_analysis')
"""

from typing import Dict, Optional, Type
from pathlib import Path


class ModelManager:
    """
    Central manager for all AI models in the system.
    Handles model loading, initialization, and coordination between different model types.
    """

    def __init__(self):
        self._models: Dict[str, object] = {}
        self._model_configs: Dict[str, dict] = {}

    def register_model(self, model_name: str, model_class: Type, config: Optional[dict] = None):
        """Register a new model with the manager"""
        self._model_configs[model_name] = config or {}

    def get_model(self, model_name: str) -> Optional[object]:
        """Retrieve a model by name, loading it if necessary"""
        return self._models.get(model_name)


# Version information
__version__ = '0.1.0'

# Make the ModelManager available at the package level
__all__ = ['ModelManager']
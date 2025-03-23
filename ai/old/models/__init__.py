from typing import Dict, Optional, Type
from pathlib import Path
import logging
from ..core.learning.fast_learner import FastLearner

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self._models: Dict[str, object] = {}
        self._model_configs: Dict[str, dict] = {}
        self.config = {
            'knowledge_base_path': Path('knowledge_base'),
            'min_confidence': 0.7,
            'max_pattern_length': 5,
            'min_frequency': 3
        }

    async def initialize_fast_learning(self):
        try:
            learner = FastLearner(self.config)
            await learner.initialize()
            self._models['fast_learner'] = learner
            return True
        except Exception as e:
            logger.error(f"Failed to initialize fast learner: {str(e)}")
            return False

    def register_model(self, model_name: str, model_class: Type, config: Optional[dict] = None):
        self._model_configs[model_name] = config or {}

    def get_model(self, model_name: str) -> Optional[object]:
        return self._models.get(model_name)

__version__ = '0.1.0'
__all__ = ['ModelManager']
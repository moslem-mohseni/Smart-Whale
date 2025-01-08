"""
Core AI System - پکیج اصلی سیستم هوش مصنوعی
"""
from typing import Dict, Any
import logging
from pathlib import Path

__version__ = '0.1.0'
__author__ = 'AI Team'

# تنظیم لاگر پایه
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# مسیرهای اصلی پروژه
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATA_DIR = PROJECT_ROOT / 'data'
CONFIGS_DIR = PROJECT_ROOT / 'core' / 'common' / 'configs'

# مقادیر پیش‌فرض تنظیمات
DEFAULT_CONFIG: Dict[str, Any] = {
    'model_management': {
        'max_models': 10,
        'default_batch_size': 32,
        'timeout': 30,
    },
    'knowledge': {
        'max_cache_size': 1000,
        'cache_ttl': 3600,
    },
    'learning': {
        'min_confidence': 0.7,
        'max_retries': 3,
    },
    'metrics': {
        'collection_interval': 60,
        'retention_days': 7,
    }
}

# Import language components
from .base.language_processor import LanguageProcessor, ProcessingResult
from .base.language_learner import LanguageLearner, LearningExample
from .persian.processor import PersianProcessor
from .persian.learner import PersianLearner

def get_language_processor(language: str, config: dict = None) -> LanguageProcessor:
    """دریافت پردازشگر زبان"""
    if language == 'fa':
        return PersianProcessor(config)
    raise ValueError(f"Language {language} not supported")

def get_language_learner(language: str, config: dict = None) -> LanguageLearner:
    """دریافت یادگیرنده زبان"""
    if language == 'fa':
        return PersianLearner(config)
    raise ValueError(f"Language {language} not supported")

__all__ = [
    'LanguageProcessor',
    'ProcessingResult',
    'LanguageLearner',
    'LearningExample',
    'get_language_processor',
    'get_language_learner',
    'DEFAULT_CONFIG',
    '__version__',
]
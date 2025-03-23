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

# نیازی به import در این مرحله نیست
# بعداً در فایل‌هایی که نیاز دارند import می‌کنیم
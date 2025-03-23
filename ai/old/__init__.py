"""
ماژول هوش مصنوعی (AI Module)

این ماژول مسئولیت مدیریت مدل‌های هوش مصنوعی سیستم را بر عهده دارد.
طراحی به گونه‌ای است که هر نوع مدل بتواند به صورت مستقل عمل کند و به راحتی
به سیستم اضافه یا از آن حذف شود.
"""

from enum import Enum
import logging
from pathlib import Path

# تنظیم لاگر
logger = logging.getLogger(__name__)

# نسخه ماژول
__version__ = '0.1.0'

# مسیر پایه برای ذخیره مدل‌ها و داده‌ها
BASE_PATH = Path(__file__).parent
MODELS_PATH = BASE_PATH / 'models'
DATA_PATH = BASE_PATH / 'data'
CACHE_PATH = BASE_PATH / 'cache'

# اطمینان از وجود مسیرهای اصلی
for path in [MODELS_PATH, DATA_PATH, CACHE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

class ModelStatus(Enum):
    """وضعیت‌های ممکن برای یک مدل"""
    INITIALIZING = 'initializing'
    READY = 'ready'
    TRAINING = 'training'
    ERROR = 'error'
    DEPRECATED = 'deprecated'

# تنظیمات مربوط به اسکیل‌پذیری و محدودیت‌های کلی سیستم
SYSTEM_CONFIG = {
    'scaling': {
        'min_instances': 1,
        'max_instances': 10,
        'cooldown_period': 300  # ثانیه
    },
    'limits': {
        'max_concurrent_models': 50,
        'max_queue_size': 10000,
        'request_timeout': 30  # ثانیه
    }
}

# ثبت اطلاعات راه‌اندازی
logger.info(f"AI Module v{__version__} initialized")
logger.info(f"Models directory: {MODELS_PATH}")
logger.info(f"Data directory: {DATA_PATH}")
logger.info(f"Cache directory: {CACHE_PATH}")
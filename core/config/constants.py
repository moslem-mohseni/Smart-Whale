# core/config/constants.py
"""
پکیج: core.config.constants
توضیحات: ثابت‌های مورد نیاز برای ماژول پیکربندی
نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

from enum import Enum
from typing import Dict, Any

class Environment(str, Enum):
    """محیط‌های اجرایی برنامه"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

# تنظیمات پیش‌فرض برای هر محیط
DEFAULT_CONFIGS: Dict[Environment, Dict[str, Any]] = {
    Environment.DEVELOPMENT: {
        "debug": True,
        "reload": True,
        "log_level": "DEBUG"
    },
    Environment.TESTING: {
        "debug": True,
        "reload": False,
        "log_level": "DEBUG"
    },
    Environment.STAGING: {
        "debug": False,
        "reload": False,
        "log_level": "INFO"
    },
    Environment.PRODUCTION: {
        "debug": False,
        "reload": False,
        "log_level": "WARNING"
    }
}

# مسیرهای نسبی برای فایل‌های پیکربندی
CONFIG_DIRECTORY = "config"
BASE_CONFIG_FILENAME = "base_config.json"
ENV_CONFIG_FILENAME_TEMPLATE = "{env}_config.json"

# کلیدهای محیطی مهم
ENV_VAR_KEYS = [
    "DATABASE_URL",
    "REDIS_URL",
    "KAFKA_BROKERS",
    "AI_MODEL_PATH",
    "TRAINING_BATCH_SIZE",
    "LEARNING_RATE"
]

# حداکثر زمان نگهداری کش (به ثانیه)
CONFIG_CACHE_TTL = 300  # 5 minutes
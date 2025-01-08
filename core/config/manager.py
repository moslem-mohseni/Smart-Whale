# core/config/manager.py
"""
پکیج: core.config.manager
توضیحات: پیاده‌سازی مدیر پیکربندی سیستم با قابلیت‌های:
- مدیریت محیط‌های مختلف (توسعه، تست، نسخه نهایی)
- بارگذاری تنظیمات از منابع مختلف با اولویت‌بندی
- پشتیبانی از کلیدهای تو در تو
- کش‌گذاری نتایج برای بهبود کارایی
- مدیریت خطا و اعتبارسنجی
- قابلیت بازخوانی در زمان اجرا

نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import os
import json
from functools import lru_cache
import logging
from datetime import datetime

# تنظیم لاگر برای ثبت رویدادها
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    مدیر پیکربندی سیستم با پشتیبانی از محیط‌های مختلف و قابلیت بازخوانی خودکار.
    این کلاس مسئولیت‌های زیر را بر عهده دارد:
    1. بارگذاری تنظیمات از منابع مختلف
    2. اعتبارسنجی و تبدیل داده‌ها
    3. مدیریت دسترسی به تنظیمات
    4. ذخیره‌سازی موقت برای بهبود کارایی
    """

    # محیط‌های معتبر سیستم
    VALID_ENVIRONMENTS = {'development', 'testing', 'staging', 'production'}

    def __init__(self, env: Optional[str] = None):
        """
        راه‌اندازی مدیر پیکربندی.

        Args:
            env: محیط اجرایی (اختیاری). اگر تعیین نشود، از متغیر محیطی APP_ENV استفاده می‌شود.

        Raises:
            ValueError: اگر محیط تعیین شده معتبر نباشد.
        """
        # تنظیم و اعتبارسنجی محیط
        env_name = env or os.getenv('APP_ENV', 'development')
        if env_name not in self.VALID_ENVIRONMENTS:
            raise ValueError(
                f"Invalid environment: {env_name}. "
                f"Valid environments are: {', '.join(self.VALID_ENVIRONMENTS)}"
            )
        self._env = env_name  # تغییر از env به _env

        # تنظیم مسیر پایه و مقداردهی اولیه
        self.base_path = Path(__file__).parent.parent.parent
        self._config: Dict[str, Any] = {}
        self._last_reload: datetime = datetime.now()

        # بارگذاری اولیه تنظیمات
        logger.info(f"Initializing ConfigManager with environment: {self._env}")
        self._load_config()

    def _load_config(self) -> None:
        """بارگذاری تنظیمات از تمام منابع با رعایت ترتیب اولویت"""
        logger.debug("Starting configuration load process")

        # بارگذاری تنظیمات پیش‌فرض
        self._config.update(self._get_default_config())

        # بارگذاری از فایل‌ها
        try:
            self._load_base_config()
        except Exception as e:
            logger.warning(f"Error loading base config: {str(e)}")

        try:
            self._load_env_config()
        except Exception as e:
            logger.warning(f"Error loading environment config: {str(e)}")

        # اعمال متغیرهای محیطی
        self._load_environment_variables()

        # ثبت زمان آخرین بازخوانی
        self._last_reload = datetime.now()
        logger.info("Configuration loaded successfully")

    def _get_default_config(self) -> Dict[str, Any]:
        """
        تنظیمات پیش‌فرض سیستم برای هر محیط.

        Returns:
            Dict[str, Any]: تنظیمات پیش‌فرض
        """
        return {
            'app': {
                'name': 'AI Trading System',
                'version': '0.1.0',
            },
            'debug': self._env == 'development',
            'logging': {
                'level': 'DEBUG' if self._env == 'development' else 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }

    def _load_base_config(self) -> None:
        """
        بارگذاری تنظیمات پایه از فایل JSON.
        این تنظیمات برای تمام محیط‌ها مشترک هستند.

        Raises:
            json.JSONDecodeError: اگر فایل JSON معتبر نباشد
            OSError: در صورت بروز خطا در خواندن فایل
        """
        config_path = self.base_path / 'config' / 'base_config.json'
        try:
            if config_path.exists():
                with open(config_path) as f:
                    self._config.update(json.load(f))
                logger.debug(f"Loaded base config from {config_path}")
            else:
                logger.warning(f"Base config file not found at {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in base config: {str(e)}")
            raise
        except OSError as e:
            logger.error(f"Error reading base config: {str(e)}")
            raise

    def _load_env_config(self) -> None:
        """
        بارگذاری تنظیمات مخصوص محیط.
        این تنظیمات می‌توانند تنظیمات پایه را بازنویسی کنند.
        """
        env_config_path = self.base_path / 'config' / f'{self._env}_config.json'
        try:
            if env_config_path.exists():
                with open(env_config_path) as f:
                    env_config = json.load(f)
                    self._deep_update(self._config, env_config)
                logger.debug(f"Loaded environment config from {env_config_path}")
            else:
                logger.warning(f"Environment config file not found at {env_config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in environment config: {str(e)}")
            raise
        except OSError as e:
            logger.error(f"Error reading environment config: {str(e)}")
            raise

    def _load_environment_variables(self) -> None:
        """
        بارگذاری و اعمال متغیرهای محیطی.
        متغیرهای محیطی بالاترین اولویت را دارند و می‌توانند سایر تنظیمات را بازنویسی کنند.
        """
        env_mappings = {
            'DATABASE_URL': 'database.url',
            'REDIS_URL': 'redis.url',
            'KAFKA_BROKERS': 'kafka.brokers',
            'AI_MODEL_PATH': 'ai.model_path',
            'TRAINING_BATCH_SIZE': 'ai.training.batch_size',
            'LEARNING_RATE': 'ai.training.learning_rate',
        }

        for env_var, config_path in env_mappings.items():
            if value := os.getenv(env_var):
                self._set_nested_value(self._config, config_path.split('.'), self._convert_value(value))
                logger.debug(f"Applied environment variable {env_var}")

    def _deep_update(self, base_dict: dict, update_dict: dict) -> None:
        """
        به‌روزرسانی عمیق دیکشنری تنظیمات.

        این متد به صورت بازگشتی تمام سطوح دیکشنری را به‌روز می‌کند.

        Args:
            base_dict: دیکشنری پایه که باید به‌روز شود
            update_dict: دیکشنری جدید که باید اعمال شود
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _set_nested_value(self, config_dict: dict, path: List[str], value: Any) -> None:
        """
        تنظیم مقدار در مسیر تو در تو.

        Args:
            config_dict: دیکشنری پیکربندی
            path: لیست بخش‌های مسیر
            value: مقدار جدید
        """
        current = config_dict
        for part in path[:-1]:
            current = current.setdefault(part, {})
        current[path[-1]] = value

    def _convert_value(self, value: str) -> Union[str, int, float, bool, list]:
        """
        تبدیل مقدار رشته‌ای به نوع مناسب.

        Args:
            value: مقدار رشته‌ای

        Returns:
            مقدار تبدیل شده به نوع مناسب
        """
        # تبدیل مقادیر منطقی
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # تبدیل اعداد
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass

        # تبدیل لیست‌ها
        if value.startswith('[') and value.endswith(']'):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        # برگرداندن رشته در صورت عدم تطابق با موارد بالا
        return value

    @lru_cache(maxsize=128)
    def get(self, key: str, default: Any = None) -> Any:
        """
        دریافت مقدار تنظیمات با پشتیبانی از کلیدهای تو در تو.

        Args:
            key: کلید مورد نظر (می‌تواند شامل نقطه باشد مانند 'database.url')
            default: مقدار پیش‌فرض در صورت عدم وجود کلید

        Returns:
            مقدار پیکربندی یا مقدار پیش‌فرض

        Examples:
            >>> config.get('database.url')
            'postgresql://localhost/db'
            >>> config.get('invalid.key', 'default')
            'default'
        """
        try:
            value = self._config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Config key not found: {key}, returning default: {default}")
            return default

    def get_all(self) -> Dict[str, Any]:
        """
        دریافت تمام تنظیمات.

        Returns:
            Dict[str, Any]: کپی از تمام تنظیمات
        """
        return self._config.copy()

    def get_ai_config(self) -> Dict[str, Any]:
        """
        دریافت تنظیمات مربوط به هوش مصنوعی.

        Returns:
            Dict[str, Any]: تنظیمات بخش هوش مصنوعی
        """
        return self.get('ai', {})

    def get_training_config(self) -> Dict[str, Any]:
        """
        دریافت تنظیمات مربوط به آموزش مدل.

        Returns:
            Dict[str, Any]: تنظیمات مربوط به آموزش
        """
        return self.get('ai.training', {})

    def reload(self) -> None:
        """
        بازخوانی تنظیمات.
        این متد تمام تنظیمات را از منابع بازخوانی می‌کند.
        """
        logger.info("Reloading configuration")
        self._config.clear()
        self._load_config()
        # پاک کردن کش
        self.get.cache_clear()

    @property
    def environment(self) -> str:
        """محیط اجرایی فعلی."""
        return self._env

    @property
    def last_reload(self) -> datetime:
        """زمان آخرین بازخوانی تنظیمات."""
        return self._last_reload


# ایجاد یک نمونه سراسری برای استفاده در کل برنامه
config_manager = ConfigManager()
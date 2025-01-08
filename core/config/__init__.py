# core/config/__init__.py
"""
System Configuration Module
------------------------
Central configuration management including:
- Environment-specific settings
- System parameters
- Feature flags
- Integration configurations
- Performance tuning parameters

This module provides a unified interface for accessing and managing
all system configurations, with support for multiple environments
and real-time configuration updates.

The configuration system follows a hierarchical structure where settings
can be overridden in the following order (highest priority last):
1. Default configurations
2. Base configuration file
3. Environment-specific configuration file
4. Environment variables
5. Runtime overrides
"""

from pathlib import Path
from typing import Dict, Any, Optional

from core.config.manager import ConfigManager
from core.config.exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    ConfigKeyError
)
from core.config.constants import Environment

# ایجاد یک نمونه سراسری از مدیر پیکربندی که در کل برنامه قابل استفاده خواهد بود
config_manager = ConfigManager()

# کلاس ConfigManager را برای حفظ سازگاری با کد قبلی نگه می‌داریم
# اما پیاده‌سازی آن را به ماژول manager.py منتقل کرده‌ایم
class ConfigManager:
    """
    کلاس میانی برای حفظ سازگاری با کد قبلی
    این کلاس صرفاً به عنوان یک wrapper برای پیاده‌سازی اصلی عمل می‌کند
    """
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._env = Environment(os.getenv('APP_ENV', 'development'))

    def load_config(self, config_path: Path) -> None:
        """بارگذاری پیکربندی از فایل"""
        config_manager.load_config(config_path)

    def get(self, key: str, default: Any = None) -> Any:
        """دریافت مقدار پیکربندی"""
        return config_manager.get(key, default)

# تعریف نمادهای قابل دسترس از بیرون ماژول
__all__ = [
    'config_manager',
    'ConfigManager',
    'ConfigError',
    'ConfigFileNotFoundError',
    'ConfigValidationError',
    'ConfigKeyError',
    'Environment'
]
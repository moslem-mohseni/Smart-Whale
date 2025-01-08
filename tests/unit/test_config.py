# tests/unit/test_config.py
"""
پکیج: tests.unit.test_config
توضیحات: تست‌های واحد برای ماژول پیکربندی
نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

import os
import pytest
from pathlib import Path
from core.config import config_manager, ConfigError, Environment


def test_config_manager_initialization():
    """تست ایجاد نمونه از مدیر پیکربندی"""
    assert config_manager is not None
    assert isinstance(config_manager._env, str)


def test_environment_detection():
    """تست تشخیص محیط اجرایی"""
    # تنظیم متغیر محیطی
    os.environ['APP_ENV'] = 'development'

    # ایجاد یک نمونه جدید برای اعمال تغییرات محیطی
    from core.config.manager import ConfigManager
    config = ConfigManager()

    assert config._env == 'development'

    # پاکسازی
    del os.environ['APP_ENV']


def test_config_get_with_default():
    """تست متد get با مقدار پیش‌فرض"""
    # کلید نامعتبر باید مقدار پیش‌فرض را برگرداند
    assert config_manager.get('invalid_key', 'default_value') == 'default_value'


def test_config_get_nested_key():
    """تست دریافت مقادیر با کلیدهای تو در تو"""
    # تنظیم یک مقدار تستی
    config_manager._config = {
        'database': {
            'url': 'test_url'
        }
    }

    assert config_manager.get('database.url') == 'test_url'


def test_invalid_environment():
    """تست رفتار سیستم با محیط نامعتبر"""
    os.environ['APP_ENV'] = 'invalid_env'

    with pytest.raises(ValueError):
        from core.config.manager import ConfigManager
        ConfigManager()

    # پاکسازی
    del os.environ['APP_ENV']
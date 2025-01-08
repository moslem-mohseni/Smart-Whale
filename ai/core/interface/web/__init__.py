"""
Web Interface Package - رابط کاربری تحت وب برای سیستم هوش مصنوعی

این پکیج شامل موارد زیر است:
- سرور وب برای ارائه رابط کاربری
- قالب‌های HTML
- فایل‌های استاتیک (CSS, JavaScript)
- مدیریت درخواست‌ها و پاسخ‌ها
"""

from .server import WebServer, create_server

__all__ = ['WebServer', 'create_server']

# تنظیمات پیش‌فرض برای رابط وب
DEFAULT_CONFIG = {
    'server': {
        'host': '0.0.0.0',
        'port': 8000,
        'debug': False,
        'reload': False
    },
    'templates': {
        'cache_size': 50,
        'auto_reload': True
    },
    'static': {
        'cache_max_age': 3600
    },
    'api': {
        'prefix': '/api',
        'timeout': 30,
        'max_upload_size': 10 * 1024 * 1024  # 10MB
    }
}

# نسخه رابط کاربری
__version__ = '0.1.0'
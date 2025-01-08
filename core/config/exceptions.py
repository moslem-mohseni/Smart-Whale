# core/config/exceptions.py
"""
پکیج: core.config.exceptions
توضیحات: کلاس‌های خطای اختصاصی برای ماژول پیکربندی
نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

class ConfigError(Exception):
    """کلاس پایه برای تمام خطاهای مربوط به پیکربندی"""
    pass

class ConfigFileNotFoundError(ConfigError):
    """هنگامی که فایل پیکربندی یافت نشود"""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.message = f"Config file not found at: {file_path}"
        super().__init__(self.message)

class ConfigValidationError(ConfigError):
    """هنگامی که اعتبارسنجی پیکربندی با شکست مواجه شود"""
    def __init__(self, details: str):
        self.details = details
        self.message = f"Config validation failed: {details}"
        super().__init__(self.message)

class ConfigKeyError(ConfigError):
    """هنگامی که کلید پیکربندی درخواستی وجود نداشته باشد"""
    def __init__(self, key: str):
        self.key = key
        self.message = f"Config key not found: {key}"
        super().__init__(self.message)
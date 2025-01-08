# infrastructure/interfaces/exceptions.py

class InfrastructureError(Exception):
    """
    کلاس پایه برای تمام خطاهای زیرساخت

    این کلاس به عنوان والد تمام خطاهای تخصصی در بخش زیرساخت استفاده می‌شود
    و امکان گروه‌بندی و مدیریت یکپارچه خطاها را فراهم می‌کند.
    """

    def __init__(self, message: str = None):
        self.message = message or "خطای نامشخص در زیرساخت"
        super().__init__(self.message)


class ConnectionError(InfrastructureError):
    """خطای مربوط به برقراری یا قطع اتصال"""

    def __init__(self, message: str = None):
        super().__init__(message or "خطا در برقراری اتصال")


class OperationError(InfrastructureError):
    """خطای مربوط به عملیات (خواندن، نوشتن و غیره)"""

    def __init__(self, message: str = None):
        super().__init__(message or "خطا در انجام عملیات")


class ConfigurationError(InfrastructureError):
    """خطای مربوط به تنظیمات نادرست"""

    def __init__(self, message: str = None):
        super().__init__(message or "خطا در تنظیمات")


class TimeoutError(InfrastructureError):
    """خطای مربوط به پایان مهلت زمانی عملیات"""

    def __init__(self, message: str = None):
        super().__init__(message or "پایان مهلت زمانی عملیات")


class ValidationError(InfrastructureError):
    """خطای مربوط به اعتبارسنجی داده‌ها"""

    def __init__(self, message: str = None):
        super().__init__(message or "خطا در اعتبارسنجی داده‌ها")
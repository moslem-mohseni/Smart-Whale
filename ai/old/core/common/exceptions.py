# ai/core/common/exceptions.py

"""
کلاس‌های خطای سفارشی برای پروژه

این ماژول تمام خطاهای سفارشی مورد نیاز در پروژه را تعریف می‌کند.
خطاها به صورت سلسله‌مراتبی تعریف شده‌اند تا مدیریت خطا ساده‌تر باشد.
"""

class AIError(Exception):
    """کلاس پایه برای تمام خطاهای سیستم"""
    def __init__(self, message: str = None, *args, **kwargs):
        self.message = message or "An error occurred in the AI system"
        super().__init__(self.message, *args)


class ConfigurationError(AIError):
    """خطاهای مربوط به پیکربندی سیستم"""
    def __init__(self, message: str = None):
        super().__init__(message or "Configuration error occurred")


class ProcessingError(AIError):
    """خطاهای مربوط به پردازش داده"""
    def __init__(self, message: str = None):
        super().__init__(message or "Processing error occurred")


class ValidationError(AIError):
    """خطاهای مربوط به اعتبارسنجی داده"""
    def __init__(self, message: str = None):
        super().__init__(message or "Validation error occurred")


class AnalysisError(AIError):
    """خطاهای مربوط به تحلیل داده"""
    def __init__(self, message: str = None):
        super().__init__(message or "Analysis error occurred")


class LearningError(AIError):
    """خطاهای مربوط به یادگیری مدل"""
    def __init__(self, message: str = None):
        super().__init__(message or "Learning error occurred")


class ResourceError(AIError):
    """خطاهای مربوط به منابع سیستم"""
    def __init__(self, message: str = None):
        super().__init__(message or "Resource error occurred")


class APIError(AIError):
    """خطاهای مربوط به API"""
    def __init__(self, message: str = None, status_code: int = None):
        self.status_code = status_code
        super().__init__(message or f"API error occurred (status: {status_code})")


class DataLoadingError(Exception):
    """خطای بارگذاری داده‌ها"""
    pass


class TrainingError(AIError):
    """خطاهای مربوط به فرآیند آموزش"""
    def __init__(self, message: str = None):
        super().__init__(message or "Training error occurred")


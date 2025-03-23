class CollectorError(Exception):
    """کلاس پایه برای خطاهای collector"""
    pass


class CollectionError(CollectorError):
    """خطا در جمع‌آوری داده"""
    pass


class ProcessingError(CollectorError):
    """خطا در پردازش داده"""
    pass


class StorageError(CollectorError):
    """خطا در ذخیره‌سازی"""
    pass


class ValidationError(CollectorError):
    """خطا در اعتبارسنجی داده"""
    pass


class CleanupError(CollectorError):
    """خطا در پاکسازی منابع"""
    pass


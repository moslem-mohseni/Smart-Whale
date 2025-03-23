class ParsBERTError(Exception):
    """خطای پایه ParsBERT"""
    pass


class ModelNotInitializedError(ParsBERTError):
    """خطای عدم مقداردهی اولیه مدل"""
    pass


class ProcessingError(ParsBERTError):
    """خطای پردازش متن"""
    pass


class CacheError(ParsBERTError):
    """خطاهای مربوط به کش"""
    pass


class RetryableError(ParsBERTError):
    """خطاهای قابل تلاش مجدد"""
    pass


class ModelOverloadError(RetryableError):
    """خطای بار اضافی روی مدل"""
    pass


class InvalidInputError(ParsBERTError):
    """خطای ورودی نامعتبر"""
    pass

# ai/core/common/utils.py

"""
توابع و کلاس‌های کمکی عمومی
"""

import time
import logging
from typing import Any, Callable, TypeVar, Optional
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_on_exception(
    retries: int = 3,
    delay: int = 1,
    exceptions: tuple = (Exception,),
    logger: Optional[logging.Logger] = None
) -> Callable:
    """دکوراتور تلاش مجدد در صورت خطا"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1}/{retries} failed: {str(e)}"
                        )
                    if attempt < retries - 1:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def measure_time(func: Callable[..., T]) -> Callable[..., T]:
    """دکوراتور اندازه‌گیری زمان اجرای تابع"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def validate_not_none(value: Any, name: str) -> None:
    """اعتبارسنجی مقدار not None"""
    if value is None:
        raise ValidationError(f"{name} cannot be None")
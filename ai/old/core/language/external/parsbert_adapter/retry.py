import asyncio
import logging
from .errors import RetryableError, ModelOverloadError, CacheError

logger = logging.getLogger(__name__)

def retry_operation(max_attempts: int = 3, delay: float = 1.0):
    """
    دکوراتوری برای اجرای مجدد عملیات‌هایی که ممکن است دچار خطاهای موقت شوند.
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ModelOverloadError, CacheError) as e:
                    last_error = e
                    logger.warning(f"Retrying {func.__name__} due to error: {e} (Attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(delay * (attempt + 1))
                except RetryableError as e:
                    logger.error(f"Non-recoverable retry error in {func.__name__}: {e}")
                    raise e  # خطاهای غیرقابل جبران نباید مجدد تلاش شوند
            raise last_error
        return wrapper
    return decorator

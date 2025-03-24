import asyncio
from functools import wraps
from typing import Callable, Any


def retry_async(retries: int = 3, delay: float = 1.0):
    """
    مکانیزم Retry برای اجرای مجدد درخواست‌های ناموفق با افزایش زمان تاخیر
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries - 1:
                        await asyncio.sleep(delay * (attempt + 1))
            raise last_exception
        return wrapper
    return decorator

# مثال استفاده از Retry Mechanism
# @retry_async(retries=5, delay=2)
# async def fetch_data():
#     raise Exception("Network error!")

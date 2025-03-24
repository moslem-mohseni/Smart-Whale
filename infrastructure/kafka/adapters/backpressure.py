import asyncio
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)


class BackpressureHandler:
    """
    مدیریت فشار ورودی (Backpressure) برای Kafka
    """

    def __init__(self, max_concurrent_requests: int = 10, queue_size: int = 50):
        """
        مقداردهی اولیه BackpressureHandler

        :param max_concurrent_requests: حداکثر تعداد پردازش‌های همزمان
        :param queue_size: حداکثر ظرفیت صف درخواست‌ها
        """
        self.max_concurrent_requests = max_concurrent_requests
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        اجرای یک تابع تحت مدیریت Backpressure

        :param func: تابعی که باید اجرا شود
        """
        await self.queue.put((func, args, kwargs))

        async with self.semaphore:
            task_func, task_args, task_kwargs = await self.queue.get()

            try:
                return await task_func(*task_args, **task_kwargs)
            except Exception as e:
                logger.error(f"Backpressure execution failed: {e}")
                raise

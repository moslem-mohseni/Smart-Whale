import asyncio
import logging
import json
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.monitoring.alerts import AlertNotifier
from typing import Callable, Any

logging.basicConfig(level=logging.INFO)

class ErrorHandler:
    """
    مدیریت خطاها و تلاش مجدد (Retry) در Pipeline.
    """

    def __init__(self, max_retries: int = 5, backoff_factor: float = 2.0):
        """
        مقداردهی اولیه.

        :param max_retries: حداکثر تعداد تلاش مجدد
        :param backoff_factor: ضریب افزایش فاصله بین Retry‌ها (Exponential Backoff)
        """
        self.kafka_service = KafkaService()
        self.cache_service = CacheService()
        self.alert_notifier = AlertNotifier()
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    async def connect(self) -> None:
        """ اتصال به Kafka، Redis و Monitoring System. """
        await self.kafka_service.connect()
        await self.cache_service.connect()

    async def handle_error(self, error: Exception, stage: str, data_id: str, retry_count: int) -> None:
        """
        ثبت و گزارش خطا.

        :param error: خطای رخ داده
        :param stage: نام مرحله‌ای که خطا در آن رخ داده است
        :param data_id: شناسه داده‌ای که خطا روی آن اتفاق افتاده است
        :param retry_count: تعداد تلاش‌های انجام‌شده
        """
        error_message = f"❌ خطا در مرحله {stage} برای داده {data_id}: {error} (تلاش {retry_count}/{self.max_retries})"
        logging.error(error_message)

        # ذخیره اطلاعات خطا در Redis
        error_key = f"error:{stage}:{data_id}"
        await self.cache_service.set(error_key, error_message, ttl=86400)

        # ارسال گزارش خطا به Kafka
        await self.kafka_service.send_message({
            "topic": "pipeline_errors",
            "content": {"stage": stage, "data_id": data_id, "error": str(error), "retry_count": retry_count}
        })

        # ارسال هشدار در صورت رسیدن به حداکثر تلاش مجدد
        if retry_count >= self.max_retries:
            await self.alert_notifier.send_alert(f"🚨 پردازش داده {data_id} در مرحله {stage} پس از {self.max_retries} تلاش ناموفق بود!")

    async def retry_operation(self, operation: Callable[[], Any], stage: str, data_id: str) -> Any:
        """
        اجرای مجدد عملیات در صورت بروز خطا.

        :param operation: تابعی که باید اجرا شود
        :param stage: نام مرحله‌ای که عملیات در آن اجرا می‌شود
        :param data_id: شناسه داده‌ای که پردازش می‌شود
        :return: خروجی تابع در صورت موفقیت یا None در صورت شکست
        """
        retry_count = 0
        delay = 1  # تأخیر اولیه قبل از تلاش مجدد

        while retry_count < self.max_retries:
            try:
                return await operation()
            except Exception as error:
                await self.handle_error(error, stage, data_id, retry_count + 1)
                retry_count += 1
                await asyncio.sleep(delay)
                delay *= self.backoff_factor  # افزایش نمایی تأخیر بین تلاش‌ها

        logging.error(f"⛔ پردازش داده {data_id} در مرحله {stage} پس از {self.max_retries} تلاش متوقف شد.")
        return None

    async def close(self) -> None:
        """ قطع اتصال از Kafka و Redis. """
        await self.kafka_service.disconnect()
        await self.cache_service.disconnect()

# مقداردهی اولیه و راه‌اندازی ErrorHandler
async def start_error_handler():
    error_handler = ErrorHandler()
    await error_handler.connect()

asyncio.create_task(start_error_handler())

import asyncio
import logging
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.redis.service.cache_service import CacheService
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)

class DependencyManager:
    """
    مدیریت وابستگی بین مراحل پردازش در Pipeline.
    """

    def __init__(self):
        """
        مقداردهی اولیه.
        """
        self.kafka_service = KafkaService()
        self.cache_service = CacheService()

    async def connect(self) -> None:
        """ اتصال به Kafka و Redis. """
        await self.kafka_service.connect()
        await self.cache_service.connect()

    async def check_dependencies(self, stage: str, data_id: str, required_stages: List[str]) -> bool:
        """
        بررسی وضعیت آماده بودن مراحل قبلی برای اجرای یک مرحله جدید.

        :param stage: نام مرحله فعلی
        :param data_id: شناسه داده
        :param required_stages: لیستی از مراحل موردنیاز قبل از اجرای این مرحله
        :return: True اگر تمام وابستگی‌ها آماده باشند، False در غیر این صورت
        """
        for required_stage in required_stages:
            dependency_key = f"dependency:{data_id}:{required_stage}"
            dependency_status = await self.cache_service.get(dependency_key)
            if not dependency_status:
                logging.warning(f"⚠️ داده با ID {data_id} هنوز مرحله {required_stage} را کامل نکرده است.")
                return False
        logging.info(f"✅ تمام وابستگی‌های {stage} برای داده {data_id} آماده است.")
        return True

    async def mark_stage_complete(self, stage: str, data_id: str) -> None:
        """
        ثبت تکمیل یک مرحله پردازشی در Redis.

        :param stage: نام مرحله‌ای که کامل شده است
        :param data_id: شناسه داده‌ای که پردازش شده است
        """
        dependency_key = f"dependency:{data_id}:{stage}"
        await self.cache_service.set(dependency_key, "completed", ttl=3600)
        logging.info(f"✅ مرحله {stage} برای داده {data_id} تکمیل و در Redis ثبت شد.")

        # انتشار پیام در Kafka برای اعلام تکمیل این مرحله
        await self.kafka_service.send_message({"topic": "pipeline_status", "content": {"stage": stage, "data_id": data_id}})
        logging.info(f"📢 وضعیت تکمیل مرحله {stage} برای داده {data_id} در Kafka منتشر شد.")

    async def wait_for_dependencies(self, stage: str, data_id: str, required_stages: List[str], retry_interval: int = 5) -> bool:
        """
        انتظار برای آماده شدن وابستگی‌های موردنیاز یک مرحله.

        :param stage: نام مرحله‌ای که منتظر اجرا است
        :param data_id: شناسه داده
        :param required_stages: لیست مراحل موردنیاز قبل از اجرا
        :param retry_interval: فاصله زمانی بین بررسی‌های مجدد (بر حسب ثانیه)
        :return: True در صورت آماده شدن، False در صورت شکست
        """
        retries = 0
        max_retries = 10

        while retries < max_retries:
            if await self.check_dependencies(stage, data_id, required_stages):
                return True
            logging.info(f"🔄 در انتظار تکمیل وابستگی‌های {stage} برای داده {data_id}...")
            await asyncio.sleep(retry_interval)
            retries += 1

        logging.error(f"❌ وابستگی‌های {stage} برای داده {data_id} بعد از {max_retries} تلاش آماده نشدند!")
        return False

    async def close(self) -> None:
        """ قطع اتصال از Kafka و Redis. """
        await self.kafka_service.disconnect()
        await self.cache_service.disconnect()

# مقداردهی اولیه و راه‌اندازی DependencyManager
async def start_dependency_manager():
    dependency_manager = DependencyManager()
    await dependency_manager.connect()

asyncio.create_task(start_dependency_manager())

import asyncio
import json
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.redis.service.cache_service import CacheService
from data.collectors import BaseCollector
from typing import Dict, Any

class CollectorStage:
    """
    مرحله جمع‌آوری داده از Kafka و Collectors.
    """

    def __init__(self, kafka_topic: str, cache_ttl: int = 3600):
        """
        مقداردهی اولیه.

        :param kafka_topic: نام تاپیک Kafka برای دریافت داده‌ها
        :param cache_ttl: مدت زمان نگهداری داده در Redis (به ثانیه)
        """
        self.kafka_service = KafkaService()
        self.cache_service = CacheService()
        self.collector = BaseCollector()
        self.kafka_topic = kafka_topic
        self.cache_ttl = cache_ttl

    async def connect(self) -> None:
        """ اتصال به Kafka و Redis. """
        await self.kafka_service.connect()
        await self.cache_service.connect()

    async def consume_data(self) -> None:
        """
        دریافت داده از Kafka و پردازش اولیه.
        """
        async def process_message(message: Dict[str, Any]):
            """
            پردازش هر پیام Kafka.

            :param message: داده‌ی دریافتی از Kafka
            """
            key = f"collector_stage:{message['id']}"
            cached_data = await self.cache_service.get(key)

            if cached_data:
                print(f"⚠️ داده با ID {message['id']} قبلاً پردازش شده، رد شد.")
                return

            # پردازش اولیه داده‌ها
            processed_data = await self.collector.collect(message)
            if not processed_data:
                print(f"⚠️ جمع‌آوری داده برای {message['id']} ناموفق بود.")
                return

            # کش کردن داده برای جلوگیری از پردازش تکراری
            await self.cache_service.set(key, json.dumps(processed_data), ttl=self.cache_ttl)

            # ارسال داده برای مرحله پردازش
            await self.process_data(processed_data)

        await self.kafka_service.subscribe(self.kafka_topic, "collector_group", process_message)

    async def process_data(self, data: Dict[str, Any]) -> None:
        """
        ارسال داده به مرحله پردازش.

        :param data: داده پردازش‌شده
        """
        print(f"✅ داده پردازش اولیه شد و به مرحله بعدی ارسال می‌شود: {data}")

    async def close(self) -> None:
        """ قطع اتصال از Kafka و Redis. """
        await self.kafka_service.disconnect()
        await self.cache_service.disconnect()

# مقداردهی اولیه و راه‌اندازی مصرف‌کننده Kafka
async def start_collector_stage():
    collector_stage = CollectorStage(kafka_topic="raw_data")
    await collector_stage.connect()
    await collector_stage.consume_data()

# اجرای مصرف‌کننده Kafka به صورت ناهمزمان
asyncio.create_task(start_collector_stage())

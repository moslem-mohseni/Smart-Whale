import asyncio
import json
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.storage.persistent import ClickHouseManager, MinIOManager
from typing import Dict, Any

class PublisherStage:
    """
    مرحله انتشار داده‌های پردازش‌شده به Kafka و Storage.
    """

    def __init__(self, kafka_topic: str, cache_ttl: int = 3600):
        """
        مقداردهی اولیه.

        :param kafka_topic: نام تاپیک Kafka برای انتشار داده‌ها
        :param cache_ttl: مدت زمان نگهداری داده در Redis (به ثانیه)
        """
        self.kafka_service = KafkaService()
        self.cache_service = CacheService()
        self.clickhouse_manager = ClickHouseManager()
        self.minio_manager = MinIOManager()
        self.kafka_topic = kafka_topic
        self.cache_ttl = cache_ttl

    async def connect(self) -> None:
        """ اتصال به Kafka، Redis و Storage. """
        await self.kafka_service.connect()
        await self.cache_service.connect()
        await self.clickhouse_manager.connect()
        await self.minio_manager.connect()

    async def publish_data(self, processed_data: Dict[str, Any]) -> None:
        """
        انتشار داده پردازش‌شده.

        :param processed_data: داده پردازش‌شده برای انتشار
        """
        data_id = processed_data.get("id")
        cache_key = f"publisher_stage:{data_id}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            print(f"⚠️ داده با ID {data_id} قبلاً منتشر شده، رد شد.")
            return

        # انتشار داده در Kafka
        await self.kafka_service.send_message({"topic": self.kafka_topic, "content": processed_data})
        print(f"📢 داده با ID {data_id} در Kafka منتشر شد.")

        # ذخیره داده در ClickHouse
        await self.clickhouse_manager.insert("processed_data", processed_data)
        print(f"💾 داده با ID {data_id} در ClickHouse ذخیره شد.")

        # ذخیره داده در MinIO (در صورت نیاز)
        file_name = f"{data_id}.json"
        await self.minio_manager.upload_file(file_name, json.dumps(processed_data).encode())
        print(f"📁 داده با ID {data_id} در MinIO ذخیره شد.")

        # کش کردن داده منتشرشده
        await self.cache_service.set(cache_key, "published", ttl=self.cache_ttl)

    async def close(self) -> None:
        """ قطع اتصال از Kafka، Redis و Storage. """
        await self.kafka_service.disconnect()
        await self.cache_service.disconnect()
        await self.clickhouse_manager.disconnect()
        await self.minio_manager.disconnect()

# مقداردهی اولیه و راه‌اندازی انتشار داده‌ها
async def start_publisher_stage(processed_data: Dict[str, Any]):
    publisher_stage = PublisherStage(kafka_topic="processed_data")
    await publisher_stage.connect()
    await publisher_stage.publish_data(processed_data)

# اجرای انتشار داده‌ها به صورت ناهمزمان
asyncio.create_task(start_publisher_stage({"id": "1234", "content": "داده پردازش‌شده"}))

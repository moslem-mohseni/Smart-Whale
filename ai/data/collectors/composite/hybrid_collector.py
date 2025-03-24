import json
import asyncio
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.cache.hash_cache import HashCache


class HybridCollector:
    """
    جمع‌آوری داده‌های ترکیبی از چندین منبع و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
    """

    def __init__(self, kafka_topic):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.hash_cache = HashCache()

    async def process_data(self, data_source, data_list):
        """پردازش داده‌ها از منبع مشخص و ارسال آن‌ها به Kafka"""
        processed_count = 0
        for data in data_list:
            data_hash = hash(json.dumps(data, ensure_ascii=False))
            if await self.hash_cache.get_file_hash(str(data_hash)):
                print(f"⚠ داده تکراری شناسایی شد: {data}")
                continue

            await self.hash_cache.store_file_hash(str(data_hash))
            message = {"source": data_source, "data": data}
            self.kafka_service.send_message(self.kafka_topic, json.dumps(message, ensure_ascii=False))
            processed_count += 1
        return processed_count


if __name__ == "__main__":
    kafka_topic = "hybrid_data"
    collector = HybridCollector(kafka_topic)

    test_data_source = "combined_source"
    test_data_list = [
        {"type": "image", "url": "https://example.com/image.jpg"},
        {"type": "video", "url": "https://example.com/video.mp4"},
        {"type": "audio", "url": "https://example.com/audio.mp3"}
    ]

    try:
        loop = asyncio.get_event_loop()
        processed_count = loop.run_until_complete(collector.process_data(test_data_source, test_data_list))
        print(f"✅ {processed_count} داده از منابع ترکیبی پردازش و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش داده‌های ترکیبی: {e}")
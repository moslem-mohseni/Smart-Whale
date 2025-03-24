import json
import asyncio
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.cache.hash_cache import HashCache


class RelationManager:
    """
    مدیریت روابط بین داده‌های جمع‌آوری‌شده و ارسال اطلاعات مرتبط به Kafka
    """

    def __init__(self, kafka_topic):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.hash_cache = HashCache()
        self.relations = {}

    async def add_relation(self, primary_data, related_data):
        """ایجاد ارتباط بین داده‌های مختلف و ارسال به Kafka"""
        primary_hash = hash(json.dumps(primary_data, ensure_ascii=False))
        related_hash = hash(json.dumps(related_data, ensure_ascii=False))

        if primary_hash not in self.relations:
            self.relations[primary_hash] = []

        if related_hash in self.relations[primary_hash]:
            print(f"⚠ داده مرتبط از قبل ثبت شده است: {related_data}")
            return False

        self.relations[primary_hash].append(related_hash)
        message = {"primary": primary_data, "related": related_data}
        self.kafka_service.send_message(self.kafka_topic, json.dumps(message, ensure_ascii=False))
        return True


if __name__ == "__main__":
    kafka_topic = "relation_data"
    manager = RelationManager(kafka_topic)

    primary_data = {"type": "article", "title": "AI Innovations"}
    related_data = {"type": "video", "title": "AI and the Future"}

    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(manager.add_relation(primary_data, related_data))
        if result:
            print(f"✅ رابطه جدید پردازش و به Kafka ارسال شد.")
        else:
            print("⚠ رابطه قبلاً در سیستم ثبت شده است.")
    except Exception as e:
        print(f"❌ خطا در مدیریت روابط داده‌ها: {e}")

import json
import hashlib
import requests
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class WebImageCollector:
    """
    جمع‌آوری تصاویر از منابع عمومی وب و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
    """

    def __init__(self, kafka_topic):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.file_service = FileService()
        self.hash_cache = HashCache()

    def calculate_image_hash(self, image_data):
        """محاسبه هش تصویر برای تشخیص فایل‌های تکراری"""
        return hashlib.sha256(image_data).hexdigest()

    async def fetch_random_image(self):
        """دریافت یک تصویر تصادفی از Picsum Photos"""
        image_url = "https://picsum.photos/500/500"
        response = requests.get(image_url)
        if response.status_code != 200:
            raise ValueError("⚠ خطا در دریافت تصویر از Picsum Photos")
        return response.content, image_url

    async def process_and_publish(self):
        """دریافت تصویر، بررسی هش، و ارسال به Kafka"""
        image_data, image_url = await self.fetch_random_image()
        image_hash = self.calculate_image_hash(image_data)

        if await self.hash_cache.get_file_hash(image_hash):
            print(f"⚠ تصویر تکراری شناسایی شد: {image_url}")
            return False

        await self.hash_cache.store_file_hash(image_hash)
        message = {"hash": image_hash, "source": image_url}
        self.kafka_service.send_message(self.kafka_topic, json.dumps(message, ensure_ascii=False))
        return True


if __name__ == "__main__":
    kafka_topic = "web_images"
    collector = WebImageCollector(kafka_topic)

    try:
        result = collector.process_and_publish()
        if result:
            print(f"✅ تصویر جدید پردازش و به Kafka ارسال شد.")
        else:
            print("⚠ تصویر قبلاً در سیستم وجود دارد.")
    except Exception as e:
        print(f"❌ خطا در پردازش تصویر وب: {e}")
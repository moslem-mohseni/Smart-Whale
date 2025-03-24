import json
import hashlib
from telethon import TelegramClient
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class SocialImageCollector:
    """
    جمع‌آوری تصاویر از تلگرام و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
    """

    def __init__(self, kafka_topic, api_id, api_hash, session_name):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.file_service = FileService()
        self.hash_cache = HashCache()
        self.client = TelegramClient(session_name, api_id, api_hash)

    def calculate_image_hash(self, image_data):
        """محاسبه هش تصویر برای تشخیص فایل‌های تکراری"""
        return hashlib.sha256(image_data).hexdigest()

    async def fetch_telegram_images(self, channel):
        """دریافت تصاویر از یک کانال تلگرام"""
        images = []
        async with self.client:
            async for message in self.client.iter_messages(channel, filter='photo'):
                image_data = await message.download_media(bytes)
                image_hash = self.calculate_image_hash(image_data)
                if await self.hash_cache.get_file_hash(image_hash):
                    print(f"⚠ تصویر تکراری شناسایی شد: {message.id}")
                    continue

                await self.hash_cache.store_file_hash(image_hash)
                images.append({"hash": image_hash, "source": f"telegram:{message.id}"})
        return images

    async def process_and_publish(self, channel):
        """دریافت تصاویر، بررسی هش، و ارسال به Kafka"""
        images = await self.fetch_telegram_images(channel)
        for image in images:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(image, ensure_ascii=False))
        return len(images)


if __name__ == "__main__":
    kafka_topic = "social_images"
    api_id = "your_api_id"
    api_hash = "your_api_hash"
    session_name = "telegram_scraper"
    collector = SocialImageCollector(kafka_topic, api_id, api_hash, session_name)

    test_channel = "@example_channel"

    try:
        image_count = collector.process_and_publish(test_channel)
        print(f"✅ {image_count} تصویر از تلگرام پردازش و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش تصاویر تلگرام: {e}")
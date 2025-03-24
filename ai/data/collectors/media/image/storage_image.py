import os
import json
import hashlib
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class StorageImageCollector:
    """
    جمع‌آوری تصاویر از یک مسیر ذخیره‌سازی و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
    """

    def __init__(self, kafka_topic, image_directory):
        self.kafka_topic = kafka_topic
        self.image_directory = image_directory
        self.kafka_service = KafkaService()
        self.file_service = FileService()
        self.hash_cache = HashCache()

    def calculate_image_hash(self, file_path):
        """محاسبه هش تصویر برای تشخیص فایل‌های تکراری"""
        with open(file_path, "rb") as file:
            file_data = file.read()
        return hashlib.sha256(file_data).hexdigest()

    async def fetch_images(self):
        """استخراج مسیر تصاویر ذخیره‌شده در دایرکتوری مشخص و بررسی تکراری بودن آن‌ها"""
        images = []
        for file_name in os.listdir(self.image_directory):
            if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                file_path = os.path.join(self.image_directory, file_name)
                image_hash = self.calculate_image_hash(file_path)

                if await self.hash_cache.get_file_hash(image_hash):
                    print(f"⚠ تصویر تکراری شناسایی شد: {file_name}")
                    continue

                await self.hash_cache.store_file_hash(image_hash)
                images.append({"file_name": file_name, "file_path": file_path, "hash": image_hash})

        return images

    async def process_and_publish(self):
        """استخراج تصاویر از دایرکتوری و ارسال به Kafka با بررسی تکراری بودن"""
        images = await self.fetch_images()
        for image in images:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(image, ensure_ascii=False))
        return len(images)


if __name__ == "__main__":
    kafka_topic = "storage_images"
    image_directory = "./images"
    collector = StorageImageCollector(kafka_topic, image_directory)

    try:
        image_count = collector.process_and_publish()
        print(f"✅ {image_count} تصویر از مسیر {image_directory} استخراج و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش تصاویر ذخیره‌شده: {e}")

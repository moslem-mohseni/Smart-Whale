import json
import hashlib
import asyncio
import requests
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class AparatVideoCollector:
    """
    جمع‌آوری ویدیو از آپارات، دانلود و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
    """

    def __init__(self, kafka_topic, download_path):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.file_service = FileService()
        self.hash_cache = HashCache()
        self.download_path = download_path

    def calculate_video_hash(self, video_url):
        """محاسبه هش ویدیو برای تشخیص فایل‌های تکراری"""
        return hashlib.sha256(video_url.encode()).hexdigest()

    async def fetch_aparat_video(self, video_url):
        """دریافت اطلاعات ویدیو از آپارات و بررسی تکراری بودن آن"""
        video_hash = self.calculate_video_hash(video_url)

        if await self.hash_cache.get_file_hash(video_hash):
            print(f"⚠ ویدیو تکراری شناسایی شد: {video_url}")
            return None

        response = requests.get(f"https://www.aparat.com/etc/api/video/videohash/{video_url}")
        if response.status_code != 200:
            raise ValueError("⚠ خطا در دریافت اطلاعات ویدیو از آپارات")

        video_info = response.json().get("video")
        if not video_info:
            return None

        video_data = {
            "title": video_info.get("title"),
            "url": video_url,
            "duration": video_info.get("duration"),
            "thumbnail": video_info.get("big_poster")
        }

        await self.hash_cache.store_file_hash(video_hash)
        return video_data, video_hash

    async def download_video(self, video_url, video_hash):
        """دانلود ویدیو و ذخیره آن در سیستم فایل"""
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            raise ValueError("⚠ خطا در دانلود ویدیو از آپارات")

        file_path = f"{self.download_path}/{video_hash}.mp4"
        with open(file_path, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=1024):
                video_file.write(chunk)

        return file_path

    async def process_and_publish(self, video_url):
        """دریافت ویدیو، بررسی هش، دانلود و ارسال به Kafka"""
        video_data, video_hash = await self.fetch_aparat_video(video_url)
        if not video_data:
            return False

        file_path = await self.download_video(video_url, video_hash)

        video_data["file_path"] = file_path
        self.kafka_service.send_message(self.kafka_topic, json.dumps(video_data, ensure_ascii=False))
        return True


if __name__ == "__main__":
    kafka_topic = "aparat_videos"
    download_path = "./downloads"
    collector = AparatVideoCollector(kafka_topic, download_path)

    test_video_url = "https://www.aparat.com/v/example_id"

    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(collector.process_and_publish(test_video_url))
        if result:
            print(f"✅ ویدیو جدید پردازش، دانلود و به Kafka ارسال شد.")
        else:
            print("⚠ ویدیو قبلاً در سیستم وجود دارد.")
    except Exception as e:
        print(f"❌ خطا در پردازش ویدیو آپارات: {e}")
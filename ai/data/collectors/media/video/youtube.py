import json
import hashlib
import asyncio
from pytube import YouTube
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class YouTubeVideoCollector:
    """
    جمع‌آوری ویدیو از YouTube، دانلود و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
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

    async def fetch_youtube_video(self, video_url):
        """دریافت اطلاعات ویدیو از YouTube و بررسی تکراری بودن آن"""
        video_hash = self.calculate_video_hash(video_url)

        if await self.hash_cache.get_file_hash(video_hash):
            print(f"⚠ ویدیو تکراری شناسایی شد: {video_url}")
            return None

        yt = YouTube(video_url)
        video_data = {
            "title": yt.title,
            "url": video_url,
            "duration": yt.length,
            "thumbnail": yt.thumbnail_url
        }

        await self.hash_cache.store_file_hash(video_hash)
        return yt, video_data, video_hash

    async def download_video(self, yt, video_hash):
        """دانلود ویدیو و ذخیره آن در سیستم فایل"""
        loop = asyncio.get_running_loop()
        file_path = await loop.run_in_executor(None,
                                               lambda: yt.streams.get_highest_resolution().download(self.download_path))
        return file_path

    async def process_and_publish(self, video_url):
        """دریافت ویدیو، بررسی هش، دانلود و ارسال به Kafka"""
        yt, video_data, video_hash = await self.fetch_youtube_video(video_url)
        if not yt:
            return False

        file_path = await self.download_video(yt, video_hash)

        video_data["file_path"] = file_path
        self.kafka_service.send_message(self.kafka_topic, json.dumps(video_data, ensure_ascii=False))
        return True


if __name__ == "__main__":
    kafka_topic = "youtube_videos"
    download_path = "./downloads"
    collector = YouTubeVideoCollector(kafka_topic, download_path)

    test_video_url = "https://www.youtube.com/watch?v=example_id"

    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(collector.process_and_publish(test_video_url))
        if result:
            print(f"✅ ویدیو جدید پردازش، دانلود و به Kafka ارسال شد.")
        else:
            print("⚠ ویدیو قبلاً در سیستم وجود دارد.")
    except Exception as e:
        print(f"❌ خطا در پردازش ویدیو YouTube: {e}")
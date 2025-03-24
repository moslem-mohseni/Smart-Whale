import json
import hashlib
import asyncio
from telethon import TelegramClient
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class SocialVideoCollector:
    """
    جمع‌آوری ویدیو از تلگرام و ارسال اطلاعات آن‌ها به Kafka با جلوگیری از ذخیره‌سازی تکراری
    """

    def __init__(self, kafka_topic, api_id, api_hash, session_name, download_path):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.file_service = FileService()
        self.hash_cache = HashCache()
        self.download_path = download_path
        self.client = TelegramClient(session_name, api_id, api_hash)

    def calculate_video_hash(self, video_data):
        """محاسبه هش ویدیو برای تشخیص فایل‌های تکراری"""
        return hashlib.sha256(video_data).hexdigest()

    async def fetch_telegram_videos(self, channel):
        """دریافت ویدیوها از یک کانال تلگرام"""
        videos = []
        async with self.client:
            async for message in self.client.iter_messages(channel, filter='video'):
                video_data = await message.download_media(bytes)
                video_hash = self.calculate_video_hash(video_data)

                if await self.hash_cache.get_file_hash(video_hash):
                    print(f"⚠ ویدیو تکراری شناسایی شد: {message.id}")
                    continue

                await self.hash_cache.store_file_hash(video_hash)
                file_path = f"{self.download_path}/{video_hash}.mp4"
                with open(file_path, "wb") as video_file:
                    video_file.write(video_data)

                videos.append({"hash": video_hash, "file_path": file_path, "source": f"telegram:{message.id}"})
        return videos

    async def process_and_publish(self, channel):
        """دریافت ویدیوها، بررسی هش، دانلود و ارسال به Kafka"""
        videos = await self.fetch_telegram_videos(channel)
        for video in videos:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(video, ensure_ascii=False))
        return len(videos)


if __name__ == "__main__":
    kafka_topic = "social_videos"
    api_id = "your_api_id"
    api_hash = "your_api_hash"
    session_name = "telegram_scraper"
    download_path = "./downloads"
    collector = SocialVideoCollector(kafka_topic, api_id, api_hash, session_name, download_path)

    test_channel = "@example_channel"

    try:
        loop = asyncio.get_event_loop()
        video_count = loop.run_until_complete(collector.process_and_publish(test_channel))
        print(f"✅ {video_count} ویدیو از تلگرام پردازش و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش ویدیوهای تلگرام: {e}")

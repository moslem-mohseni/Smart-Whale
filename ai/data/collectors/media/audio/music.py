import json
import hashlib
import requests
import asyncio
from telethon import TelegramClient
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class MusicCollector:
    """
    جمع‌آوری موسیقی‌های فارسی از تلگرام و ارسال اطلاعات آن‌ها به Kafka
    """

    def __init__(self, kafka_topic, api_id, api_hash, session_name, download_path):
        self.kafka_topic = kafka_topic
        self.kafka_service = KafkaService()
        self.file_service = FileService()
        self.hash_cache = HashCache()
        self.download_path = download_path
        self.client = TelegramClient(session_name, api_id, api_hash)

    def calculate_audio_hash(self, audio_data):
        """محاسبه هش فایل صوتی برای تشخیص فایل‌های تکراری"""
        return hashlib.sha256(audio_data).hexdigest()

    async def fetch_telegram_music(self, channel):
        """دریافت موسیقی‌های فارسی از تلگرام"""
        music_files = []
        async with self.client:
            async for message in self.client.iter_messages(channel, filter='audio'):
                audio_data = await message.download_media(bytes)
                audio_hash = self.calculate_audio_hash(audio_data)

                if await self.hash_cache.get_file_hash(audio_hash):
                    print(f"⚠ موسیقی تکراری شناسایی شد: {message.id}")
                    continue

                await self.hash_cache.store_file_hash(audio_hash)
                file_path = f"{self.download_path}/{audio_hash}.mp3"
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_data)

                music_files.append({"hash": audio_hash, "file_path": file_path, "source": f"telegram:{message.id}"})
        return music_files

    async def process_and_publish(self, channel):
        """دریافت موسیقی‌ها، بررسی هش، دانلود و ارسال به Kafka"""
        music_files = await self.fetch_telegram_music(channel)

        for music in music_files:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(music, ensure_ascii=False))

        return len(music_files)


if __name__ == "__main__":
    kafka_topic = "music_data"
    api_id = "your_api_id"
    api_hash = "your_api_hash"
    session_name = "telegram_scraper"
    download_path = "./downloads"
    collector = MusicCollector(kafka_topic, api_id, api_hash, session_name, download_path)

    test_channel = "@example_music_channel"

    try:
        loop = asyncio.get_event_loop()
        music_count = loop.run_until_complete(collector.process_and_publish(test_channel))
        print(f"✅ {music_count} موسیقی از تلگرام پردازش و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش موسیقی‌ها: {e}")

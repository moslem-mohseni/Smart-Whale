import json
import hashlib
import requests
import asyncio
from telethon import TelegramClient
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class PodcastCollector:
    """
    جمع‌آوری پادکست‌های فارسی از تلگرام و FarsPod و ارسال اطلاعات آن‌ها به Kafka
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

    async def fetch_telegram_podcasts(self, channel):
        """دریافت پادکست‌های فارسی از تلگرام"""
        podcasts = []
        async with self.client:
            async for message in self.client.iter_messages(channel, filter='audio'):
                audio_data = await message.download_media(bytes)
                audio_hash = self.calculate_audio_hash(audio_data)

                if await self.hash_cache.get_file_hash(audio_hash):
                    print(f"⚠ پادکست تکراری شناسایی شد: {message.id}")
                    continue

                await self.hash_cache.store_file_hash(audio_hash)
                file_path = f"{self.download_path}/{audio_hash}.mp3"
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_data)

                podcasts.append({"hash": audio_hash, "file_path": file_path, "source": f"telegram:{message.id}"})
        return podcasts

    def fetch_farspod_podcasts(self):
        """دریافت لیست پادکست‌ها از FarsPod"""
        response = requests.get("https://farspod.com/api/latest-podcasts")
        if response.status_code != 200:
            raise ValueError("⚠ خطا در دریافت اطلاعات از FarsPod")
        return response.json().get("podcasts", [])

    async def process_and_publish(self, channel):
        """دریافت پادکست‌ها، بررسی هش، دانلود و ارسال به Kafka"""
        podcasts = await self.fetch_telegram_podcasts(channel)
        farspod_podcasts = self.fetch_farspod_podcasts()

        for podcast in farspod_podcasts:
            podcast_hash = self.calculate_audio_hash(podcast["url"].encode())
            if await self.hash_cache.get_file_hash(podcast_hash):
                print(f"⚠ پادکست تکراری شناسایی شد: {podcast['title']}")
                continue

            await self.hash_cache.store_file_hash(podcast_hash)
            self.kafka_service.send_message(self.kafka_topic, json.dumps(podcast, ensure_ascii=False))

        for podcast in podcasts:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(podcast, ensure_ascii=False))

        return len(podcasts) + len(farspod_podcasts)


if __name__ == "__main__":
    kafka_topic = "podcast_data"
    api_id = "your_api_id"
    api_hash = "your_api_hash"
    session_name = "telegram_scraper"
    download_path = "./downloads"
    collector = PodcastCollector(kafka_topic, api_id, api_hash, session_name, download_path)

    test_channel = "@example_podcast_channel"

    try:
        loop = asyncio.get_event_loop()
        podcast_count = loop.run_until_complete(collector.process_and_publish(test_channel))
        print(f"✅ {podcast_count} پادکست از تلگرام و FarsPod پردازش و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش پادکست‌ها: {e}")

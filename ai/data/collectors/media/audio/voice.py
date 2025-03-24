import json
import hashlib
import asyncio
from telethon import TelegramClient
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.file_management.service.file_service import FileService
from infrastructure.file_management.cache.hash_cache import HashCache


class VoiceCollector:
    """
    جمع‌آوری ویس‌های فارسی از تلگرام و ارسال اطلاعات آن‌ها به Kafka
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

    async def fetch_telegram_voices(self, channel):
        """دریافت ویس‌های فارسی از تلگرام"""
        voices = []
        async with self.client:
            async for message in self.client.iter_messages(channel, filter='voice'):
                audio_data = await message.download_media(bytes)
                audio_hash = self.calculate_audio_hash(audio_data)

                if await self.hash_cache.get_file_hash(audio_hash):
                    print(f"⚠ ویس تکراری شناسایی شد: {message.id}")
                    continue

                await self.hash_cache.store_file_hash(audio_hash)
                file_path = f"{self.download_path}/{audio_hash}.ogg"
                with open(file_path, "wb") as audio_file:
                    audio_file.write(audio_data)

                voices.append({"hash": audio_hash, "file_path": file_path, "source": f"telegram:{message.id}"})
        return voices

    async def process_and_publish(self, channel):
        """دریافت ویس‌ها، بررسی هش، دانلود و ارسال به Kafka"""
        voices = await self.fetch_telegram_voices(channel)

        for voice in voices:
            self.kafka_service.send_message(self.kafka_topic, json.dumps(voice, ensure_ascii=False))

        return len(voices)


if __name__ == "__main__":
    kafka_topic = "voice_data"
    api_id = "your_api_id"
    api_hash = "your_api_hash"
    session_name = "telegram_scraper"
    download_path = "./downloads"
    collector = VoiceCollector(kafka_topic, api_id, api_hash, session_name, download_path)

    test_channel = "@example_voice_channel"

    try:
        loop = asyncio.get_event_loop()
        voice_count = loop.run_until_complete(collector.process_and_publish(test_channel))
        print(f"✅ {voice_count} ویس از تلگرام پردازش و به Kafka ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در پردازش ویس‌ها: {e}")

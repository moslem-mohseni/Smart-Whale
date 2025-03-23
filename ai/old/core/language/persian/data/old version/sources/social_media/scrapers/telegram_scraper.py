import asyncio
import logging
from telethon import TelegramClient, events
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.kafka.service.topic_manager import TopicManager
from infrastructure.clickhouse.service.analytics_service import AnalyticsService

# اطلاعات احراز هویت تلگرام (باید مقداردهی شوند)
API_ID = "your_api_id"
API_HASH = "your_api_hash"
SESSION_NAME = "telegram_scraper"

# سرویس‌های `Redis`، `Kafka` و `ClickHouse`
cache_service = CacheService()
kafka_service = KafkaService()
topic_manager = TopicManager()
analytics_service = AnalyticsService()

# ایجاد `Topic` در `Kafka` در صورت نیاز
KafkaService.create_topic("telegram_messages")

# تنظیمات لاگ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("telegram_scraper_debug.log", encoding="utf-8"), logging.StreamHandler()]
)

class TelegramScraper:
    """اسکرپر پیام‌های تلگرام"""

    def __init__(self):
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        self.messages = set()
        self.load_cached_messages()

    def load_cached_messages(self):
        """بارگذاری پیام‌های کش شده از `Redis`"""
        cached_messages = cache_service.get("telegram_messages") or []
        self.messages = set(cached_messages)
        logging.info(f"🔹 {len(self.messages)} پیام از `Redis` بارگذاری شد.")

    async def start(self):
        """اتصال به تلگرام و گوش دادن به پیام‌های جدید"""
        await self.client.start()
        logging.info("✅ اتصال به تلگرام برقرار شد.")

        @self.client.on(events.NewMessage)
        async def handler(event):
            message_text = event.message.message.strip()
            if message_text and message_text not in self.messages:
                self.messages.add(message_text)
                cache_service.set("telegram_messages", list(self.messages))  # ذخیره در `Redis`
                kafka_service.send_message("telegram_messages", message_text)  # ارسال به `Kafka`
                analytics_service.insert_row("telegram_data", {"message": message_text, "created_at": "NOW()"})  # ذخیره در `ClickHouse`
                logging.info(f"✅ پیام جدید ذخیره و پردازش شد: {message_text[:50]}...")

        logging.info("🔍 در حال گوش دادن به پیام‌های جدید...")
        await self.client.run_until_disconnected()


if __name__ == "__main__":
    scraper = TelegramScraper()
    asyncio.run(scraper.start())

import logging
from typing import Callable, Awaitable
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.kafka.domain.models import Message

class KafkaConsumer:
    """
    این کلاس مصرف‌کننده پیام‌های Kafka را مدیریت کرده و پیام‌ها را پردازش می‌کند.
    """

    def __init__(self, kafka_service: KafkaService):
        self.kafka_service = kafka_service
        logging.info("✅ KafkaConsumer مقداردهی شد و ارتباط با KafkaService برقرار شد.")

    async def subscribe(self, topic: str, group_id: str, handler: Callable[[Message], Awaitable[None]]):
        """
        اشتراک در یک موضوع Kafka برای دریافت و پردازش پیام‌ها.

        :param topic: نام موضوع Kafka.
        :param group_id: شناسه‌ی گروه مصرف‌کننده.
        :param handler: تابع پردازشگر پیام.
        """
        try:
            await self.kafka_service.subscribe(topic, group_id, handler)
            logging.info(f"📥 اشتراک در موضوع Kafka انجام شد. [Topic: {topic}, Group ID: {group_id}]")
        except Exception as e:
            logging.error(f"❌ خطا در اشتراک در موضوع Kafka: {e}")

    async def stop_all(self):
        """
        متوقف کردن تمام مصرف‌کننده‌ها.
        """
        try:
            await self.kafka_service.stop_all()
            logging.info("🛑 تمامی مصرف‌کننده‌های Kafka متوقف شدند.")
        except Exception as e:
            logging.error(f"❌ خطا در توقف مصرف‌کننده‌های Kafka: {e}")

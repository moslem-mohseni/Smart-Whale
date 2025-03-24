import logging
from infrastructure.kafka.service.kafka_service import KafkaService
from infrastructure.kafka.domain.models import Message


class KafkaProducer:
    """
    این کلاس تولیدکننده پیام Kafka برای پردازش داده‌های زبانی را مدیریت می‌کند.
    """

    def __init__(self, kafka_service: KafkaService):
        self.kafka_service = kafka_service
        logging.info("✅ KafkaProducer مقداردهی شد و ارتباط با KafkaService برقرار شد.")

    async def send_message(self, topic: str, content: str, metadata: dict = None):
        """
        ارسال پیام به Kafka.

        :param topic: نام موضوع Kafka که پیام به آن ارسال می‌شود.
        :param content: محتوای پیام.
        :param metadata: اطلاعات اضافی پیام (اختیاری).
        """
        try:
            message = Message(topic=topic, content=content, metadata=metadata or {})
            await self.kafka_service.send_message(message)
            logging.info(f"📤 پیام به Kafka ارسال شد. [Topic: {topic}]")
        except Exception as e:
            logging.error(f"❌ خطا در ارسال پیام به Kafka: {e}")

    async def send_batch_messages(self, topic: str, messages: list):
        """
        ارسال دسته‌ای پیام‌ها به Kafka.

        :param topic: نام موضوع Kafka که پیام‌ها به آن ارسال می‌شوند.
        :param messages: لیستی از پیام‌ها.
        """
        try:
            batch_messages = [Message(topic=topic, content=msg) for msg in messages]
            await self.kafka_service.send_messages_batch(batch_messages)
            logging.info(f"📤 {len(messages)} پیام به‌صورت دسته‌ای به Kafka ارسال شد. [Topic: {topic}]")
        except Exception as e:
            logging.error(f"❌ خطا در ارسال دسته‌ای پیام‌ها به Kafka: {e}")

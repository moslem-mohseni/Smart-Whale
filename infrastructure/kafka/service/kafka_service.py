# infrastructure/kafka/service/kafka_service.py
from ..config.settings import KafkaConfig
from ..adapters.producer import MessageProducer
from ..adapters.consumer import MessageConsumer
from ..domain.models import Message, TopicConfig
from typing import Callable, Awaitable
import logging

logger = logging.getLogger(__name__)


class KafkaService:
    """
    سرویس مدیریت ارتباط با کافکا

    این کلاس یک نقطه مرکزی برای تعامل با کافکا فراهم می‌کند و
    مدیریت تولیدکننده و مصرف‌کننده پیام را بر عهده دارد.
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self._producer = None
        self._consumers = {}

    @property
    def producer(self) -> MessageProducer:
        """دریافت نمونه تولیدکننده"""
        if not self._producer:
            self._producer = MessageProducer(self.config)
        return self._producer

    def get_consumer(self, group_id: str) -> MessageConsumer:
        """
        دریافت یا ایجاد یک مصرف‌کننده

        Args:
            group_id: شناسه گروه مصرف‌کننده

        Returns:
            نمونه‌ای از مصرف‌کننده پیام
        """
        if group_id not in self._consumers:
            consumer_config = KafkaConfig(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=f"{self.config.client_id}-{group_id}",
                group_id=group_id
            )
            self._consumers[group_id] = MessageConsumer(consumer_config)
        return self._consumers[group_id]

    async def send_message(self, message: Message) -> None:
        """
        ارسال یک پیام به Kafka

        Args:
            message: شیء پیام شامل اطلاعات موضوع و محتوا

        Raises:
            ValueError: اگر موضوع پیام خالی باشد یا محتوای پیام None باشد
            ConnectionError: اگر تولیدکننده مقداردهی نشده باشد
        """
        # بررسی اعتبار پیام
        if not message.topic:
            raise ValueError("Topic cannot be empty.")
        if not message.content:
            raise ValueError("Message content cannot be None.")

        # بررسی مقداردهی تولیدکننده
        if not self._producer:
            raise ConnectionError("Kafka producer is not initialized.")

        # ارسال پیام
        try:
            await self.producer.send(message)
            logger.info(f"Message sent to topic {message.topic}: {message.content}")
        except Exception as e:
            logger.error(f"Failed to send message to topic {message.topic}: {e}")
            raise

    async def send_messages(self, messages):
        """ارسال دسته‌ای پیام‌ها"""
        for message in messages:
            await self.send_message(message)

    async def subscribe(self, topic: str, group_id: str,
                        handler: Callable[[Message], Awaitable[None]]) -> None:
        """اشتراک در یک موضوع"""
        consumer = await self.get_consumer(group_id)
        self._consumers[group_id] = consumer  # ذخیره مصرف‌کننده در دیکشنری
        await consumer.subscribe(topic, handler)

    async def stop_all(self):
        """توقف تمام مصرف‌کننده‌ها"""
        for consumer in self._consumers.values():
            await consumer.stop()

    async def shutdown(self):
        """پاک‌سازی منابع و قطع اتصال"""
        if self._producer:
            await self._producer.stop()

    async def create_topic(self, topic_config):
        """ایجاد یک موضوع جدید در Kafka"""
        if not self._producer:
            raise ConnectionError("Kafka producer is not initialized.")
        await self._producer.create_topic(topic_config)

    async def stop_consumer(self, group_id: str) -> None:
        """
        متوقف کردن یک مصرف‌کننده خاص

        Args:
            group_id: شناسه گروه مصرف‌کننده که باید متوقف شود

        Raises:
            ValueError: اگر مصرف‌کننده‌ای با این group_id وجود نداشته باشد
        """
        if group_id not in self._consumers:
            raise ValueError(f"Consumer with group_id '{group_id}' does not exist.")

        consumer = self._consumers[group_id]
        await consumer.stop()  # فراخوانی متد stop روی مصرف‌کننده
        del self._consumers[group_id]  # حذف مصرف‌کننده از دیکشنری



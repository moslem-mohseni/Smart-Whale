# infrastructure/kafka/adapters/consumer.py
from typing import Callable, Awaitable, Any
from ..domain.models import Message
from ..config.settings import KafkaConfig
from confluent_kafka import Consumer as KafkaConsumer
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MessageConsumer:
    """
    کلاس مصرف‌کننده پیام

    این کلاس مسئولیت دریافت و پردازش پیام‌های کافکا را بر عهده دارد.
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self._consumer = None
        self._running = False

    def _create_consumer(self):
        """ایجاد یک نمونه از مصرف‌کننده کافکا"""
        if not self._consumer:
            self._consumer = KafkaConsumer(self.config.get_consumer_config())

    async def subscribe(self, topic: str, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        اشتراک در یک موضوع و پردازش پیام‌ها

        Args:
            topic: نام موضوع
            handler: تابعی که برای پردازش هر پیام فراخوانی می‌شود
        """
        self._create_consumer()
        self._consumer.subscribe([topic])
        self._running = True

        try:
            while self._running:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

                try:
                    # تبدیل پیام دریافتی به مدل Message
                    value = json.loads(msg.value().decode('utf-8'))
                    message = Message(
                        topic=msg.topic(),
                        content=value['content'],
                        timestamp=datetime.fromisoformat(value['timestamp']),
                        metadata=value.get('metadata')
                    )

                    # فراخوانی handler برای پردازش پیام
                    await handler(message)

                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")

        finally:
            self._consumer.close()

    async def stop(self):
        """توقف دریافت پیام‌ها"""
        self._running = False
        if self._consumer:
            self._consumer.close()
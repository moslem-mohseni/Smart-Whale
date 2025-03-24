import asyncio
import json
import logging
from typing import Callable, Awaitable
from ..config.settings import KafkaConfig
from ..domain.models import Message
from ..adapters.connection_pool import KafkaConnectionPool
from ..adapters.retry_mechanism import RetryMechanism
from ..adapters.circuit_breaker import CircuitBreaker
from ..adapters.backpressure import BackpressureHandler
from confluent_kafka import Consumer as KafkaConsumer

logger = logging.getLogger(__name__)


class MessageConsumer:
    """
    مصرف‌کننده پیام Kafka با مکانیزم‌های بهینه‌سازی شده
    """

    def __init__(self, config: KafkaConfig):
        """
        مقداردهی اولیه مصرف‌کننده Kafka

        :param config: تنظیمات Kafka
        """
        self.config = config
        self.pool = KafkaConnectionPool(config)
        self.retry_mechanism = RetryMechanism()
        self.circuit_breaker = CircuitBreaker()
        self.backpressure = BackpressureHandler()

    async def consume(self, topic: str, group_id: str, handler: Callable[[Message], Awaitable[None]]):
        """
        مصرف پیام‌های Kafka از یک `topic` مشخص

        :param topic: نام `topic`
        :param group_id: `group.id` مربوط به مصرف‌کننده
        :param handler: تابعی که پیام را پردازش می‌کند
        """
        consumer = await self.pool.get_consumer(group_id)
        consumer.subscribe([topic])

        try:
            while True:
                async with self.backpressure.semaphore:
                    msg = consumer.poll(timeout=1.0)

                    if msg is None:
                        continue

                    if msg.error():
                        logger.error(f"Consumer error: {msg.error()}")
                        continue

                    try:
                        # تبدیل پیام دریافتی به مدل `Message`
                        value = json.loads(msg.value().decode("utf-8"))
                        message = Message(
                            topic=msg.topic(),
                            content=value.get("content"),
                            metadata=value.get("metadata")
                        )

                        # استفاده از `Circuit Breaker` و `Retry Mechanism`
                        await self.circuit_breaker.execute(
                            self.retry_mechanism.execute, handler, message
                        )

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        finally:
            await self.pool.release_consumer(consumer)

    async def stop(self):
        """توقف مصرف‌کننده Kafka"""
        await self.pool.close_all()

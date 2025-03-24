import asyncio
import json
import logging
from ..config.settings import KafkaConfig
from ..domain.models import Message
from ..adapters.connection_pool import KafkaConnectionPool
from ..adapters.retry_mechanism import RetryMechanism
from ..adapters.circuit_breaker import CircuitBreaker
from ..service.message_cache import MessageCache
from confluent_kafka import Producer as KafkaProducer

logger = logging.getLogger(__name__)


class MessageProducer:
    """
    تولیدکننده پیام Kafka با مکانیزم‌های بهینه‌سازی شده
    """

    def __init__(self, config: KafkaConfig):
        """
        مقداردهی اولیه تولیدکننده Kafka

        :param config: تنظیمات Kafka
        """
        self.config = config
        self.pool = KafkaConnectionPool(config)
        self.retry_mechanism = RetryMechanism()
        self.circuit_breaker = CircuitBreaker()
        self.cache = MessageCache()

    async def send(self, message: Message) -> None:
        """
        ارسال پیام به Kafka با مدیریت `Retry`, `Circuit Breaker`, و `Cache`

        :param message: شیء پیام Kafka
        """
        if await self.cache.is_duplicate(message.content):
            logger.info(f"Duplicate message detected, skipping: {message.content}")
            return

        producer = await self.pool.get_producer()
        try:
            payload = json.dumps({
                "content": message.content,
                "metadata": message.metadata
            }).encode("utf-8")

            await self.circuit_breaker.execute(
                self.retry_mechanism.execute,
                producer.produce,
                topic=message.topic,
                value=payload
            )
            producer.flush()
            logger.info(f"Message sent to {message.topic}")

        except Exception as e:
            logger.error(f"Failed to send message: {e}")
        finally:
            await self.pool.release_producer(producer)

    async def send_batch(self, messages: list[Message]) -> None:
        """
        ارسال دسته‌ای پیام‌ها به Kafka

        :param messages: لیستی از پیام‌های Kafka
        """
        filtered_messages = [msg for msg in messages if not await self.cache.is_duplicate(msg.content)]

        if not filtered_messages:
            logger.info("No new messages to send.")
            return

        producer = await self.pool.get_producer()
        try:
            for message in filtered_messages:
                payload = json.dumps({
                    "content": message.content,
                    "metadata": message.metadata
                }).encode("utf-8")

                producer.produce(topic=message.topic, value=payload)

            producer.flush()
            logger.info(f"Batch of {len(filtered_messages)} messages sent successfully.")

        except Exception as e:
            logger.error(f"Failed to send batch messages: {e}")
        finally:
            await self.pool.release_producer(producer)

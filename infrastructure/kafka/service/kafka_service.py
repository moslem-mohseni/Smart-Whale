import asyncio
import logging
from typing import Callable, Awaitable, List
from ..config.settings import KafkaConfig
from ..adapters.connection_pool import KafkaConnectionPool
from ..adapters.circuit_breaker import CircuitBreaker
from ..adapters.retry_mechanism import RetryMechanism
from ..adapters.backpressure import BackpressureHandler
from ..service.batch_processor import BatchProcessor
from ..service.message_cache import MessageCache
from ..service.partition_manager import PartitionManager
from ..domain.models import Message

logger = logging.getLogger(__name__)


class KafkaService:
    """
    مدیریت تعامل با Kafka با استفاده از ماژول‌های بهینه‌سازی شده
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.pool = KafkaConnectionPool(config)
        self.circuit_breaker = CircuitBreaker()
        self.retry_mechanism = RetryMechanism()
        self.backpressure = BackpressureHandler()
        self.batch_processor = BatchProcessor(self.pool)
        self.message_cache = MessageCache()
        self.partition_manager = PartitionManager(config)

    async def send_message(self, message: Message) -> None:
        """
        ارسال یک پیام به Kafka با مکانیزم‌های بهینه‌سازی
        """
        if await self.message_cache.is_duplicate(message.content):
            logger.info("Duplicate message detected, skipping send.")
            return

        producer = await self.pool.get_producer()
        try:
            await self.circuit_breaker.execute(
                self.retry_mechanism.execute,
                producer.produce,
                topic=message.topic,
                value=message.content.encode("utf-8")
            )
            producer.flush()
            logger.info(f"Message sent to {message.topic}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
        finally:
            await self.pool.release_producer(producer)

    async def send_messages_batch(self, messages: List[Message]) -> None:
        """
        ارسال دسته‌ای پیام‌ها به Kafka
        """
        for message in messages:
            if await self.message_cache.is_duplicate(message.content):
                messages.remove(message)

        await self.batch_processor.add_messages(messages)

    async def subscribe(self, topic: str, group_id: str, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        اشتراک در یک `topic` با مدیریت Backpressure
        """
        consumer = await self.pool.get_consumer(group_id)
        consumer.subscribe([topic])

        try:
            while True:
                async with self.backpressure.semaphore:
                    msg = consumer.poll(1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        logger.error(f"Consumer error: {msg.error()}")
                        continue

                    await handler(Message(topic=msg.topic(), content=msg.value().decode("utf-8")))
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}")
        finally:
            await self.pool.release_consumer(consumer)

    async def manage_partitions(self, topic: str, new_partition_count: int) -> None:
        """
        افزایش تعداد پارتیشن‌ها برای `topic` مشخص‌شده
        """
        success = self.partition_manager.increase_partitions(topic, new_partition_count)
        if success:
            logger.info(f"Partitions for {topic} increased to {new_partition_count}")
        else:
            logger.error(f"Failed to increase partitions for {topic}")

    async def shutdown(self):
        """
        پاک‌سازی و بستن تمام اتصالات Kafka
        """
        await self.pool.close_all()
        await self.message_cache.close()

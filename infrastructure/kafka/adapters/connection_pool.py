import asyncio
from confluent_kafka import Producer, Consumer
from ..config.settings import KafkaConfig
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KafkaConnectionPool:
    """
    مدیریت Connection Pool برای Kafka (تولیدکننده و مصرف‌کننده)
    """

    def __init__(self, config: KafkaConfig, min_size: int = 2, max_size: int = 10, ttl: int = 300):
        """
        مقداردهی اولیه Pool

        :param config: تنظیمات Kafka
        :param min_size: حداقل تعداد اتصالات فعال
        :param max_size: حداکثر تعداد اتصالات
        :param ttl: مدت زمان زنده ماندن هر اتصال (ثانیه)
        """
        self.config = config
        self.min_size = min_size
        self.max_size = max_size
        self.ttl = ttl

        self._producer_pool = asyncio.Queue(maxsize=max_size)
        self._consumer_pool = asyncio.Queue(maxsize=max_size)

        self._active_producers = set()
        self._active_consumers = set()

    async def _create_producer(self) -> Producer:
        """ایجاد یک اتصال جدید برای تولیدکننده"""
        if len(self._active_producers) >= self.max_size:
            logger.warning("Producer pool reached max limit.")
            return await self._producer_pool.get()

        producer = Producer(self.config.get_producer_config())
        self._active_producers.add(producer)
        return producer

    async def _create_consumer(self, group_id: str) -> Consumer:
        """ایجاد یک اتصال جدید برای مصرف‌کننده"""
        if len(self._active_consumers) >= self.max_size:
            logger.warning("Consumer pool reached max limit.")
            return await self._consumer_pool.get()

        consumer_config = self.config.get_consumer_config()
        consumer_config['group.id'] = group_id  # مصرف‌کننده نیاز به group_id دارد

        consumer = Consumer(consumer_config)
        self._active_consumers.add(consumer)
        return consumer

    async def get_producer(self) -> Producer:
        """دریافت یک تولیدکننده از Pool"""
        if self._producer_pool.qsize() > 0:
            return await self._producer_pool.get()
        return await self._create_producer()

    async def get_consumer(self, group_id: str) -> Consumer:
        """دریافت یک مصرف‌کننده از Pool"""
        if self._consumer_pool.qsize() > 0:
            return await self._consumer_pool.get()
        return await self._create_consumer(group_id)

    async def release_producer(self, producer: Producer):
        """بازگرداندن تولیدکننده به Pool"""
        if len(self._active_producers) > self.min_size:
            self._active_producers.remove(producer)
        else:
            await self._producer_pool.put(producer)

    async def release_consumer(self, consumer: Consumer):
        """بازگرداندن مصرف‌کننده به Pool"""
        if len(self._active_consumers) > self.min_size:
            self._active_consumers.remove(consumer)
        else:
            await self._consumer_pool.put(consumer)

    async def close_all(self):
        """بستن تمام اتصالات فعال"""
        for producer in self._active_producers:
            producer.flush()
        self._active_producers.clear()

        for consumer in self._active_consumers:
            consumer.close()
        self._active_consumers.clear()

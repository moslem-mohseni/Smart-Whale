import asyncio
import logging
from typing import List
from ..adapters.connection_pool import KafkaConnectionPool
from ..domain.models import Message

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    پردازش دسته‌ای پیام‌های Kafka
    """

    def __init__(self, pool: KafkaConnectionPool, batch_size: int = 10, batch_timeout: float = 2.0):
        """
        مقداردهی اولیه پردازش دسته‌ای

        :param pool: شیء Connection Pool برای مدیریت ارتباطات Kafka
        :param batch_size: تعداد پیام‌ها در هر دسته
        :param batch_timeout: حداکثر زمان انتظار برای تکمیل یک دسته (ثانیه)
        """
        self.pool = pool
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self._message_queue = asyncio.Queue()

    async def add_message(self, message: Message):
        """
        افزودن پیام جدید به صف پردازش دسته‌ای

        :param message: شیء پیام Kafka
        """
        await self._message_queue.put(message)

    async def add_messages(self, messages: List[Message]):
        """
        افزودن چندین پیام به صف پردازش دسته‌ای

        :param messages: لیستی از پیام‌های Kafka
        """
        for message in messages:
            await self._message_queue.put(message)

    async def _send_batch(self, messages: List[Message]):
        """ارسال دسته‌ای پیام‌ها به Kafka"""
        producer = await self.pool.get_producer()
        try:
            for message in messages:
                producer.produce(
                    topic=message.topic,
                    value=message.content.encode('utf-8')
                )
            producer.flush()
            logger.info(f"Batch of {len(messages)} messages sent successfully.")
        except Exception as e:
            logger.error(f"Failed to send batch messages: {e}")
        finally:
            await self.pool.release_producer(producer)

    async def process_batches(self):
        """
        مدیریت ارسال پیام‌ها به صورت دسته‌ای
        """
        while True:
            batch = []
            try:
                for _ in range(self.batch_size):
                    message = await asyncio.wait_for(self._message_queue.get(), timeout=self.batch_timeout)
                    batch.append(message)
            except asyncio.TimeoutError:
                pass  # زمان انتظار به پایان رسید، ارسال دسته‌ای را آغاز کن

            if batch:
                await self._send_batch(batch)

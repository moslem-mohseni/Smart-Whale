import pytest
import asyncio
from infrastructure.kafka.config import KafkaConfig
from infrastructure.kafka.adapters.connection_pool import KafkaConnectionPool


@pytest.mark.asyncio
async def test_connection_pool():
    config = KafkaConfig()
    pool = KafkaConnectionPool(config)

    producer1 = await pool.get_producer()
    producer2 = await pool.get_producer()

    assert producer1 is not None
    assert producer2 is not None
    assert producer1 != producer2  # اتصال جدید باید دریافت شود

    await pool.release_producer(producer1)
    await pool.release_producer(producer2)

    await pool.close_all()

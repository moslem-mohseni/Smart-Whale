import pytest
import asyncio
from infrastructure.kafka.config import KafkaConfig
from infrastructure.kafka.adapters.producer import MessageProducer
from infrastructure.kafka.domain import Message


@pytest.mark.asyncio
async def test_producer():
    config = KafkaConfig()
    producer = MessageProducer(config)

    message = Message(topic="test_topic", content="Hello Kafka!")
    await producer.send(message)

    # اگر خطایی رخ ندهد، تست موفق است
    assert True

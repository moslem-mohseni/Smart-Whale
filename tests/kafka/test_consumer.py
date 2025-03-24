import pytest
import asyncio
from infrastructure.kafka.config import KafkaConfig
from infrastructure.kafka.adapters.consumer import MessageConsumer
from infrastructure.kafka.domain import Message


@pytest.mark.asyncio
async def test_consumer():
    config = KafkaConfig()
    consumer = MessageConsumer(config)

    async def handler(message: Message):
        assert message.topic == "test_topic"
        assert message.content == "Hello Kafka!"

    await consumer.consume("test_topic", "test_group", handler)

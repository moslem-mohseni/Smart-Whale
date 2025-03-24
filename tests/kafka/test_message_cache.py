import pytest
import asyncio
from infrastructure.kafka.service.message_cache import MessageCache


@pytest.mark.asyncio
async def test_message_cache():
    cache = MessageCache()
    await cache.connect()

    message = "Test Message"

    # اولین بار نباید تکراری باشد
    is_duplicate = await cache.is_duplicate(message)
    assert is_duplicate is False

    # بار دوم باید تکراری باشد
    is_duplicate = await cache.is_duplicate(message)
    assert is_duplicate is True

    await cache.close()

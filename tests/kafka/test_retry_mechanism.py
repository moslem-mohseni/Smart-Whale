import pytest
import asyncio
from infrastructure.kafka.adapters.retry_mechanism import RetryMechanism


@pytest.mark.asyncio
async def test_retry_mechanism():
    retry_mechanism = RetryMechanism(max_retries=3)

    attempt_counter = 0

    async def failing_function():
        nonlocal attempt_counter
        attempt_counter += 1
        raise Exception("Simulated Failure")

    try:
        await retry_mechanism.execute(failing_function)
    except Exception:
        pass

    assert attempt_counter == 3

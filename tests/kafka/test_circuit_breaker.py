import pytest
import asyncio
from infrastructure.kafka.adapters.circuit_breaker import CircuitBreaker


@pytest.mark.asyncio
async def test_circuit_breaker():
    circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=5)

    def failing_function():
        raise Exception("Simulated Failure")

    # سه بار شکست باید باعث باز شدن Circuit شود
    for _ in range(3):
        try:
            circuit_breaker.execute(failing_function)
        except Exception:
            pass

    assert circuit_breaker.state == "OPEN"

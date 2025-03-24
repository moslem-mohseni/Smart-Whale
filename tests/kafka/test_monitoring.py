import pytest
import asyncio
from infrastructure.kafka.monitoring.metrics import KafkaMetrics
from infrastructure.kafka.monitoring.health_check import KafkaHealthCheck
from infrastructure.kafka.config import KafkaConfig


@pytest.mark.asyncio
async def test_kafka_metrics():
    metrics = KafkaMetrics()
    metrics.record_message_sent()
    metrics.record_message_received()

    assert metrics.get_throughput()["messages_sent_per_sec"] >= 0
    assert metrics.get_system_metrics()["cpu_usage"] >= 0


@pytest.mark.asyncio
async def test_kafka_health_check():
    config = KafkaConfig()
    health_check = KafkaHealthCheck(config)
    report = await health_check.get_health_report()

    assert report["kafka_status"] in ["Healthy", "Unhealthy"]

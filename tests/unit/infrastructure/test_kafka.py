import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime
from infrastructure.kafka import (
    KafkaService,
    KafkaConfig,
    Message,
    TopicConfig,
    MessageProducer,
    MessageConsumer
)
from infrastructure.interfaces import ConnectionError, OperationError


@pytest.fixture
def kafka_config():
    """فیکسچر برای ایجاد تنظیمات Kafka"""
    return KafkaConfig(
        bootstrap_servers=["localhost:9092"],
        client_id="test_client",
        group_id="test_group"
    )


@pytest_asyncio.fixture
async def kafka_service(kafka_config):
    """فیکسچر برای ایجاد سرویس Kafka با mock producer و consumer"""
    service = KafkaService(kafka_config)
    service._producer = AsyncMock(spec=MessageProducer)
    service._producer.stop = AsyncMock()  # اضافه کردن متد stop به mock producer
    service._consumers = {}

    # تعریف mock مناسب برای consumer
    mock_consumer = AsyncMock(spec=MessageConsumer)
    mock_consumer.subscribe = AsyncMock()  # شبیه‌سازی subscribe به صورت async
    mock_consumer.stop = AsyncMock()  # شبیه‌سازی stop به صورت async
    service.get_consumer = AsyncMock(return_value=mock_consumer)  # بازگشت mock_consumer

    yield service
    await service.shutdown()




@pytest.mark.asyncio
async def test_kafka_service_initialization(kafka_service, kafka_config):
    """تست مقداردهی اولیه سرویس Kafka"""
    assert kafka_service.config == kafka_config
    assert kafka_service._producer is not None


@pytest.mark.asyncio
async def test_kafka_message_publishing(kafka_service):
    """تست انتشار پیام در Kafka"""
    # ایجاد یک پیام تستی
    test_message = Message(
        topic="test_topic",
        content={"key": "value"},
        timestamp=datetime.now()
    )

    # تست ارسال پیام
    await kafka_service.send_message(test_message)
    kafka_service._producer.send.assert_called_once_with(test_message)


@pytest.mark.asyncio
async def test_kafka_message_subscription(kafka_service):
    """تست اشتراک و دریافت پیام در Kafka"""

    # تعریف یک handler تستی
    async def test_handler(message):
        return message

    # تست اشتراک در یک موضوع
    await kafka_service.subscribe(
        topic="test_topic",
        group_id="test_group",
        handler=test_handler
    )

    # بررسی فراخوانی متد subscribe
    consumer = await kafka_service.get_consumer("test_group")
    consumer.subscribe.assert_called_once_with(
        "test_topic",
        test_handler
    )



@pytest.mark.asyncio
async def test_kafka_topic_management(kafka_service):
    """تست مدیریت موضوعات در Kafka"""
    # تعریف یک موضوع تستی
    test_topic = TopicConfig(
        name="test_topic",
        partitions=3,
        replication_factor=1,
        configs={
            'retention.ms': 86400000  # 24 ساعت
        }
    )

    # Mock کردن عملیات مدیریت موضوع
    kafka_service._producer.create_topic = AsyncMock()

    # تست ایجاد موضوع
    await kafka_service.create_topic(test_topic)
    kafka_service._producer.create_topic.assert_called_once_with(test_topic)


@pytest.mark.asyncio
async def test_kafka_error_handling(kafka_service):
    """تست مدیریت خطاها در Kafka"""
    # شبیه‌سازی خطای ارسال پیام
    kafka_service._producer.send.side_effect = OperationError("Send failed")

    with pytest.raises(OperationError):
        await kafka_service.send_message(Message(
            topic="test_topic",
            content="test_message"
        ))

    # شبیه‌سازی خطای اشتراک
    kafka_service.get_consumer.side_effect = ConnectionError("Consumer creation failed")

    with pytest.raises(ConnectionError):
        await kafka_service.subscribe(
            topic="test_topic",
            group_id="test_group",
            handler=AsyncMock()
        )


@pytest.mark.asyncio
async def test_kafka_message_batch_processing(kafka_service):
    """تست پردازش دسته‌ای پیام‌ها"""
    # ایجاد چند پیام تستی
    messages = [
        Message(topic="test_topic", content=f"message_{i}")
        for i in range(3)
    ]

    # تست ارسال دسته‌ای
    kafka_service._producer.send = AsyncMock()  # Mock send method
    await kafka_service.send_messages(messages)
    assert kafka_service._producer.send.call_count == len(messages)


@pytest.mark.asyncio
async def test_kafka_consumer_group_management(kafka_service):
    """تست مدیریت گروه‌های مصرف‌کننده"""
    group_id = "test_group"

    # Mock کردن consumer
    consumer = AsyncMock(spec=MessageConsumer)
    kafka_service.get_consumer.return_value = consumer

    # تست ایجاد consumer
    await kafka_service.subscribe(
        topic="test_topic",
        group_id=group_id,
        handler=AsyncMock()
    )

    # بررسی ذخیره consumer در دیکشنری
    assert group_id in kafka_service._consumers.keys()

    # تست توقف consumer
    await kafka_service.stop_consumer(group_id)
    consumer.stop.assert_called_once()


@pytest.mark.asyncio
async def test_kafka_message_validation(kafka_service):
    """تست اعتبارسنجی پیام‌ها"""
    # تست پیام با موضوع خالی
    with pytest.raises(ValueError):
        await kafka_service.send_message(Message(
            topic="",
            content="test_message"
        ))

    # تست پیام بدون محتوا
    with pytest.raises(ValueError):
        await kafka_service.send_message(Message(
            topic="test_topic",
            content=None
        ))

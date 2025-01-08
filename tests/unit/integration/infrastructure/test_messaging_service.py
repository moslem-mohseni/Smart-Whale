# tests/integration/infrastructure/test_messaging_service.py

import pytest
import asyncio
import docker
from datetime import datetime
from typing import AsyncGenerator, List
from infrastructure.kafka import (
    KafkaService,
    KafkaConfig,
    Message,
    TopicConfig
)


class KafkaTestEnvironment:
    """
    مدیریت محیط تست Kafka

    این کلاس مسئول راه‌اندازی و مدیریت یک محیط کامل Kafka برای تست است.
    شامل راه‌اندازی ZooKeeper و Kafka broker می‌شود تا محیطی مشابه تولید
    برای اجرای تست‌ها فراهم شود.
    """

    def __init__(self):
        self.client = docker.from_env()
        self.containers = {}
        self.network = None

    async def setup(self) -> dict:
        """راه‌اندازی محیط تست Kafka"""
        # ایجاد شبکه داکر
        self.network = self.client.networks.create("kafka_test_net", driver="bridge")

        # راه‌اندازی ZooKeeper
        zk_container = self.client.containers.run(
            'wurstmeister/zookeeper',
            ports={'2181/tcp': 2181},
            detach=True,
            remove=True,
            name='test_zookeeper',
            network="kafka_test_net"
        )
        self.containers['zookeeper'] = zk_container

        # انتظار برای آماده شدن ZooKeeper
        await asyncio.sleep(5)

        # راه‌اندازی Kafka
        kafka_container = self.client.containers.run(
            'wurstmeister/kafka',
            ports={'9092/tcp': 9092},
            environment={
                'KAFKA_ADVERTISED_HOST_NAME': 'localhost',
                'KAFKA_ZOOKEEPER_CONNECT': 'test_zookeeper:2181',
                'KAFKA_CREATE_TOPICS': 'test_topic:1:1'
            },
            detach=True,
            remove=True,
            name='test_kafka',
            network="kafka_test_net"
        )
        self.containers['kafka'] = kafka_container

        # انتظار برای آماده شدن Kafka
        await asyncio.sleep(10)

        return {
            'bootstrap_servers': ['localhost:9092'],
            'zookeeper': 'localhost:2181'
        }

    async def cleanup(self):
        """پاکسازی و حذف محیط تست"""
        for container in self.containers.values():
            container.stop()
        if self.network:
            self.network.remove()
        self.containers.clear()


@pytest.fixture(scope="session")
async def kafka_environment() -> AsyncGenerator[KafkaTestEnvironment, None]:
    """فیکسچر برای مدیریت محیط تست Kafka"""
    env = KafkaTestEnvironment()
    await env.setup()
    yield env
    await env.cleanup()


@pytest.fixture(scope="session")
async def kafka_config(kafka_environment) -> KafkaConfig:
    """فیکسچر برای تنظیمات Kafka"""
    params = await kafka_environment.setup()
    return KafkaConfig(
        bootstrap_servers=params['bootstrap_servers'],
        client_id='test_client',
        group_id='test_group'
    )


@pytest.fixture
async def kafka_service(kafka_config) -> AsyncGenerator[KafkaService, None]:
    """فیکسچر برای سرویس پیام‌رسانی"""
    service = KafkaService(kafka_config)
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.mark.integration
async def test_message_publishing_and_consumption(kafka_service):
    """
    تست انتشار و مصرف پیام

    این تست بررسی می‌کند که پیام‌ها به درستی منتشر می‌شوند و
    مصرف‌کننده‌ها می‌توانند آن‌ها را دریافت کنند.
    """
    test_topic = "test_topic"
    test_messages = []
    received_messages = []

    # ایجاد handler برای دریافت پیام‌ها
    async def message_handler(message):
        received_messages.append(message)

    # اشتراک در موضوع
    await kafka_service.subscribe(test_topic, "test_consumer", message_handler)

    # ارسال چند پیام تست
    for i in range(5):
        message = Message(
            topic=test_topic,
            content=f"test_message_{i}",
            timestamp=datetime.now()
        )
        test_messages.append(message)
        await kafka_service.send_message(message)

    # انتظار برای دریافت پیام‌ها
    await asyncio.sleep(2)

    # بررسی دریافت تمام پیام‌ها
    assert len(received_messages) == len(test_messages)
    for sent, received in zip(test_messages, received_messages):
        assert sent.content == received.content


@pytest.mark.integration
async def test_multiple_consumers(kafka_service):
    """
    تست چندین مصرف‌کننده

    بررسی می‌کند که چندین مصرف‌کننده می‌توانند به طور همزمان
    از یک موضوع مشترک پیام دریافت کنند.
    """
    test_topic = "multi_consumer_test"
    message_count = 10
    consumer_count = 3
    received_messages: List[List] = [[] for _ in range(consumer_count)]

    # ایجاد چندین مصرف‌کننده
    async def create_consumer(consumer_id):
        async def handler(message):
            received_messages[consumer_id].append(message)

        await kafka_service.subscribe(
            test_topic,
            f"consumer_group_{consumer_id}",
            handler
        )

    # راه‌اندازی مصرف‌کننده‌ها
    for i in range(consumer_count):
        await create_consumer(i)

    # ارسال پیام‌ها
    for i in range(message_count):
        await kafka_service.send_message(Message(
            topic=test_topic,
            content=f"message_{i}"
        ))

    # انتظار برای دریافت پیام‌ها
    await asyncio.sleep(3)

    # بررسی دریافت پیام‌ها توسط همه مصرف‌کننده‌ها
    for consumer_messages in received_messages:
        assert len(consumer_messages) > 0


@pytest.mark.integration
async def test_message_ordering(kafka_service):
    """
    تست ترتیب پیام‌ها

    بررسی می‌کند که پیام‌ها در همان ترتیبی که ارسال شده‌اند
    دریافت می‌شوند.
    """
    test_topic = "ordering_test"
    message_count = 100
    received_messages = []

    async def order_handler(message):
        received_messages.append(message)

    await kafka_service.subscribe(test_topic, "order_consumer", order_handler)

    # ارسال پیام‌های شماره‌گذاری شده
    sent_messages = [
        Message(topic=test_topic, content=f"order_{i}")
        for i in range(message_count)
    ]

    for message in sent_messages:
        await kafka_service.send_message(message)

    # انتظار برای دریافت پیام‌ها
    await asyncio.sleep(3)

    # بررسی ترتیب پیام‌ها
    assert len(received_messages) == len(sent_messages)
    for i, (sent, received) in enumerate(zip(sent_messages, received_messages)):
        assert sent.content == received.content


@pytest.mark.integration
async def test_topic_management(kafka_service):
    """
    تست مدیریت موضوعات

    بررسی می‌کند که سیستم می‌تواند موضوعات جدید ایجاد کند و
    تنظیمات آن‌ها را مدیریت کند.
    """
    # تعریف یک موضوع جدید
    topic_config = TopicConfig(
        name="managed_topic",
        partitions=3,
        replication_factor=1,
        configs={
            'cleanup.policy': 'delete',
            'retention.ms': '86400000'  # 24 ساعت
        }
    )

    # ایجاد موضوع
    await kafka_service.create_topic(topic_config)

    # بررسی امکان ارسال و دریافت پیام در موضوع جدید
    test_message = Message(
        topic=topic_config.name,
        content="test_content"
    )

    received = None

    async def test_handler(message):
        nonlocal received
        received = message

    await kafka_service.subscribe(topic_config.name, "test_group", test_handler)
    await kafka_service.send_message(test_message)

    # انتظار برای دریافت پیام
    await asyncio.sleep(2)
    assert received is not None
    assert received.content == test_message.content
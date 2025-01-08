# infrastructure/kafka/migrations/001_create_base_topics.py

"""
این فایل مسئول ایجاد موضوعات پایه در کافکا است.
برخلاف دیتابیس‌های رابطه‌ای، در کافکا migration به معنای تغییر ساختار داده نیست،
بلکه بیشتر برای مدیریت موضوعات، تنظیمات و سیاست‌های نگهداری داده استفاده می‌شود.
"""

from ..domain.models import TopicConfig
from ..service.kafka_service import KafkaService
from confluent_kafka.admin import AdminClient, NewTopic
import logging

logger = logging.getLogger(__name__)


async def upgrade(kafka_service: KafkaService):
    """
    ایجاد موضوعات پایه مورد نیاز سیستم

    این تابع موضوعات اصلی که سیستم به آنها نیاز دارد را ایجاد می‌کند.
    هر موضوع با تنظیمات خاص خود (مثل تعداد پارتیشن و فاکتور تکرار) تعریف می‌شود.
    """
    base_topics = [
        TopicConfig(
            name="system.events",
            partitions=3,
            replication_factor=2,
            configs={
                'retention.ms': 604800000,  # 7 روز
                'cleanup.policy': 'delete'
            }
        ),
        TopicConfig(
            name="user.activities",
            partitions=5,
            replication_factor=2,
            configs={
                'retention.ms': 2592000000,  # 30 روز
                'cleanup.policy': 'delete'
            }
        ),
        TopicConfig(
            name="analytics.data",
            partitions=10,
            replication_factor=2,
            configs={
                'retention.ms': 7776000000,  # 90 روز
                'cleanup.policy': 'compact'
            }
        )
    ]

    admin_client = AdminClient({'bootstrap.servers': ','.join(kafka_service.config.bootstrap_servers)})

    new_topics = [
        NewTopic(
            topic.name,
            num_partitions=topic.partitions,
            replication_factor=topic.replication_factor,
            config=topic.configs
        )
        for topic in base_topics
    ]

    try:
        futures = admin_client.create_topics(new_topics)
        for topic, future in futures.items():
            try:
                future.result()
                logger.info(f"Topic {topic} created successfully")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Topic {topic} already exists")
                else:
                    logger.error(f"Failed to create topic {topic}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in topic creation: {str(e)}")
        raise


async def downgrade(kafka_service: KafkaService):
    """
    حذف موضوعات ایجاد شده

    این تابع در صورت نیاز به بازگشت تغییرات، موضوعات ایجاد شده را حذف می‌کند.
    البته باید توجه داشت که حذف موضوعات در محیط تولید باید با احتیاط انجام شود.
    """
    topics_to_delete = ["system.events", "user.activities", "analytics.data"]

    admin_client = AdminClient({'bootstrap.servers': ','.join(kafka_service.config.bootstrap_servers)})

    try:
        futures = admin_client.delete_topics(topics_to_delete)
        for topic, future in futures.items():
            try:
                future.result()
                logger.info(f"Topic {topic} deleted successfully")
            except Exception as e:
                logger.error(f"Failed to delete topic {topic}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in topic deletion: {str(e)}")
        raise
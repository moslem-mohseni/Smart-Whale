from confluent_kafka.admin import AdminClient, NewTopic
from ..config.settings import KafkaConfig
import logging

logger = logging.getLogger(__name__)


class BaseTopicMigration:
    """
    ایجاد `topics` ضروری در Kafka هنگام راه‌اندازی اولیه سیستم
    """

    def __init__(self, config: KafkaConfig):
        """
        مقداردهی اولیه کلاس

        :param config: تنظیمات Kafka
        """
        self.config = config
        self.admin_client = AdminClient({"bootstrap.servers": ",".join(self.config.bootstrap_servers)})

    def create_topics(self):
        """
        ایجاد `topics` پایه‌ای در Kafka
        """
        base_topics = [
            {"name": "system.events", "partitions": 3, "replication_factor": 2, "retention_ms": 604800000},  # 7 روز
            {"name": "user.activities", "partitions": 5, "replication_factor": 2, "retention_ms": 2592000000},  # 30 روز
            {"name": "analytics.data", "partitions": 10, "replication_factor": 2, "retention_ms": 7776000000},  # 90 روز
        ]

        existing_topics = self.admin_client.list_topics(timeout=5).topics.keys()

        topics_to_create = [
            NewTopic(topic["name"], num_partitions=topic["partitions"], replication_factor=topic["replication_factor"])
            for topic in base_topics if topic["name"] not in existing_topics
        ]

        if not topics_to_create:
            logger.info("All base topics already exist.")
            return

        try:
            future = self.admin_client.create_topics(topics_to_create)

            for topic, f in future.items():
                try:
                    f.result()  # مسدود شدن تا تکمیل ایجاد `topic`
                    logger.info(f"Topic {topic} created successfully.")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.info(f"Topic {topic} already exists.")
                    else:
                        logger.error(f"Failed to create topic {topic}: {e}")

        except Exception as e:
            logger.error(f"Error while creating base topics: {e}")


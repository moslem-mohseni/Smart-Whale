from confluent_kafka.admin import AdminClient, NewTopic
from ..config.settings import KafkaConfig
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class TopicManager:
    """
    مدیریت `topics` در Kafka
    """

    def __init__(self, config: KafkaConfig):
        """
        مقداردهی اولیه مدیر `topics`

        :param config: تنظیمات Kafka
        """
        self.config = config
        self.admin_client = AdminClient({"bootstrap.servers": ",".join(self.config.bootstrap_servers)})

    def create_topic(self, topic_name: str, num_partitions: int = 3, replication_factor: int = 1) -> bool:
        """
        ایجاد `topic` جدید در Kafka

        :param topic_name: نام `topic`
        :param num_partitions: تعداد پارتیشن‌ها
        :param replication_factor: تعداد `replicas`
        :return: `True` در صورت موفقیت، `False` در غیر این صورت
        """
        try:
            topic_list = [NewTopic(topic_name, num_partitions, replication_factor)]
            future = self.admin_client.create_topics(topic_list)

            for topic, f in future.items():
                try:
                    f.result()  # مسدود شدن تا تکمیل ایجاد `topic`
                    logger.info(f"Topic {topic} created successfully.")
                    return True
                except Exception as e:
                    logger.error(f"Failed to create topic {topic}: {e}")
                    return False

        except Exception as e:
            logger.error(f"Error while creating topic {topic_name}: {e}")
            return False

    def delete_topic(self, topic_name: str) -> bool:
        """
        حذف `topic` از Kafka

        :param topic_name: نام `topic`
        :return: `True` در صورت موفقیت، `False` در غیر این صورت
        """
        try:
            future = self.admin_client.delete_topics([topic_name])

            for topic, f in future.items():
                try:
                    f.result()  # مسدود شدن تا تکمیل حذف `topic`
                    logger.info(f"Topic {topic} deleted successfully.")
                    return True
                except Exception as e:
                    logger.error(f"Failed to delete topic {topic}: {e}")
                    return False

        except Exception as e:
            logger.error(f"Error while deleting topic {topic_name}: {e}")
            return False

    def list_topics(self) -> Dict:
        """
        دریافت لیست `topics` موجود در Kafka

        :return: دیکشنری شامل `topics` موجود و تعداد پارتیشن‌هایشان
        """
        try:
            metadata = self.admin_client.list_topics(timeout=10)
            topics_info = {
                topic: len(metadata.topics[topic].partitions)
                for topic in metadata.topics
            }
            return topics_info

        except Exception as e:
            logger.error(f"Failed to list topics: {e}")
            return {}

from confluent_kafka.admin import AdminClient, NewPartitions
from ..config.settings import KafkaConfig
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PartitionManager:
    """
    مدیریت پارتیشن‌های Kafka
    """

    def __init__(self, config: KafkaConfig):
        """
        مقداردهی اولیه مدیریت پارتیشن‌ها

        :param config: تنظیمات Kafka
        """
        self.config = config
        self.admin_client = AdminClient({"bootstrap.servers": ",".join(self.config.bootstrap_servers)})

    def list_partitions(self, topic_name: str) -> Dict:
        """
        دریافت اطلاعات مربوط به پارتیشن‌های یک `topic`

        :param topic_name: نام `topic`
        :return: اطلاعات پارتیشن‌ها شامل تعداد، leader و replication
        """
        try:
            metadata = self.admin_client.list_topics(timeout=10)
            if topic_name not in metadata.topics:
                logger.warning(f"Topic {topic_name} not found.")
                return {}

            topic_metadata = metadata.topics[topic_name]
            return {
                "partition_count": len(topic_metadata.partitions),
                "partitions": {
                    p_id: {
                        "leader": p.leader,
                        "replicas": p.replicas,
                        "isrs": p.isrs
                    }
                    for p_id, p in topic_metadata.partitions.items()
                }
            }
        except Exception as e:
            logger.error(f"Failed to list partitions for {topic_name}: {e}")
            return {}

    def increase_partitions(self, topic_name: str, new_partition_count: int) -> bool:
        """
        افزایش تعداد پارتیشن‌های یک `topic`

        :param topic_name: نام `topic`
        :param new_partition_count: تعداد کل پارتیشن‌های جدید
        :return: `True` در صورت موفقیت، `False` در غیر این صورت
        """
        try:
            current_metadata = self.list_partitions(topic_name)
            if not current_metadata:
                logger.error(f"Cannot increase partitions: Topic {topic_name} not found.")
                return False

            current_count = current_metadata["partition_count"]
            if new_partition_count <= current_count:
                logger.warning(f"New partition count must be greater than {current_count}.")
                return False

            future = self.admin_client.create_partitions(
                {topic_name: NewPartitions(new_partition_count)}
            )
            future[topic_name].result()  # مسدود شدن تا تکمیل عملیات
            logger.info(f"Successfully increased partitions for {topic_name} to {new_partition_count}.")
            return True

        except Exception as e:
            logger.error(f"Failed to increase partitions for {topic_name}: {e}")
            return False

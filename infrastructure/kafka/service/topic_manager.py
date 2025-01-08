# infrastructure/kafka/scripts/topic_manager.py

"""
این اسکریپت ابزارهایی برای مدیریت موضوعات کافکا فراهم می‌کند.
با استفاده از این ابزارها می‌توانیم موضوعات را بررسی، ایجاد و مدیریت کنیم.
"""

from confluent_kafka.admin import AdminClient, ConfigResource
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# تعریف مقدار ثابت برای RESOURCE_TYPE_TOPIC
RESOURCE_TYPE_TOPIC = 2  # مقدار ثابت مرتبط با موضوعات Kafka

class TopicManager:
    """مدیریت موضوعات کافکا"""

    def __init__(self, bootstrap_servers: List[str]):
        self.admin_client = AdminClient({
            'bootstrap.servers': ','.join(bootstrap_servers)
        })

    async def list_topics(self) -> List[str]:
        """
        دریافت لیست تمام موضوعات موجود

        Returns:
            لیست نام موضوعات
        """
        metadata = self.admin_client.list_topics(timeout=10)
        return list(metadata.topics.keys())

    async def get_topic_config(self, topic_name: str) -> Dict:
        """
        دریافت تنظیمات یک موضوع

        Args:
            topic_name: نام موضوع

        Returns:
            دیکشنری حاوی تنظیمات موضوع
        """
        resource = ConfigResource(RESOURCE_TYPE_TOPIC, topic_name)
        result = self.admin_client.describe_configs([resource])
        configs = result[resource].result()
        return {key: config.value for key, config in configs.items()}

    async def get_topic_metrics(self, topic_name: str) -> Dict:
        """
        دریافت متریک‌های یک موضوع

        Args:
            topic_name: نام موضوع

        Returns:
            دیکشنری حاوی متریک‌های موضوع
        """
        metadata = self.admin_client.list_topics(topic=topic_name, timeout=10)
        topic_metadata = metadata.topics[topic_name]

        return {
            'partition_count': len(topic_metadata.partitions),
            'partition_info': [
                {
                    'id': p.id,
                    'leader': p.leader,
                    'replicas': p.replicas,
                    'isrs': p.isrs
                }
                for p in topic_metadata.partitions.values()
            ]
        }

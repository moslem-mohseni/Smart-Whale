from confluent_kafka.admin import AdminClient, ConfigResource
from ..config.settings import KafkaConfig
import logging
import time
from typing import Dict

logger = logging.getLogger(__name__)


class KafkaMaintenance:
    """
    عملیات نگهداری و پاک‌سازی Kafka
    """

    def __init__(self, config: KafkaConfig):
        """
        مقداردهی اولیه کلاس

        :param config: تنظیمات Kafka
        """
        self.config = config
        self.admin_client = AdminClient({"bootstrap.servers": ",".join(self.config.bootstrap_servers)})

    def check_health(self) -> Dict[str, str]:
        """
        بررسی سلامت Kafka از طریق ارسال و دریافت پیام تستی

        :return: وضعیت سلامت Kafka
        """
        try:
            metadata = self.admin_client.list_topics(timeout=5)
            if metadata.brokers:
                return {"status": "Healthy", "message": "Kafka is running and accessible."}
            else:
                return {"status": "Unhealthy", "message": "No brokers available."}
        except Exception as e:
            logger.error(f"Kafka health check failed: {e}")
            return {"status": "Unhealthy", "message": str(e)}

    def cleanup_old_messages(self, topic: str, retention_days: int) -> bool:
        """
        تنظیم `retention.ms` برای حذف پیام‌های قدیمی از Kafka

        :param topic: نام `topic`
        :param retention_days: مدت نگه‌داری پیام‌ها بر حسب روز
        :return: `True` در صورت موفقیت، `False` در غیر این صورت
        """
        retention_ms = retention_days * 24 * 60 * 60 * 1000  # تبدیل روز به میلی‌ثانیه

        try:
            config_resource = ConfigResource("TOPIC", topic, {"retention.ms": str(retention_ms)})
            future = self.admin_client.alter_configs([config_resource])

            future[config_resource].result()  # مسدود شدن تا اتمام عملیات
            logger.info(f"Retention policy for {topic} set to {retention_days} days.")
            return True

        except Exception as e:
            logger.error(f"Failed to update retention policy for {topic}: {e}")
            return False

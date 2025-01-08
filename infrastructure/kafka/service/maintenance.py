# infrastructure/kafka/scripts/maintenance.py

"""
این اسکریپت شامل توابعی برای نگهداری و مدیریت سیستم کافکا است.
عملیاتی مانند پاکسازی داده‌های قدیمی، بررسی سلامت سیستم و تهیه گزارش‌های مختلف.
"""

from ..service.kafka_service import KafkaService
from typing import List, Dict
import logging
import time

logger = logging.getLogger(__name__)


class KafkaMaintenance:
    """عملیات نگهداری کافکا"""

    def __init__(self, kafka_service: KafkaService):
        self.kafka_service = kafka_service

    async def check_health(self) -> Dict:
        """
        بررسی سلامت کلی سیستم کافکا

        Returns:
            دیکشنری حاوی وضعیت سلامت سیستم
        """
        try:
            # تلاش برای ارسال و دریافت یک پیام تست
            test_topic = "health.check"
            test_message = {
                'type': 'health_check',
                'timestamp': time.time()
            }

            # ارسال پیام تست
            await self.kafka_service.send_message(test_topic, test_message)

            # دریافت و بررسی پیام
            received = False
            async for msg in self.kafka_service.get_consumer("health_checker"):
                if msg.value().get('type') == 'health_check':
                    received = True
                    break

            return {
                'status': 'healthy' if received else 'unhealthy',
                'producer_status': 'working',
                'consumer_status': 'working' if received else 'not_working',
                'timestamp': time.time()
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }

    async def cleanup_old_messages(self, topic: str, older_than_days: int) -> int:
        """
        پاکسازی پیام‌های قدیمی از یک موضوع

        Args:
            topic: نام موضوع
            older_than_days: پیام‌های قدیمی‌تر از این تعداد روز حذف می‌شوند

        Returns:
            تعداد پیام‌های حذف شده
        """
        # محاسبه زمان برش
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)

        try:
            # تنظیم سیاست نگهداری
            admin = AdminClient({
                'bootstrap.servers': ','.join(self.kafka_service.config.bootstrap_servers)
            })

            # اعمال تنظیمات جدید
            config = {
                'cleanup.policy': 'delete',
                'retention.ms': str(int(cutoff_time * 1000))
            }

            resource = ConfigResource(
                RESOURCE_TYPE_TOPIC,
                topic,
                config
            )

            future = admin.alter_configs([resource])
            future[resource].result()

            return 0  # تعداد دقیق پیام‌های حذف شده قابل محاسبه نیست

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise
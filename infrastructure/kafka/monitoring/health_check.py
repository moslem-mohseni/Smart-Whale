import time
import logging
from confluent_kafka.admin import AdminClient
from ..config.settings import KafkaConfig
from ..service.kafka_service import KafkaService
from typing import Dict

logger = logging.getLogger(__name__)


class KafkaHealthCheck:
    """
    بررسی سلامت Kafka
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self.admin_client = AdminClient({'bootstrap.servers': ','.join(config.bootstrap_servers)})
        self.kafka_service = KafkaService(config)

    async def check_kafka_connection(self) -> bool:
        """
        بررسی اتصال به Kafka

        :return: True اگر Kafka در دسترس باشد، False در غیر این صورت
        """
        try:
            cluster_metadata = self.admin_client.list_topics(timeout=5)
            if cluster_metadata.brokers:
                return True
        except Exception as e:
            logger.error(f"Kafka connection failed: {e}")
        return False

    async def check_producer_health(self) -> bool:
        """
        بررسی سلامت تولیدکننده Kafka
        """
        try:
            test_message = "health_check"
            test_topic = "kafka_health_check"
            await self.kafka_service.send_message(test_topic, test_message)
            return True
        except Exception as e:
            logger.error(f"Kafka producer health check failed: {e}")
            return False

    async def check_consumer_health(self) -> bool:
        """
        بررسی سلامت مصرف‌کننده Kafka
        """
        try:
            test_topic = "kafka_health_check"
            messages = await self.kafka_service.subscribe(test_topic, "health_checker", lambda msg: msg)
            return bool(messages)
        except Exception as e:
            logger.error(f"Kafka consumer health check failed: {e}")
            return False

    async def check_latency(self) -> float:
        """
        اندازه‌گیری میزان تاخیر (latency) در ارسال و دریافت پیام

        :return: مقدار تاخیر بر حسب ثانیه
        """
        start_time = time.time()
        is_producer_healthy = await self.check_producer_health()
        is_consumer_healthy = await self.check_consumer_health()
        if is_producer_healthy and is_consumer_healthy:
            return time.time() - start_time
        return -1

    async def get_health_report(self) -> Dict[str, any]:
        """
        دریافت وضعیت سلامت Kafka

        :return: دیکشنری شامل اطلاعات سلامت Kafka
        """
        kafka_status = await self.check_kafka_connection()
        producer_status = await self.check_producer_health()
        consumer_status = await self.check_consumer_health()
        latency = await self.check_latency()

        return {
            "kafka_status": "Healthy" if kafka_status else "Unhealthy",
            "producer_status": "Healthy" if producer_status else "Unhealthy",
            "consumer_status": "Healthy" if consumer_status else "Unhealthy",
            "latency": latency if latency >= 0 else "Unknown"
        }

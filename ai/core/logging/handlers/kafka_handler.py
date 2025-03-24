import logging
import json
from kafka import KafkaProducer

class KafkaLogHandler:
    def __init__(self, kafka_servers, topic="logs", log_level=logging.INFO):
        """
        ارسال لاگ‌های سیستم به Kafka
        :param kafka_servers: لیست سرورهای Kafka (مثلاً: ["localhost:9092"])
        :param topic: نام Topic برای ارسال لاگ‌ها
        :param log_level: سطح لاگ (پیش‌فرض: INFO)
        """
        self.logger = logging.getLogger("KafkaLogger")
        self.logger.setLevel(log_level)

        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
            self.topic = topic
            self.logger.info("✅ اتصال به Kafka برقرار شد.")
        except Exception as e:
            self.logger.error(f"❌ خطا در اتصال به Kafka: {e}")
            self.producer = None

    def log(self, level, message):
        """ ارسال لاگ به Kafka """
        if not self.producer:
            self.logger.error("⚠️ Kafka Producer مقداردهی نشده است، لاگ ارسال نمی‌شود.")
            return

        log_entry = {
            "level": level.upper(),
            "message": message,
            "logger": "KafkaLogger"
        }

        try:
            self.producer.send(self.topic, log_entry)
            self.logger.info(f"📡 لاگ ارسال شد به Kafka → Topic: {self.topic}")
        except Exception as e:
            self.logger.error(f"❌ خطا در ارسال لاگ به Kafka: {e}")

    def info(self, message):
        """ ارسال لاگ در سطح INFO به Kafka """
        self.log("info", message)

    def error(self, message):
        """ ارسال لاگ در سطح ERROR به Kafka """
        self.log("error", message)

"""
ماژول `messaging/` وظیفه‌ی مدیریت ارتباطات Kafka را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `kafka_producer.py` → ارسال پیام‌های پردازشی زبان به Kafka
- `kafka_consumer.py` → دریافت و پردازش پیام‌های Kafka
"""

from .kafka_producer import KafkaProducer
from .kafka_consumer import KafkaConsumer
from infrastructure.kafka.service.kafka_service import KafkaService

# مقداردهی اولیه KafkaService
kafka_service = KafkaService()

# مقداردهی اولیه KafkaProducer و KafkaConsumer
kafka_producer = KafkaProducer(kafka_service)
kafka_consumer = KafkaConsumer(kafka_service)

__all__ = [
    "kafka_producer",
    "kafka_consumer",
    "KafkaProducer",
    "KafkaConsumer",
]

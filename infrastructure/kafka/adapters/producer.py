

# infrastructure/kafka/adapters/producer.py
from ..domain.models import Message
from ..config.settings import KafkaConfig
from ...interfaces import MessagingInterface
from confluent_kafka import Producer as KafkaProducer
import json
import logging

logger = logging.getLogger(__name__)


class MessageProducer:
    """
    کلاس تولیدکننده پیام

    این کلاس مسئولیت ارسال پیام‌ها به کافکا را بر عهده دارد.
    """

    def __init__(self, config: KafkaConfig):
        self.config = config
        self._producer = None

    def _create_producer(self):
        """ایجاد یک نمونه از تولیدکننده کافکا"""
        if not self._producer:
            self._producer = KafkaProducer(self.config.get_producer_config())

    async def send(self, message: Message) -> None:
        """
        ارسال یک پیام به کافکا

        Args:
            message: پیامی که باید ارسال شود
        """
        self._create_producer()

        try:
            # تبدیل محتوا به JSON
            value = json.dumps({
                'content': message.content,
                'timestamp': message.timestamp.isoformat(),
                'metadata': message.metadata
            }).encode('utf-8')

            # ارسال پیام
            self._producer.produce(
                topic=message.topic,
                value=value,
                on_delivery=self._delivery_report
            )

            # اطمینان از ارسال همه پیام‌های در صف
            self._producer.flush()

        except Exception as e:
            logger.error(f"Error sending message to Kafka: {str(e)}")
            raise

    def _delivery_report(self, err, msg):
        """گزارش وضعیت تحویل پیام"""
        if err is not None:
            logger.error(f'Message delivery failed: {str(err)}')
        else:
            logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')
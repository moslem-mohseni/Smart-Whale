import asyncio
import logging
import json
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Callable, Optional

logging.basicConfig(level=logging.INFO)


class StreamProcessor:
    def __init__(self,
                 kafka_bootstrap_servers: str,
                 topic: str,
                 group_id: str,
                 process_function: Callable[[dict], None],
                 output_topic: Optional[str] = None):
        """
        پردازشگر جریانی داده‌ها با پشتیبانی از Kafka

        :param kafka_bootstrap_servers: آدرس Kafka Cluster
        :param topic: تاپیک Kafka که داده‌ها را دریافت می‌کند
        :param group_id: گروه مصرف‌کننده Kafka
        :param process_function: تابع پردازش داده که باید بر روی هر پیام اجرا شود
        :param output_topic: در صورت نیاز، ارسال داده‌های پردازش شده به این تاپیک
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self.process_function = process_function
        self.output_topic = output_topic

        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=True,
            auto_offset_reset='earliest',
        )

        if self.output_topic:
            self.producer = AIOKafkaProducer(bootstrap_servers=self.kafka_bootstrap_servers)
        else:
            self.producer = None

    async def start(self):
        """
        شروع پردازش داده‌های جریانی از Kafka
        """
        await self.consumer.start()
        if self.producer:
            await self.producer.start()

        try:
            logging.info(f"✅ Stream Processor Started for Topic: {self.topic}")
            async for message in self.consumer:
                data = json.loads(message.value.decode("utf-8"))
                logging.info(f"📥 Received Message: {data}")

                # پردازش داده
                processed_data = await self._process_data(data)

                # ارسال به Kafka (در صورت نیاز)
                if self.output_topic and processed_data:
                    await self._send_to_kafka(processed_data)

        except Exception as e:
            logging.error(f"❌ Error in Stream Processing: {str(e)}")

        finally:
            await self.consumer.stop()
            if self.producer:
                await self.producer.stop()

    async def _process_data(self, data: dict) -> Optional[dict]:
        """
        اجرای پردازش اصلی روی داده‌ها

        :param data: داده ورودی
        :return: داده پردازش شده (در صورت موفقیت)
        """
        try:
            processed_data = self.process_function(data)
            logging.info(f"✅ Processed Data: {processed_data}")
            return processed_data
        except Exception as e:
            logging.error(f"❌ Processing Error: {str(e)}")
            return None

    async def _send_to_kafka(self, processed_data: dict):
        """
        ارسال داده‌های پردازش شده به Kafka

        :param processed_data: داده‌ی پردازش شده
        """
        if self.producer:
            try:
                message = json.dumps(processed_data).encode("utf-8")
                await self.producer.send_and_wait(self.output_topic, message)
                logging.info(f"📤 Sent Processed Data to {self.output_topic}: {processed_data}")
            except Exception as e:
                logging.error(f"❌ Error Sending Data to Kafka: {str(e)}")


async def sample_processing_function(data: dict) -> dict:
    """
    یک تابع نمونه برای پردازش داده‌های جریانی
    :param data: داده ورودی
    :return: داده پردازش شده
    """
    return {
        "processed_data": data.get("raw_data", "").upper(),  # تبدیل به حروف بزرگ
        "timestamp": data.get("timestamp", ""),
    }


async def test_stream_processor():
    """
    تست عملکرد StreamProcessor
    """
    processor = StreamProcessor(
        kafka_bootstrap_servers="localhost:9092",
        topic="raw_data_stream",
        group_id="stream_processor_group",
        process_function=sample_processing_function,
        output_topic="processed_data_stream",
    )

    await processor.start()


# اجرای تست
if __name__ == "__main__":
    asyncio.run(test_stream_processor())

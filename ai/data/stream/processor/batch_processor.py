import asyncio
import logging
import json
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from typing import Callable, Optional, List
from buffer import buffer_manager

logging.basicConfig(level=logging.INFO)


class BatchProcessor:
    def __init__(self,
                 kafka_bootstrap_servers: str,
                 input_topic: str,
                 output_topic: Optional[str],
                 group_id: str,
                 batch_size: int = 100,
                 batch_interval: float = 5.0,
                 process_function: Callable[[List[dict]], List[dict]] = None):
        """
        پردازشگر دسته‌ای داده‌ها با پشتیبانی از Kafka

        :param kafka_bootstrap_servers: آدرس Kafka Cluster
        :param input_topic: تاپیک Kafka که داده‌ها را دریافت می‌کند
        :param output_topic: تاپیک خروجی Kafka برای انتشار داده‌های پردازش‌شده (در صورت نیاز)
        :param group_id: گروه مصرف‌کننده Kafka
        :param batch_size: حداکثر تعداد پیام‌ها در هر دسته
        :param batch_interval: فاصله‌ی زمانی برای ارسال دسته‌ی داده‌ها (بر حسب ثانیه)
        :param process_function: تابع پردازش دسته‌ای که روی داده‌ها اجرا می‌شود
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.group_id = group_id
        self.batch_size = batch_size
        self.batch_interval = batch_interval
        self.process_function = process_function or self.default_process_function

        self.consumer = AIOKafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.kafka_bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            auto_offset_reset='earliest',
        )

        self.producer = AIOKafkaProducer(bootstrap_servers=self.kafka_bootstrap_servers) if self.output_topic else None

        self.buffer = buffer_manager.buffer  # استفاده از SmartBuffer
        self.running = True

    async def start(self):
        """
        شروع پردازش دسته‌ای داده‌ها از Kafka
        """
        await self.consumer.start()
        if self.producer:
            await self.producer.start()

        asyncio.create_task(self._batch_dispatcher())  # اجرای پردازش دسته‌ای در پس‌زمینه

        try:
            logging.info(f"✅ Batch Processor Started for Topic: {self.input_topic}")
            async for message in self.consumer:
                data = json.loads(message.value.decode("utf-8"))
                logging.info(f"📥 Received Message: {data}")

                await self.buffer.add(data)

        except Exception as e:
            logging.error(f"❌ Error in Batch Processing: {str(e)}")

        finally:
            await self.consumer.stop()
            if self.producer:
                await self.producer.stop()

    async def _batch_dispatcher(self):
        """
        بررسی و ارسال دسته‌های داده‌ها در فواصل زمانی مشخص
        """
        while self.running:
            await asyncio.sleep(self.batch_interval)

            batch = []
            for _ in range(self.batch_size):
                data = await self.buffer.get()
                if data:
                    batch.append(data)
                else:
                    break  # اگر داده‌ای در بافر نباشد، حلقه متوقف می‌شود

            if batch:
                processed_batch = await self._process_batch(batch)

                if self.output_topic and processed_batch:
                    await self._send_to_kafka(processed_batch)

    async def _process_batch(self, batch: List[dict]) -> List[dict]:
        """
        اجرای پردازش اصلی روی دسته‌ای از داده‌ها

        :param batch: لیست داده‌های خام
        :return: لیست داده‌های پردازش‌شده
        """
        try:
            processed_batch = self.process_function(batch)
            logging.info(f"✅ Processed Batch of {len(batch)} messages")
            return processed_batch
        except Exception as e:
            logging.error(f"❌ Batch Processing Error: {str(e)}")
            return []

    async def _send_to_kafka(self, processed_batch: List[dict]):
        """
        ارسال دسته‌ی پردازش‌شده به Kafka

        :param processed_batch: لیست داده‌های پردازش‌شده
        """
        if self.producer:
            try:
                for item in processed_batch:
                    message = json.dumps(item).encode("utf-8")
                    await self.producer.send_and_wait(self.output_topic, message)

                logging.info(f"📤 Sent {len(processed_batch)} Processed Messages to {self.output_topic}")

            except Exception as e:
                logging.error(f"❌ Error Sending Batch to Kafka: {str(e)}")

    async def stop(self):
        """
        توقف پردازش دسته‌ای
        """
        self.running = False
        await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        logging.info("⛔ Batch Processor Stopped.")

    @staticmethod
    def default_process_function(batch: List[dict]) -> List[dict]:
        """
        پردازش پیش‌فرض برای داده‌های ورودی

        :param batch: لیست داده‌های ورودی
        :return: لیست داده‌های پردازش‌شده
        """
        return [{"processed_data": item.get("raw_data", "").upper()} for item in batch]  # تبدیل به حروف بزرگ


async def test_batch_processor():
    """
    تست عملکرد BatchProcessor
    """
    processor = BatchProcessor(
        kafka_bootstrap_servers="localhost:9092",
        input_topic="raw_data_batch",
        output_topic="processed_data_batch",
        group_id="batch_processor_group",
        batch_size=10,
        batch_interval=5,
        process_function=BatchProcessor.default_process_function,
    )

    await processor.start()


# اجرای تست
if __name__ == "__main__":
    asyncio.run(test_batch_processor())

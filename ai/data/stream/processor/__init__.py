import asyncio
import logging

from .stream_processor import StreamProcessor
from .batch_processor import BatchProcessor
from .flow_controller import FlowController

logging.basicConfig(level=logging.INFO)


class ProcessorManager:
    def __init__(self,
                 kafka_bootstrap_servers: str = "localhost:9092",
                 stream_topic: str = "raw_data_stream",
                 batch_input_topic: str = "raw_data_batch",
                 batch_output_topic: str = "processed_data_batch",
                 stream_group_id: str = "stream_processor_group",
                 batch_group_id: str = "batch_processor_group",
                 batch_size: int = 100,
                 batch_interval: float = 5.0,
                 max_queue_size: int = 5000,
                 batch_processing_threshold: int = 1000):
        """
        مدیریت یکپارچه‌ی پردازشگرهای جریانی، دسته‌ای و کنترل‌کننده‌ی جریان داده

        :param kafka_bootstrap_servers: آدرس Kafka Cluster
        :param stream_topic: تاپیک Kafka که داده‌های جریانی دریافت می‌کند
        :param batch_input_topic: تاپیک Kafka که داده‌های دسته‌ای دریافت می‌کند
        :param batch_output_topic: تاپیک Kafka که داده‌های پردازش‌شده را منتشر می‌کند
        :param stream_group_id: گروه مصرف‌کننده Kafka برای پردازش جریانی
        :param batch_group_id: گروه مصرف‌کننده Kafka برای پردازش دسته‌ای
        :param batch_size: تعداد پیام‌ها در هر دسته پردازش
        :param batch_interval: فاصله‌ی زمانی پردازش دسته‌ای
        :param max_queue_size: حداکثر اندازه‌ی بافر داده
        :param batch_processing_threshold: حد آستانه‌ی ارسال داده‌ها به پردازش دسته‌ای
        """
        self.stream_processor = StreamProcessor(
            kafka_bootstrap_servers=kafka_bootstrap_servers,
            topic=stream_topic,
            group_id=stream_group_id,
            process_function=self.default_stream_processing,
            output_topic=batch_input_topic
        )

        self.batch_processor = BatchProcessor(
            kafka_bootstrap_servers=kafka_bootstrap_servers,
            input_topic=batch_input_topic,
            output_topic=batch_output_topic,
            group_id=batch_group_id,
            batch_size=batch_size,
            batch_interval=batch_interval,
            process_function=self.default_batch_processing,
        )

        self.flow_controller = FlowController(
            max_queue_size=max_queue_size,
            batch_processing_threshold=batch_processing_threshold
        )

    async def start_processors(self):
        """
        شروع پردازشگرهای جریانی، دسته‌ای و کنترل‌کننده‌ی جریان
        """
        await self.flow_controller.attach_processors(self.stream_processor, self.batch_processor)

        asyncio.create_task(self.flow_controller.monitor_flow())
        asyncio.create_task(self.stream_processor.start())
        asyncio.create_task(self.batch_processor.start())

        logging.info("✅ All processors have been started!")

    async def stop_processors(self):
        """
        توقف تمامی پردازشگرها
        """
        await self.flow_controller.stop()
        await self.stream_processor.stop()
        await self.batch_processor.stop()

        logging.info("⛔ All processors have been stopped!")

    @staticmethod
    def default_stream_processing(data: dict) -> dict:
        """
        پردازش پیش‌فرض داده‌های جریانی
        :param data: داده ورودی
        :return: داده پردازش‌شده
        """
        return {"processed_stream_data": data.get("raw_data", "").upper()}  # تبدیل به حروف بزرگ

    @staticmethod
    def default_batch_processing(batch: list) -> list:
        """
        پردازش پیش‌فرض برای داده‌های دسته‌ای
        :param batch: لیست داده‌های ورودی
        :return: لیست داده‌های پردازش‌شده
        """
        return [{"processed_batch_data": item.get("raw_data", "").lower()} for item in batch]  # تبدیل به حروف کوچک


# مقداردهی اولیه‌ی ماژول
processor_manager = ProcessorManager()

# راه‌اندازی پردازشگرها
asyncio.create_task(processor_manager.start_processors())

# API ماژول
__all__ = ["processor_manager", "StreamProcessor", "BatchProcessor", "FlowController"]

import asyncio
import logging
from pipeline.stages import CollectorStage, ProcessorStage, PublisherStage
from infrastructure.kafka.service.kafka_service import KafkaService
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)


class FlowManager:
    """
    مدیریت جریان اجرای مراحل مختلف در Pipeline.
    """

    def __init__(self, kafka_topic: str = "raw_data"):
        """
        مقداردهی اولیه.

        :param kafka_topic: تاپیک Kafka برای دریافت داده‌های اولیه
        """
        self.kafka_service = KafkaService()
        self.collector_stage = CollectorStage(kafka_topic=kafka_topic)
        self.processor_stage = ProcessorStage()
        self.publisher_stage = PublisherStage(kafka_topic="processed_data")

    async def connect(self) -> None:
        """ اتصال به Kafka و سایر سرویس‌های موردنیاز. """
        await self.kafka_service.connect()
        await self.collector_stage.connect()
        await self.processor_stage.connect()
        await self.publisher_stage.connect()

    async def process_message(self, raw_data: Dict[str, Any]) -> None:
        """
        اجرای پردازش داده‌ها از ابتدا تا انتشار.

        :param raw_data: داده‌ی دریافتی از Kafka
        """
        try:
            logging.info(f"🔄 شروع پردازش داده با ID: {raw_data.get('id')}")

            # پردازش مرحله جمع‌آوری داده
            collected_data = await self.collector_stage.process_data(raw_data)
            if not collected_data:
                logging.warning(f"⚠️ داده با ID {raw_data.get('id')} در مرحله جمع‌آوری رد شد.")
                return

            # پردازش داده‌های جمع‌آوری‌شده
            processed_data = await self.processor_stage.process_data(collected_data)
            if not processed_data:
                logging.warning(f"⚠️ داده با ID {raw_data.get('id')} در مرحله پردازش رد شد.")
                return

            # انتشار داده‌های پردازش‌شده
            await self.publisher_stage.publish_data(processed_data)
            logging.info(f"✅ داده با ID {raw_data.get('id')} با موفقیت پردازش و منتشر شد.")

        except Exception as e:
            logging.error(f"❌ خطا در پردازش داده: {e}")

    async def start_pipeline(self) -> None:
        """
        راه‌اندازی فرآیند پردازش داده‌ها از Kafka.
        """

        async def message_handler(message: Dict[str, Any]):
            await self.process_message(message)

        await self.kafka_service.subscribe("raw_data", "pipeline_group", message_handler)

    async def close(self) -> None:
        """ قطع اتصال از Kafka و سایر سرویس‌ها. """
        await self.kafka_service.disconnect()
        await self.collector_stage.close()
        await self.processor_stage.close()
        await self.publisher_stage.close()


# مقداردهی اولیه و راه‌اندازی Pipeline
async def start_pipeline():
    flow_manager = FlowManager()
    await flow_manager.connect()
    await flow_manager.start_pipeline()


# اجرای Pipeline به‌صورت ناهمزمان
asyncio.create_task(start_pipeline())

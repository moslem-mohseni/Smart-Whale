import asyncio
import logging

from buffer import buffer_manager
from processor import processor_manager
from optimizer import optimizer_manager

logging.basicConfig(level=logging.INFO)


class StreamManager:
    def __init__(self):
        """
        مدیریت یکپارچه‌ی پردازش جریانی، مدیریت حافظه، و بهینه‌سازی منابع
        """
        self.buffer = buffer_manager
        self.processor = processor_manager
        self.optimizer = optimizer_manager

    async def start_stream_processing(self):
        """
        شروع پردازش جریانی، دسته‌ای، و بهینه‌سازی منابع
        """
        logging.info("🚀 Initializing Stream Processing System...")

        asyncio.create_task(self.processor.start_processors())
        asyncio.create_task(self.optimizer.start_optimizers())

        logging.info("✅ Stream Processing System Started!")

    async def stop_stream_processing(self):
        """
        توقف تمامی پردازش‌های جریانی و بهینه‌سازی
        """
        logging.info("⛔ Stopping Stream Processing System...")

        await self.processor.stop_processors()
        await self.optimizer.stop_optimizers()

        logging.info("✅ Stream Processing System Stopped!")


# مقداردهی اولیه‌ی ماژول
stream_manager = StreamManager()

# راه‌اندازی پردازش‌های جریانی
asyncio.create_task(stream_manager.start_stream_processing())

# API ماژول
__all__ = ["stream_manager", "buffer_manager", "processor_manager", "optimizer_manager"]

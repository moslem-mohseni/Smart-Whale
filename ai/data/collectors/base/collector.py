import asyncio
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)


class BaseCollector(ABC):
    """
    کلاس پایه برای جمع‌آوری داده از منابع مختلف.
    تمام جمع‌آورنده‌های داده باید از این کلاس ارث‌بری کنند.
    """

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.is_active = False

    @abstractmethod
    async def collect_data(self):
        """
        متد جمع‌آوری داده که در کلاس‌های فرزند پیاده‌سازی می‌شود.
        """
        pass

    async def start_collection(self):
        """
        متد شروع جمع‌آوری داده
        """
        self.is_active = True
        logging.info(f"📡 Starting data collection from {self.source_name}...")

        while self.is_active:
            try:
                data = await self.collect_data()
                await self.process_data(data)
            except Exception as e:
                logging.error(f"❌ Error collecting data from {self.source_name}: {e}")

            await asyncio.sleep(1)  # جلوگیری از اجرای بیش از حد سریع

    async def stop_collection(self):
        """
        متد متوقف‌سازی جمع‌آوری داده
        """
        self.is_active = False
        logging.info(f"⛔ Stopping data collection from {self.source_name}...")

    async def process_data(self, data):
        """
        پردازش داده‌های جمع‌آوری‌شده و ارسال به Stream
        """
        if data:
            logging.info(f"✅ Collected data from {self.source_name}: {data}")
            # در این نسخه، فقط لاگ می‌کنیم و داده را برمی‌گردانیم
            return data
        else:
            logging.warning(f"⚠ No data collected from {self.source_name}.")
            return None


# نمونه یک جمع‌آورنده داده برای تست
class ExampleCollector(BaseCollector):
    async def collect_data(self):
        return {"message": "Sample data"}


if __name__ == "__main__":
    collector = ExampleCollector("TestSource")
    asyncio.run(collector.start_collection())
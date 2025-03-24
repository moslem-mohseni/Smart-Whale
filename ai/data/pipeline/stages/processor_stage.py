import asyncio
import json
from infrastructure.redis.service.cache_service import CacheService
from data.intelligence import StreamOptimizer, PerformanceAnalyzer
from typing import Dict, Any

class ProcessorStage:
    """
    مرحله پردازش داده‌های دریافتی از Collector Stage.
    """

    def __init__(self, cache_ttl: int = 3600):
        """
        مقداردهی اولیه.

        :param cache_ttl: مدت زمان نگهداری داده در Redis (به ثانیه)
        """
        self.cache_service = CacheService()
        self.stream_optimizer = StreamOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.cache_ttl = cache_ttl

    async def connect(self) -> None:
        """ اتصال به Redis. """
        await self.cache_service.connect()

    async def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        پردازش و بهینه‌سازی داده‌ها.

        :param raw_data: داده‌های دریافتی از Collector Stage
        :return: داده‌های پردازش‌شده
        """
        data_id = raw_data.get("id")
        cache_key = f"processor_stage:{data_id}"
        cached_result = await self.cache_service.get(cache_key)

        if cached_result:
            print(f"⚠️ داده با ID {data_id} قبلاً پردازش شده، رد شد.")
            return json.loads(cached_result)

        # اجرای بهینه‌سازی پردازش
        optimized_data = await self.stream_optimizer.optimize(raw_data)

        # تحلیل عملکرد پردازش
        processing_metrics = await self.performance_analyzer.analyze(optimized_data)
        print(f"📊 متریک‌های پردازش: {processing_metrics}")

        # کش کردن داده پردازش‌شده
        await self.cache_service.set(cache_key, json.dumps(optimized_data), ttl=self.cache_ttl)

        return optimized_data

    async def send_to_publisher(self, processed_data: Dict[str, Any]) -> None:
        """
        ارسال داده پردازش‌شده به Publisher Stage.

        :param processed_data: داده‌های پردازش‌شده
        """
        print(f"✅ داده پردازش شد و برای انتشار آماده است: {processed_data}")

    async def close(self) -> None:
        """ قطع اتصال از Redis. """
        await self.cache_service.disconnect()

# مقداردهی اولیه و راه‌اندازی پردازشگر داده‌ها
async def start_processor_stage(raw_data: Dict[str, Any]):
    processor_stage = ProcessorStage()
    await processor_stage.connect()
    processed_data = await processor_stage.process_data(raw_data)
    await processor_stage.send_to_publisher(processed_data)

# اجرای پردازشگر به صورت ناهمزمان برای هر داده
asyncio.create_task(start_processor_stage({"id": "1234", "content": "مثال داده"}))

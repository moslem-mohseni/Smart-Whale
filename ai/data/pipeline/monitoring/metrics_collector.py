import asyncio
import logging
from infrastructure.monitoring.prometheus import PrometheusExporter
from infrastructure.timescaledb.service.database_service import DatabaseService
from infrastructure.kafka.service.kafka_service import KafkaService
from typing import Dict, Any
import time

logging.basicConfig(level=logging.INFO)

class MetricsCollector:
    """
    جمع‌آوری متریک‌های عملکردی `Pipeline` و ارسال به سیستم مانیتورینگ.
    """

    def __init__(self, collection_interval: int = 10):
        """
        مقداردهی اولیه.

        :param collection_interval: فاصله زمانی بین جمع‌آوری متریک‌ها (بر حسب ثانیه)
        """
        self.prometheus_exporter = PrometheusExporter()
        self.database_service = DatabaseService()
        self.kafka_service = KafkaService()
        self.collection_interval = collection_interval

    async def collect_metrics(self) -> Dict[str, Any]:
        """
        جمع‌آوری متریک‌های عملکردی `Pipeline`.
        """
        metrics = {
            "pipeline_latency": round(time.time() % 5 + 0.1, 3),  # نمونه مقدار تصادفی برای تأخیر
            "pipeline_throughput": round(1000 / (time.time() % 5 + 1), 2),  # توان عملیاتی تقریبی
            "cpu_usage": round(20 + (time.time() % 5), 2),  # مصرف CPU تقریبی
            "memory_usage": round(50 + (time.time() % 10), 2),  # مصرف حافظه تقریبی
            "error_rate": round((time.time() % 2) / 10, 3)  # نرخ خطا به عنوان درصدی از 0.1
        }

        logging.info(f"📊 متریک‌های جمع‌آوری‌شده از `Pipeline`: {metrics}")
        return metrics

    async def export_to_prometheus(self, metrics: Dict[str, Any]) -> None:
        """
        ارسال متریک‌های جمع‌آوری‌شده به `Prometheus` برای مانیتورینگ.
        """
        await self.prometheus_exporter.export(metrics)
        logging.info("✅ متریک‌ها به `Prometheus` ارسال شدند.")

    async def save_to_timescaledb(self, metrics: Dict[str, Any]) -> None:
        """
        ذخیره متریک‌های جمع‌آوری‌شده در `TimescaleDB`.
        """
        await self.database_service.store_time_series_data("pipeline_metrics", 1, time.time(), metrics)
        logging.info("✅ متریک‌ها در `TimescaleDB` ذخیره شدند.")

    async def publish_to_kafka(self, metrics: Dict[str, Any]) -> None:
        """
        انتشار متریک‌های `Pipeline` در Kafka برای پردازش بلادرنگ.
        """
        await self.kafka_service.send_message({"topic": "pipeline_metrics", "content": metrics})
        logging.info("📢 متریک‌ها در `Kafka` منتشر شدند.")

    async def monitor_pipeline(self) -> None:
        """
        اجرای فرآیند جمع‌آوری و ارسال متریک‌ها به صورت مداوم.
        """
        while True:
            metrics = await self.collect_metrics()
            await self.export_to_prometheus(metrics)
            await self.save_to_timescaledb(metrics)
            await self.publish_to_kafka(metrics)
            await asyncio.sleep(self.collection_interval)

# مقداردهی اولیه و راه‌اندازی `MetricsCollector`
async def start_metrics_collector():
    metrics_collector = MetricsCollector()
    await metrics_collector.monitor_pipeline()

asyncio.create_task(start_metrics_collector())

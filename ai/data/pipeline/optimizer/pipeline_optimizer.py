import asyncio
import logging
from infrastructure.monitoring.metrics import MetricsCollector
from data.intelligence import StreamOptimizer, PerformanceAnalyzer
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)


class PipelineOptimizer:
    """
    بهینه‌سازی کلی Pipeline برای بهبود عملکرد و کاهش مصرف منابع.
    """

    def __init__(self, optimization_interval: int = 60):
        """
        مقداردهی اولیه.

        :param optimization_interval: فاصله زمانی بین اجرای فرآیندهای بهینه‌سازی (بر حسب ثانیه)
        """
        self.metrics_collector = MetricsCollector()
        self.stream_optimizer = StreamOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.optimization_interval = optimization_interval

    async def collect_metrics(self) -> Dict[str, Any]:
        """
        جمع‌آوری متریک‌های عملکردی `Pipeline`.
        """
        metrics = await self.metrics_collector.collect()
        logging.info(f"📊 متریک‌های جمع‌آوری‌شده از `Pipeline`: {metrics}")
        return metrics

    async def analyze_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحلیل عملکرد `Pipeline` بر اساس متریک‌های جمع‌آوری‌شده.
        """
        performance_report = await self.performance_analyzer.analyze(metrics)
        logging.info(f"📈 تحلیل عملکرد `Pipeline`: {performance_report}")
        return performance_report

    async def optimize_pipeline(self) -> None:
        """
        اجرای فرآیند بهینه‌سازی `Pipeline`.
        """
        logging.info("🔄 شروع فرآیند بهینه‌سازی `Pipeline`...")

        metrics = await self.collect_metrics()
        performance_report = await self.analyze_performance(metrics)

        # بهینه‌سازی جریان داده بر اساس تحلیل عملکرد
        await self.stream_optimizer.optimize(performance_report)

        logging.info("✅ بهینه‌سازی `Pipeline` با موفقیت انجام شد.")

    async def start_optimization_loop(self) -> None:
        """
        اجرای مداوم فرآیندهای بهینه‌سازی در بازه‌های زمانی مشخص.
        """
        while True:
            await self.optimize_pipeline()
            await asyncio.sleep(self.optimization_interval)


# مقداردهی اولیه و راه‌اندازی `PipelineOptimizer`
async def start_pipeline_optimizer():
    pipeline_optimizer = PipelineOptimizer()
    await pipeline_optimizer.start_optimization_loop()


asyncio.create_task(start_pipeline_optimizer())

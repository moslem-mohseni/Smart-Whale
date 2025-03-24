import asyncio
import logging
from infrastructure.monitoring.metrics import MetricsCollector
from data.intelligence import ThroughputOptimizer, ResourceBalancer
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)


class TransitionOptimizer:
    """
    بهینه‌سازی انتقال داده بین مراحل `Pipeline`.
    """

    def __init__(self, optimization_interval: int = 20):
        """
        مقداردهی اولیه.

        :param optimization_interval: فاصله زمانی بین اجرای فرآیندهای بهینه‌سازی انتقال داده (بر حسب ثانیه)
        """
        self.metrics_collector = MetricsCollector()
        self.throughput_optimizer = ThroughputOptimizer()
        self.resource_balancer = ResourceBalancer()
        self.optimization_interval = optimization_interval

    async def collect_transition_metrics(self) -> Dict[str, Any]:
        """
        جمع‌آوری متریک‌های انتقال داده بین مراحل `Pipeline`.
        """
        metrics = await self.metrics_collector.collect()
        logging.info(f"📊 متریک‌های انتقال داده بین مراحل `Pipeline` جمع‌آوری شد: {metrics}")
        return metrics

    async def analyze_transition_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحلیل عملکرد انتقال داده بین مراحل `Pipeline` بر اساس متریک‌های جمع‌آوری‌شده.
        """
        performance_report = await self.throughput_optimizer.analyze(metrics)
        logging.info(f"📈 تحلیل عملکرد انتقال داده: {performance_report}")
        return performance_report

    async def optimize_transitions(self) -> None:
        """
        اجرای فرآیند بهینه‌سازی انتقال داده‌ها.
        """
        logging.info("🔄 شروع فرآیند بهینه‌سازی انتقال داده بین مراحل `Pipeline`...")

        metrics = await self.collect_transition_metrics()
        performance_report = await self.analyze_transition_performance(metrics)

        # بهینه‌سازی تخصیص منابع برای انتقال داده
        await self.resource_balancer.balance_load(performance_report)

        logging.info("✅ بهینه‌سازی انتقال داده‌ها در `Pipeline` با موفقیت انجام شد.")

    async def start_transition_optimization_loop(self) -> None:
        """
        اجرای مداوم فرآیندهای بهینه‌سازی انتقال داده در بازه‌های زمانی مشخص.
        """
        while True:
            await self.optimize_transitions()
            await asyncio.sleep(self.optimization_interval)


# مقداردهی اولیه و راه‌اندازی `TransitionOptimizer`
async def start_transition_optimizer():
    transition_optimizer = TransitionOptimizer()
    await transition_optimizer.start_transition_optimization_loop()


asyncio.create_task(start_transition_optimizer())

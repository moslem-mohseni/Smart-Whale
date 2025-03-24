import asyncio
import logging

from .throughput_optimizer import ThroughputOptimizer
from .resource_balancer import ResourceBalancer

logging.basicConfig(level=logging.INFO)


class OptimizerManager:
    def __init__(self,
                 target_throughput: int = 1000,
                 max_latency: float = 1.0,
                 adjust_interval: float = 2.0,
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 75.0,
                 gpu_threshold: float = 85.0,
                 monitor_interval: float = 2.0):
        """
        مدیریت یکپارچه‌ی بهینه‌سازی توان عملیاتی و تعادل منابع پردازشی

        :param target_throughput: حداکثر تعداد پیام‌هایی که در یک ثانیه باید پردازش شوند
        :param max_latency: حداکثر تأخیر مجاز در پردازش هر پیام (ثانیه)
        :param adjust_interval: فاصله زمانی بررسی وضعیت توان عملیاتی (ثانیه)
        :param cpu_threshold: حداکثر میزان مجاز استفاده از CPU (٪)
        :param memory_threshold: حداکثر میزان مجاز استفاده از حافظه (٪)
        :param gpu_threshold: حداکثر میزان مجاز استفاده از GPU (٪)
        :param monitor_interval: فاصله زمانی برای بررسی وضعیت منابع (ثانیه)
        """
        self.throughput_optimizer = ThroughputOptimizer(
            target_throughput=target_throughput,
            max_latency=max_latency,
            adjust_interval=adjust_interval
        )

        self.resource_balancer = ResourceBalancer(
            cpu_threshold=cpu_threshold,
            memory_threshold=memory_threshold,
            gpu_threshold=gpu_threshold,
            monitor_interval=monitor_interval
        )

    async def start_optimizers(self):
        """
        شروع بهینه‌سازی توان عملیاتی و تعادل منابع
        """
        asyncio.create_task(self.throughput_optimizer.monitor_throughput())
        asyncio.create_task(self.resource_balancer.monitor_resources())

        logging.info("✅ All optimizers have been started!")

    async def stop_optimizers(self):
        """
        توقف تمامی فرآیندهای بهینه‌سازی
        """
        await self.throughput_optimizer.stop()
        await self.resource_balancer.stop()

        logging.info("⛔ All optimizers have been stopped!")


# مقداردهی اولیه‌ی ماژول
optimizer_manager = OptimizerManager()

# راه‌اندازی پردازشگرهای بهینه‌سازی
asyncio.create_task(optimizer_manager.start_optimizers())

# API ماژول
__all__ = ["optimizer_manager", "ThroughputOptimizer", "ResourceBalancer"]

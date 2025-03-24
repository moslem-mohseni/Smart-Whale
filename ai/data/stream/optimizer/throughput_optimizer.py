import asyncio
import logging
import time
from typing import Optional, Callable
from ai.core.resource_management.monitor.resource_monitor import ResourceMonitor
from ai.core.resilience.circuit_breaker.breaker_manager import CircuitBreaker

logging.basicConfig(level=logging.INFO)


class ThroughputOptimizer:
    def __init__(self,
                 target_throughput: int = 1000,
                 max_latency: float = 1.0,
                 adjust_interval: float = 2.0,
                 scaling_function: Optional[Callable] = None):
        """
        بهینه‌ساز توان عملیاتی پردازش جریانی

        :param target_throughput: حداکثر تعداد پیام‌هایی که در یک ثانیه باید پردازش شوند
        :param max_latency: حداکثر تأخیر مجاز در پردازش هر پیام (بر حسب ثانیه)
        :param adjust_interval: فاصله زمانی بررسی وضعیت توان عملیاتی (بر حسب ثانیه)
        :param scaling_function: تابع سفارشی برای مقیاس‌پذیری در هنگام افزایش بار پردازشی
        """
        self.target_throughput = target_throughput
        self.max_latency = max_latency
        self.adjust_interval = adjust_interval
        self.scaling_function = scaling_function

        self.running = True
        self.last_check_time = time.time()
        self.processed_count = 0

        self.resource_monitor = ResourceMonitor()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10)

    async def monitor_throughput(self):
        """
        پایش توان عملیاتی پردازش داده‌ها و بهینه‌سازی در صورت نیاز
        """
        while self.running:
            elapsed_time = time.time() - self.last_check_time
            throughput = self.processed_count / elapsed_time if elapsed_time > 0 else 0

            logging.info(f"📊 Current Throughput: {throughput:.2f} msg/sec (Target: {self.target_throughput} msg/sec)")

            # بررسی مصرف منابع
            cpu_usage = await self.resource_monitor.get_cpu_usage()
            memory_usage = await self.resource_monitor.get_memory_usage()

            logging.info(f"💾 CPU Usage: {cpu_usage:.2f}%, Memory Usage: {memory_usage:.2f}%")

            # اعمال Circuit Breaker در صورت مصرف بیش از حد منابع
            if cpu_usage > 90 or memory_usage > 85:
                self.circuit_breaker.trip()
                logging.critical("🚨 Circuit Breaker Activated! Too much resource usage!")
                await asyncio.sleep(5)  # مکث برای کاهش فشار سیستم
                self.circuit_breaker.reset()

            if throughput < self.target_throughput * 0.8:
                logging.warning("⚠️ Throughput below threshold! Adjusting processing speed...")
                await self.increase_processing_rate()

            if throughput > self.target_throughput * 1.2:
                logging.warning("⚠️ Throughput exceeding safe limits! Reducing processing rate...")
                await self.decrease_processing_rate()

            # تنظیم مجدد شمارنده
            self.processed_count = 0
            self.last_check_time = time.time()
            await asyncio.sleep(self.adjust_interval)

    async def record_processed_message(self):
        """
        ثبت یک پیام پردازش‌شده برای محاسبه‌ی توان عملیاتی
        """
        self.processed_count += 1

    async def increase_processing_rate(self):
        """
        افزایش نرخ پردازش در صورت افت توان عملیاتی
        """
        logging.info("🚀 Increasing processing rate...")

        if self.scaling_function:
            self.scaling_function(scale_up=True)  # افزایش ظرفیت پردازشی

        # کاهش تأخیر پردازش برای افزایش نرخ پردازش
        await asyncio.sleep(max(0.01, self.max_latency - 0.1))

    async def decrease_processing_rate(self):
        """
        کاهش نرخ پردازش در صورت افزایش بیش از حد توان عملیاتی
        """
        logging.info("🔻 Reducing processing rate...")

        if self.scaling_function:
            self.scaling_function(scale_up=False)  # کاهش ظرفیت پردازشی

        # افزایش تأخیر پردازش برای کنترل بار پردازشگرها
        await asyncio.sleep(min(2.0, self.max_latency + 0.2))

    async def stop(self):
        """
        توقف بهینه‌سازی توان عملیاتی
        """
        self.running = False
        logging.info("⛔ Stopping Throughput Optimizer monitoring.")


async def test_throughput_optimizer():
    """
    تست عملکرد ThroughputOptimizer
    """
    optimizer = ThroughputOptimizer(target_throughput=500)

    asyncio.create_task(optimizer.monitor_throughput())

    # **شبیه‌سازی پردازش داده‌ها**
    for _ in range(2000):
        await optimizer.record_processed_message()
        await asyncio.sleep(0.002)  # شبیه‌سازی زمان پردازش

    await asyncio.sleep(5)  # اجازه می‌دهد مانیتورینگ اجرا شود
    await optimizer.stop()
    print("✅ Throughput Optimizer Test Passed!")


# اجرای تست
if __name__ == "__main__":
    asyncio.run(test_throughput_optimizer())

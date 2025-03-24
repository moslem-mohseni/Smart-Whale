import asyncio
import logging
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.optimizer.resource_optimizer import ResourceOptimizer
from core.resilience.circuit_breaker.breaker_manager import CircuitBreaker

logging.basicConfig(level=logging.INFO)


class ResourceBalancer:
    def __init__(self,
                 cpu_threshold: float = 80.0,
                 memory_threshold: float = 75.0,
                 gpu_threshold: float = 85.0,
                 monitor_interval: float = 2.0):
        """
        متعادل‌ساز منابع برای پردازش جریانی

        :param cpu_threshold: حداکثر میزان مجاز استفاده از CPU (٪)
        :param memory_threshold: حداکثر میزان مجاز استفاده از حافظه (٪)
        :param gpu_threshold: حداکثر میزان مجاز استفاده از GPU (٪)
        :param monitor_interval: فاصله زمانی برای بررسی وضعیت منابع (ثانیه)
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold
        self.monitor_interval = monitor_interval

        self.resource_monitor = ResourceMonitor()
        self.resource_optimizer = ResourceOptimizer()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=10)

        self.running = True

    async def monitor_resources(self):
        """
        پایش منابع سیستم و اعمال بهینه‌سازی در صورت نیاز
        """
        while self.running:
            cpu_usage = await self.resource_monitor.get_cpu_usage()
            memory_usage = await self.resource_monitor.get_memory_usage()
            gpu_usage = await self.resource_monitor.get_gpu_usage()

            logging.info(
                f"🔍 Resource Usage - CPU: {cpu_usage:.2f}%, Memory: {memory_usage:.2f}%, GPU: {gpu_usage:.2f}%")

            # بررسی شرایط بحرانی و اعمال Circuit Breaker
            if cpu_usage > 95 or memory_usage > 90 or gpu_usage > 90:
                logging.critical("🚨 Circuit Breaker Activated! Resource usage too high!")
                self.circuit_breaker.trip()
                await asyncio.sleep(5)  # مکث برای کاهش فشار سیستم
                self.circuit_breaker.reset()

            # اگر منابع از آستانه عبور کنند، اقدام به بهینه‌سازی
            if cpu_usage > self.cpu_threshold:
                logging.warning("⚠️ High CPU usage detected! Optimizing workload distribution...")
                await self.optimize_cpu_usage()

            if memory_usage > self.memory_threshold:
                logging.warning("⚠️ High Memory usage detected! Optimizing memory allocation...")
                await self.optimize_memory_usage()

            if gpu_usage > self.gpu_threshold:
                logging.warning("⚠️ High GPU usage detected! Redistributing GPU load...")
                await self.optimize_gpu_usage()

            await asyncio.sleep(self.monitor_interval)

    async def optimize_cpu_usage(self):
        """
        بهینه‌سازی مصرف CPU با تغییر تخصیص پردازش‌ها
        """
        logging.info("🚀 Adjusting CPU allocation...")
        await self.resource_optimizer.optimize_cpu_allocation()
        await asyncio.sleep(1)  # تنظیمات نیاز به زمان دارند

    async def optimize_memory_usage(self):
        """
        بهینه‌سازی مصرف حافظه
        """
        logging.info("🚀 Optimizing memory usage...")
        await self.resource_optimizer.optimize_memory_allocation()
        await asyncio.sleep(1)

    async def optimize_gpu_usage(self):
        """
        تنظیم تخصیص بار پردازشی روی GPU
        """
        logging.info("🚀 Adjusting GPU workload distribution...")
        await self.resource_optimizer.optimize_gpu_allocation()
        await asyncio.sleep(1)

    async def stop(self):
        """
        توقف متعادل‌سازی منابع
        """
        self.running = False
        logging.info("⛔ Stopping Resource Balancer monitoring.")


async def test_resource_balancer():
    """
    تست عملکرد ResourceBalancer
    """
    balancer = ResourceBalancer()

    asyncio.create_task(balancer.monitor_resources())

    await asyncio.sleep(10)  # اجازه می‌دهد مانیتورینگ اجرا شود
    await balancer.stop()
    print("✅ Resource Balancer Test Passed!")


# اجرای تست
if __name__ == "__main__":
    asyncio.run(test_resource_balancer())

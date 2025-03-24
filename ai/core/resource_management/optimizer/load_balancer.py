import logging
import torch
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.optimizer.resource_optimizer import ResourceOptimizer

class LoadBalancer:
    def __init__(self, resource_monitor: ResourceMonitor, resource_optimizer: ResourceOptimizer):
        """
        متعادل‌کننده بار بین CPU و GPU برای توزیع پردازش‌ها
        :param resource_monitor: نمونه‌ای از ResourceMonitor برای بررسی مصرف منابع
        :param resource_optimizer: نمونه‌ای از ResourceOptimizer برای بهینه‌سازی توزیع بار
        """
        self.resource_monitor = resource_monitor
        self.resource_optimizer = resource_optimizer
        self.logger = logging.getLogger("LoadBalancer")

    def assign_task(self, task_type: str):
        """
        تخصیص وظایف پردازشی به CPU یا GPU بر اساس میزان مصرف منابع
        :param task_type: نوع وظیفه پردازشی (مانند 'light', 'heavy')
        :return: پردازنده‌ای که وظیفه به آن تخصیص داده شده است (CPU یا GPU)
        """
        cpu_usage = self.resource_monitor.cpu_usage.collect()[0].samples[0].value
        memory_usage = self.resource_monitor.memory_usage.collect()[0].samples[0].value
        gpu_usage = self.resource_monitor.gpu_usage.collect()[0].samples[0].value if self.resource_monitor.gpu_usage else 0

        # بررسی امکان استفاده از GPU
        gpu_available = torch.cuda.is_available()

        if task_type == "heavy" and gpu_available and gpu_usage < 90:
            self.logger.info("✅ وظیفه پردازشی سنگین به GPU منتقل شد.")
            return "GPU"
        elif task_type == "light" or not gpu_available or cpu_usage < 80:
            self.logger.info("✅ وظیفه پردازشی سبک به CPU تخصیص یافت.")
            return "CPU"
        else:
            self.logger.warning("⚠️ منابع بیش از حد اشغال شده‌اند، تخصیص انجام نشد.")
            return "None"

    def rebalance_tasks(self):
        """
        بررسی وضعیت مصرف منابع و جابه‌جایی پردازش‌ها بین CPU و GPU در صورت نیاز
        """
        suggestions = self.resource_optimizer.analyze_and_optimize()

        if "reduce_cpu_tasks" in suggestions:
            self.logger.info("🔄 برخی وظایف پردازشی از CPU به GPU منتقل می‌شوند.")

        if "optimize_gpu_tasks" in suggestions:
            self.logger.info("🔄 برخی وظایف سنگین از GPU به CPU منتقل می‌شوند.")

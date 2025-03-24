from typing import Dict, Any
from core.resource_management.allocator.cpu_allocator import CPUAllocator
from core.resource_management.allocator.memory_allocator import MemoryAllocator
from core.resource_management.allocator.gpu_allocator import GPUAllocator
from core.resource_management.monitor.resource_monitor import ResourceMonitor
from core.resource_management.monitor.threshold_manager import ThresholdManager

class ResourceAllocator:
    """
    تخصیص منابع پردازشی برای اجرای بهینه وظایف.
    """

    def __init__(self, max_cpu_usage: float = 0.8, max_memory_usage: float = 0.75):
        """
        مقداردهی اولیه تخصیص منابع.

        :param max_cpu_usage: حداکثر میزان مجاز استفاده از CPU (نسبت از ۰ تا ۱)
        :param max_memory_usage: حداکثر میزان مجاز استفاده از حافظه (نسبت از ۰ تا ۱)
        """
        self.cpu_allocator = CPUAllocator()
        self.memory_allocator = MemoryAllocator()
        self.gpu_allocator = GPUAllocator()
        self.resource_monitor = ResourceMonitor()
        self.threshold_manager = ThresholdManager()
        self.max_cpu_usage = max_cpu_usage
        self.max_memory_usage = max_memory_usage

    def get_system_resources(self) -> Dict[str, Any]:
        """
        بررسی وضعیت فعلی منابع سیستم.

        :return: دیکشنری شامل میزان مصرف CPU، حافظه و تعداد پردازنده‌های موجود.
        """
        return self.resource_monitor.get_system_resources()

    def can_allocate_resources(self, cpu_demand: float, memory_demand: float, gpu_demand: float = 0.0) -> bool:
        """
        بررسی امکان تخصیص منابع برای یک وظیفه.

        :param cpu_demand: میزان مورد نیاز CPU (نسبت از ۰ تا ۱)
        :param memory_demand: میزان مورد نیاز حافظه (نسبت از ۰ تا ۱)
        :param gpu_demand: میزان مورد نیاز GPU (نسبت از ۰ تا ۱)
        :return: `True` اگر منابع کافی برای تخصیص وجود داشته باشد، در غیر این صورت `False`
        """
        resources = self.get_system_resources()

        return (
            resources["cpu_usage"] + cpu_demand <= self.max_cpu_usage and
            resources["memory_usage"] + memory_demand <= self.max_memory_usage and
            self.gpu_allocator.can_allocate(gpu_demand)
        )

    def allocate_resources(self, cpu_demand: float, memory_demand: float, gpu_demand: float = 0.0) -> Dict[str, Any]:
        """
        تخصیص منابع پردازشی به یک وظیفه، در صورت امکان.

        :param cpu_demand: میزان مورد نیاز CPU (نسبت از ۰ تا ۱)
        :param memory_demand: میزان مورد نیاز حافظه (نسبت از ۰ تا ۱)
        :param gpu_demand: میزان مورد نیاز GPU (نسبت از ۰ تا ۱)
        :return: دیکشنری شامل وضعیت تخصیص منابع
        """
        if self.threshold_manager.is_threshold_breached():
            return {
                "status": "denied",
                "reason": "System resources exceeded safe threshold"
            }

        if self.can_allocate_resources(cpu_demand, memory_demand, gpu_demand):
            self.cpu_allocator.allocate(cpu_demand)
            self.memory_allocator.allocate(memory_demand)
            self.gpu_allocator.allocate(gpu_demand)

            return {
                "status": "allocated",
                "cpu_assigned": cpu_demand,
                "memory_assigned": memory_demand,
                "gpu_assigned": gpu_demand
            }
        else:
            return {
                "status": "denied",
                "reason": "Insufficient system resources"
            }

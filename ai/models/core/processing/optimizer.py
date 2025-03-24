from typing import Dict, Any

class ProcessingOptimizer:
    """
    ماژول بهینه‌سازی تخصیص منابع و مسیرهای پردازشی برای مدیریت هوشمند وظایف.
    """

    def __init__(self, max_cpu: float = 100.0, max_memory: int = 8192, max_gpu: float = 50.0):
        """
        مقداردهی اولیه برای مدیریت منابع پردازشی.
        :param max_cpu: حداکثر میزان CPU قابل تخصیص (درصدی).
        :param max_memory: حداکثر میزان حافظه (MB).
        :param max_gpu: حداکثر میزان GPU قابل تخصیص (درصدی).
        """
        self.max_cpu = max_cpu
        self.max_memory = max_memory
        self.max_gpu = max_gpu
        self.allocated_resources: Dict[str, Dict[str, float]] = {}

    def allocate_resources(self, complexity: str) -> bool:
        """
        تخصیص منابع پردازشی بر اساس سطح پیچیدگی وظیفه.
        :param complexity: سطح پیچیدگی وظیفه (quick, normal, deep).
        :return: موفقیت یا شکست تخصیص منابع.
        """
        resource_requirements = {
            "quick": {"cpu": 10.0, "memory": 256, "gpu": 2.0},
            "normal": {"cpu": 25.0, "memory": 1024, "gpu": 5.0},
            "deep": {"cpu": 50.0, "memory": 4096, "gpu": 20.0}
        }

        if complexity not in resource_requirements:
            return False  # سطح پیچیدگی نامعتبر است

        required = resource_requirements[complexity]
        if self.get_available_resources()["cpu"] >= required["cpu"] and \
           self.get_available_resources()["memory"] >= required["memory"] and \
           self.get_available_resources()["gpu"] >= required["gpu"]:
            self.allocated_resources[complexity] = required
            return True

        return False  # منابع کافی برای پردازش وجود ندارد

    def release_resources(self, complexity: str):
        """
        آزادسازی منابع تخصیص‌یافته پس از پایان پردازش.
        :param complexity: سطح پیچیدگی وظیفه‌ای که منابعش آزاد می‌شود.
        """
        if complexity in self.allocated_resources:
            del self.allocated_resources[complexity]

    def get_available_resources(self) -> Dict[str, float]:
        """
        دریافت میزان منابع پردازشی باقی‌مانده.
        :return: دیکشنری شامل اطلاعات منابع در دسترس.
        """
        allocated_cpu = sum(res["cpu"] for res in self.allocated_resources.values())
        allocated_memory = sum(res["memory"] for res in self.allocated_resources.values())
        allocated_gpu = sum(res["gpu"] for res in self.allocated_resources.values())

        return {
            "cpu": self.max_cpu - allocated_cpu,
            "memory": self.max_memory - allocated_memory,
            "gpu": self.max_gpu - allocated_gpu
        }

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت تخصیص منابع پردازشی.
        :return: دیکشنری شامل اطلاعات منابع تخصیص‌یافته و باقی‌مانده.
        """
        return {
            "total_cpu": self.max_cpu,
            "allocated_cpu": sum(res["cpu"] for res in self.allocated_resources.values()),
            "available_cpu": self.get_available_resources()["cpu"],
            "total_memory": self.max_memory,
            "allocated_memory": sum(res["memory"] for res in self.allocated_resources.values()),
            "available_memory": self.get_available_resources()["memory"],
            "total_gpu": self.max_gpu,
            "allocated_gpu": sum(res["gpu"] for res in self.allocated_resources.values()),
            "available_gpu": self.get_available_resources()["gpu"],
            "allocated_resources": self.allocated_resources
        }

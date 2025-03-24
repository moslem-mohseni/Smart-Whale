from typing import Dict


class ResourceCoordinator:
    """
    مدیریت تخصیص منابع پردازشی بین مدل‌های هوش مصنوعی
    """

    def __init__(self):
        self.resource_pool: Dict[str, Dict[str, float]] = {}  # منابع تخصیص‌یافته به هر مدل

    def allocate_resources(self, model_name: str, cpu: float, memory: float) -> bool:
        """
        تخصیص منابع پردازشی به یک مدل مشخص‌شده
        :param model_name: نام مدل
        :param cpu: مقدار پردازنده (واحد مجازی CPU)
        :param memory: مقدار حافظه (GB)
        :return: مقدار بولین که نشان می‌دهد تخصیص موفقیت‌آمیز بوده یا خیر
        """
        if model_name in self.resource_pool:
            return False  # منابع قبلاً تخصیص داده شده است

        self.resource_pool[model_name] = {"cpu": cpu, "memory": memory}
        return True

    def release_resources(self, model_name: str) -> bool:
        """
        آزادسازی منابع پردازشی یک مدل پس از اتمام پردازش
        :param model_name: نام مدل
        :return: مقدار بولین که نشان می‌دهد آزادسازی موفقیت‌آمیز بوده یا خیر
        """
        if model_name in self.resource_pool:
            del self.resource_pool[model_name]
            return True
        return False

    def get_allocated_resources(self, model_name: str) -> Dict[str, float]:
        """
        دریافت اطلاعات منابع تخصیص‌یافته به یک مدل خاص
        :param model_name: نام مدل
        :return: دیکشنری شامل مقدار پردازنده و حافظه اختصاص‌یافته یا مقدار پیش‌فرض
        """
        return self.resource_pool.get(model_name, {"cpu": 0.0, "memory": 0.0})


# نمونه استفاده از ResourceCoordinator برای تست
if __name__ == "__main__":
    coordinator = ResourceCoordinator()
    allocated = coordinator.allocate_resources("model_a", 2.0, 4.0)
    print(f"Resources allocated to model_a: {allocated}")
    print(f"Allocated Resources: {coordinator.get_allocated_resources('model_a')}")
    released = coordinator.release_resources("model_a")
    print(f"Resources released from model_a: {released}")

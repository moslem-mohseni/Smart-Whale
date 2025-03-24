from typing import Dict, Any


class ResourceManager:
    """
    این کلاس مسئول مدیریت تخصیص منابع به دسته‌های پردازشی است.
    """

    def __init__(self, max_cpu: float = 0.9, max_memory: float = 0.8):
        """
        مقداردهی اولیه با محدودیت‌های منابع.
        """
        self.max_cpu = max_cpu  # حداکثر میزان استفاده از CPU
        self.max_memory = max_memory  # حداکثر میزان استفاده از حافظه

    def allocate_resources(self, batch_data: Dict[str, Any]) -> Dict[str, float]:
        """
        تخصیص منابع پردازشی به دسته داده.
        """
        batch_size = batch_data.get("batch_size", 1)
        cpu_usage = min(self.max_cpu, 0.1 + (batch_size * 0.005))
        memory_usage = min(self.max_memory, 0.2 + (batch_size * 0.01))

        return {
            "cpu": round(cpu_usage, 2),
            "memory": round(memory_usage, 2)
        }

    def release_resources(self, used_resources: Dict[str, float]) -> None:
        """
        آزادسازی منابع پس از اتمام پردازش.
        """
        print(f"🔄 آزادسازی منابع: CPU={used_resources['cpu']} | Memory={used_resources['memory']}")

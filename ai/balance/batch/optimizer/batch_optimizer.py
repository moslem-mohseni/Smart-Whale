from typing import List, Dict, Any


class BatchOptimizer:
    """
    این کلاس مسئول بهینه‌سازی ترکیب دسته‌ها و تخصیص منابع بهینه برای پردازش دسته‌ای است.
    """

    def __init__(self, max_cpu: float = 0.8, max_memory: float = 0.7):
        """
        مقداردهی اولیه با محدودیت‌های منابع.
        """
        self.max_cpu = max_cpu  # حداکثر درصد استفاده از CPU
        self.max_memory = max_memory  # حداکثر درصد استفاده از حافظه

    def calculate_optimal_resources(self, batch_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        محاسبه منابع مورد نیاز برای پردازش دسته داده.
        """
        batch_size = len(batch_data)
        cpu_usage = min(self.max_cpu, 0.1 + (batch_size * 0.005))
        memory_usage = min(self.max_memory, 0.2 + (batch_size * 0.01))

        return {
            "cpu": round(cpu_usage, 2),
            "memory": round(memory_usage, 2)
        }

    def optimize_batch_composition(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        بهینه‌سازی ترکیب دسته‌ها برای پردازش سریع‌تر.
        """
        batch_data.sort(key=lambda x: x.get("priority", 1), reverse=True)  # مرتب‌سازی بر اساس اولویت
        return batch_data

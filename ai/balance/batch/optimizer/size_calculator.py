from typing import List, Dict, Any


class SizeCalculator:
    """
    این کلاس مسئول محاسبه اندازه‌ی بهینه دسته‌های پردازشی است.
    """

    def __init__(self, base_size: int = 50, max_size: int = 200, scaling_factor: float = 1.2):
        """
        مقداردهی اولیه با اندازه‌های پیش‌فرض دسته‌ها.
        """
        self.base_size = base_size  # اندازه‌ی پایه‌ی دسته‌ها
        self.max_size = max_size  # حداکثر اندازه‌ی مجاز دسته
        self.scaling_factor = scaling_factor  # ضریب افزایش دسته بر اساس پردازش‌های قبلی

    def calculate_batch_size(self, previous_batches: List[Dict[str, Any]]) -> int:
        """
        محاسبه اندازه‌ی بهینه دسته با استفاده از داده‌های پردازش قبلی.
        """
        if not previous_batches:
            return self.base_size  # در صورت نبود داده قبلی، مقدار پیش‌فرض استفاده می‌شود

        avg_time = sum(batch["processing_time"] for batch in previous_batches) / len(previous_batches)
        optimal_size = int(self.base_size * (self.scaling_factor / (avg_time + 1)))

        return min(self.max_size, max(self.base_size, optimal_size))

"""
BatchOptimizer Module
-----------------------
این فایل مسئول بهینه‌سازی دسته‌های آموزشی (batches) برای فرآیند آموزش مدل‌های خودآموزی است.
هدف اصلی این ماژول کاهش سربار محاسباتی (مثلاً کاهش padding در توالی‌های متنی) از طریق گروه‌بندی نمونه‌های مشابه (به‌ویژه از نظر طول) می‌باشد.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند (مانند مرتب‌سازی بر اساس طول و سپس دسته‌بندی بهینه) پیاده‌سازی شده است.
"""

import logging
from abc import ABC
from typing import Any, Dict, List, Optional

from ..base.base_component import BaseComponent


class BatchOptimizer(BaseComponent, ABC):
    """
    BatchOptimizer مسئول تقسیم داده‌های آموزشی به دسته‌های بهینه است.

    امکانات:
      - مرتب‌سازی نمونه‌های آموزشی بر اساس ویژگی کلیدی (مثلاً طول توالی) برای کاهش سربار padding.
      - گروه‌بندی نمونه‌های مشابه به صورت دسته‌ای با حداکثر اندازه مشخص.
      - قابلیت پیکربندی اندازه دسته (max_batch_size) و ویژگی کلیدی مرتب‌سازی (sort_key).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه BatchOptimizer.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "max_batch_size": حداکثر تعداد نمونه در هر دسته (پیش‌فرض: 32)
                - "sort_key": کلید مورد استفاده برای مرتب‌سازی (پیش‌فرض: "length")
                - "shuffle": آیا قبل از دسته‌بندی داده‌ها را به صورت تصادفی مخلوط کند (پیش‌فرض: False)
        """
        super().__init__(component_type="batch_optimizer", config=config)
        self.logger = logging.getLogger("BatchOptimizer")
        self.max_batch_size = int(self.config.get("max_batch_size", 32))
        self.sort_key = self.config.get("sort_key", "length")
        self.shuffle = bool(self.config.get("shuffle", False))
        self.logger.info(f"[BatchOptimizer] Initialized with max_batch_size={self.max_batch_size}, "
                         f"sort_key='{self.sort_key}', shuffle={self.shuffle}")

    def optimize_batches(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        تقسیم داده‌های آموزشی به دسته‌های بهینه با استفاده از مرتب‌سازی بر اساس ویژگی کلیدی (مثلاً طول).

        Args:
            data (List[Dict[str, Any]]): لیستی از نمونه‌های آموزشی؛ هر نمونه یک دیکشنری است
                                         که باید شامل کلیدی مطابق با sort_key باشد (مثلاً "length").

        Returns:
            List[List[Dict[str, Any]]]: لیستی از دسته‌های بهینه؛ هر دسته یک لیست از نمونه‌هاست.
        """
        try:
            if not data:
                self.logger.warning("[BatchOptimizer] No data provided for batch optimization.")
                return []

            # در صورت فعال بودن shuffle، داده‌ها را تصادفی مخلوط می‌کنیم.
            if self.shuffle:
                from random import shuffle
                shuffle(data)
                self.logger.debug("[BatchOptimizer] Data shuffled before batch optimization.")

            # مرتب‌سازی داده‌ها بر اساس sort_key در صورت موجود بودن آن در نمونه‌ها
            if all(self.sort_key in item for item in data):
                data.sort(key=lambda item: item[self.sort_key])
                self.logger.debug(f"[BatchOptimizer] Data sorted by key '{self.sort_key}'.")
            else:
                self.logger.warning(
                    f"[BatchOptimizer] Not all data items contain the key '{self.sort_key}'. Skipping sorting.")

            # تقسیم داده‌ها به دسته‌های با حداکثر اندازه max_batch_size
            batches = [data[i:i + self.max_batch_size] for i in range(0, len(data), self.max_batch_size)]
            self.logger.info(f"[BatchOptimizer] Optimized into {len(batches)} batches from {len(data)} samples.")
            self.increment_metric("batches_optimized")
            return batches
        except Exception as e:
            self.logger.error(f"[BatchOptimizer] Error during batch optimization: {str(e)}")
            self.record_error_metric()
            return []


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # نمونه داده‌های آموزشی؛ فرض کنید هر نمونه دارای کلید "length" است.
    sample_data = [
        {"id": 1, "text": "Short sentence.", "length": 3},
        {"id": 2, "text": "This is a longer sentence for testing.", "length": 7},
        {"id": 3, "text": "Medium length sentence here.", "length": 5},
        {"id": 4, "text": "Another short one.", "length": 3},
        {"id": 5, "text": "A very very long sentence that exceeds expectations for batch grouping.", "length": 12},
        {"id": 6, "text": "Tiny.", "length": 1},
        {"id": 7, "text": "An example sentence.", "length": 4},
        {"id": 8, "text": "More examples to test batch optimizer.", "length": 6}
    ]

    optimizer = BatchOptimizer(config={"max_batch_size": 3, "sort_key": "length", "shuffle": False})
    batches = optimizer.optimize_batches(sample_data)
    for idx, batch in enumerate(batches):
        print(f"Batch {idx + 1}:")
        for item in batch:
            print(item)

from typing import List, Dict, Any


class BatchSplitter:
    """
    این کلاس وظیفه تقسیم دسته‌های بزرگ به دسته‌های کوچک‌تر را بر عهده دارد.
    """

    def __init__(self, max_batch_size: int = 100):
        """
        مقداردهی اولیه با اندازه پیش‌فرض دسته‌های پردازشی.
        """
        self.max_batch_size = max_batch_size

    def split_large_batches(self, batch_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        تقسیم دسته‌های بزرگ به دسته‌های کوچک‌تر بر اساس اندازه تعیین‌شده.
        """
        if len(batch_data) <= self.max_batch_size:
            return [batch_data]  # نیازی به تقسیم نیست

        return [batch_data[i:i + self.max_batch_size] for i in range(0, len(batch_data), self.max_batch_size)]

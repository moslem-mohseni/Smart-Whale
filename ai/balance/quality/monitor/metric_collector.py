from typing import List, Dict, Any


class MetricCollector:
    """
    این کلاس مسئول جمع‌آوری متریک‌های کیفیت داده‌ها و پردازش‌ها است.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تعریف ساختار ذخیره‌سازی متریک‌ها.
        """
        self.metrics = []

    def collect_metric(self, data_id: str, quality_score: float, processing_time: float) -> None:
        """
        ذخیره یک متریک جدید از کیفیت داده‌ها.
        """
        self.metrics.append({
            "data_id": data_id,
            "quality_score": quality_score,
            "processing_time": processing_time
        })

    def get_average_metrics(self) -> Dict[str, float]:
        """
        محاسبه میانگین کیفیت داده‌ها و زمان پردازش.
        """
        if not self.metrics:
            return {"average_quality": 1.0, "average_processing_time": 0.0}

        avg_quality = sum(m["quality_score"] for m in self.metrics) / len(self.metrics)
        avg_time = sum(m["processing_time"] for m in self.metrics) / len(self.metrics)

        return {
            "average_quality": round(avg_quality, 2),
            "average_processing_time": round(avg_time, 2)
        }

    def reset_metrics(self) -> None:
        """
        پاک‌سازی متریک‌های ذخیره‌شده.
        """
        self.metrics.clear()

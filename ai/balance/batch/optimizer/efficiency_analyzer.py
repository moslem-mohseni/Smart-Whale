from typing import List, Dict, Any


class EfficiencyAnalyzer:
    """
    این کلاس مسئول تحلیل و ارزیابی کارایی پردازش دسته‌ها است.
    """

    def __init__(self, threshold: float = 1.5):
        """
        مقداردهی اولیه با آستانه‌ی عملکردی برای تشخیص ناکارآمدی.
        """
        self.threshold = threshold  # آستانه‌ی شناسایی دسته‌های ناکارآمد

    def analyze_efficiency(self, batch_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        بررسی و تحلیل دسته‌هایی که بهینه پردازش نشده‌اند.
        """
        inefficient_batches = []

        for batch in batch_results:
            efficiency_ratio = batch["processing_time"] / (batch["batch_size"] + 1)

            if efficiency_ratio > self.threshold:
                inefficient_batches.append({
                    "batch_id": batch["batch_id"],
                    "efficiency_ratio": round(efficiency_ratio, 2),
                    "suggested_optimization": "Reduce batch size or allocate more resources"
                })

        return inefficient_batches

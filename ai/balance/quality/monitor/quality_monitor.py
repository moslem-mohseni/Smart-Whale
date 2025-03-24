from typing import List, Dict, Any


class QualityMonitor:
    """
    این کلاس مسئول پایش کیفیت داده‌ها و پردازش‌های انجام‌شده است.
    """

    def __init__(self, quality_threshold: float = 0.8):
        """
        مقداردهی اولیه با تنظیم آستانه‌ی کیفیت داده‌ها.
        """
        self.quality_threshold = quality_threshold  # حداقل میزان کیفیت قابل قبول

    def evaluate_data_quality(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        بررسی کیفیت داده‌های ورودی و شناسایی موارد نامعتبر.
        """
        low_quality_data = []

        for data in data_batch:
            quality_score = data.get("quality_score", 1.0)

            if quality_score < self.quality_threshold:
                low_quality_data.append({
                    "data_id": data.get("data_id"),
                    "quality_score": quality_score,
                    "issue": "Quality below threshold"
                })

        return low_quality_data

    def monitor_trends(self, historical_quality: List[float]) -> Dict[str, Any]:
        """
        تحلیل روندهای کیفیت داده‌ها و تشخیص کاهش کیفیت.
        """
        avg_quality = sum(historical_quality) / len(historical_quality) if historical_quality else 1.0
        trend_status = "Stable" if avg_quality >= self.quality_threshold else "Declining"

        return {
            "average_quality": round(avg_quality, 2),
            "trend": trend_status
        }

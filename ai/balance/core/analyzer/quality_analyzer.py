from typing import Dict, Any, List
from collections import defaultdict

class QualityAnalyzer:
    """
    تحلیل‌گر کیفیت داده‌ها برای بررسی میزان صحت و کیفیت داده‌های مصرف‌شده در مدل‌ها.
    """

    def __init__(self):
        # ذخیره کیفیت داده‌ها برای هر مدل
        self.data_quality = defaultdict(lambda: {"total_records": 0, "invalid_records": 0, "noise_level": 0.0})

    def log_data_quality(self, model_id: str, total_records: int, invalid_records: int, noise_level: float) -> None:
        """
        ثبت کیفیت داده‌های استفاده‌شده در یک مدل.

        :param model_id: شناسه مدل
        :param total_records: تعداد کل داده‌های پردازش‌شده
        :param invalid_records: تعداد داده‌های نامعتبر
        :param noise_level: سطح نویز در داده (عدد بین ۰ تا ۱)
        """
        self.data_quality[model_id]["total_records"] += total_records
        self.data_quality[model_id]["invalid_records"] += invalid_records
        self.data_quality[model_id]["noise_level"] = max(self.data_quality[model_id]["noise_level"], noise_level)

    def analyze_quality(self) -> Dict[str, Any]:
        """
        بررسی کیفیت داده‌ها و شناسایی مدل‌هایی که داده‌های آن‌ها کیفیت پایینی دارد.

        :return: دیکشنری شامل وضعیت کیفیت داده‌ها و مدل‌های دارای مشکل
        """
        if not self.data_quality:
            return {"status": "No data recorded"}

        quality_issues = {
            model_id: data for model_id, data in self.data_quality.items()
            if data["invalid_records"] / max(data["total_records"], 1) > 0.1 or data["noise_level"] > 0.5
        }

        return {
            "total_models_evaluated": len(self.data_quality),
            "quality_issues_detected": len(quality_issues),
            "problematic_models": quality_issues
        }

    def detect_low_quality_models(self, invalid_threshold: float = 0.1, noise_threshold: float = 0.5) -> List[str]:
        """
        شناسایی مدل‌هایی که داده‌های ورودی آن‌ها کیفیت پایینی دارد.

        :param invalid_threshold: حد آستانه داده‌های نامعتبر (پیش‌فرض ۱۰٪)
        :param noise_threshold: حد آستانه نویز در داده‌ها (پیش‌فرض ۰.۵)
        :return: لیست مدل‌هایی که داده‌های آن‌ها کیفیت پایینی دارد.
        """
        analysis = self.analyze_quality()
        return list(analysis["problematic_models"].keys()) if "problematic_models" in analysis else []

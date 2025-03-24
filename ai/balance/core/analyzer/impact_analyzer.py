from typing import Dict, Any, List
from collections import defaultdict

class ImpactAnalyzer:
    """
    تحلیل‌گر تأثیر تصمیمات پردازشی بر عملکرد مدل‌ها و منابع داده‌ای.
    """

    def __init__(self):
        # ذخیره اطلاعات مربوط به تأثیر داده‌ها و منابع بر مدل‌ها
        self.impact_data = defaultdict(lambda: {"performance_change": 0.0, "resource_usage": 0, "quality_shift": 0.0})

    def log_impact(self, model_id: str, performance_change: float, resource_usage: int, quality_shift: float) -> None:
        """
        ثبت تأثیر تغییرات در داده‌ها و منابع بر عملکرد مدل.

        :param model_id: شناسه مدل
        :param performance_change: تغییر عملکرد مدل (به درصد، مقدار مثبت به معنای بهبود است)
        :param resource_usage: میزان مصرف منابع پردازشی (برحسب واحد پردازشی)
        :param quality_shift: تغییر کیفیت داده‌های مصرف‌شده (عدد بین ۰ تا ۱)
        """
        self.impact_data[model_id]["performance_change"] += performance_change
        self.impact_data[model_id]["resource_usage"] += resource_usage
        self.impact_data[model_id]["quality_shift"] += quality_shift

    def analyze_impact(self) -> Dict[str, Any]:
        """
        بررسی تأثیرات تصمیمات و تحلیل عملکرد مدل‌ها و منابع.

        :return: دیکشنری شامل تحلیل تأثیرات و مدل‌های تحت تأثیر
        """
        if not self.impact_data:
            return {"status": "No impact data recorded"}

        significant_impacts = {
            model_id: data for model_id, data in self.impact_data.items()
            if abs(data["performance_change"]) > 5 or data["resource_usage"] > 1000 or abs(data["quality_shift"]) > 0.2
        }

        return {
            "total_models_analyzed": len(self.impact_data),
            "significant_impacts_detected": len(significant_impacts),
            "affected_models": significant_impacts
        }

    def detect_high_impact_models(self, performance_threshold: float = 5.0, resource_threshold: int = 1000, quality_threshold: float = 0.2) -> List[str]:
        """
        شناسایی مدل‌هایی که بیشترین تأثیر را از تغییرات داده‌ای و منابع دریافت کرده‌اند.

        :param performance_threshold: آستانه تغییر عملکرد (۵٪ به‌صورت پیش‌فرض)
        :param resource_threshold: آستانه مصرف منابع (۱۰۰۰ واحد)
        :param quality_threshold: آستانه تغییر کیفیت داده (۰.۲)
        :return: لیست مدل‌هایی که بیشترین تأثیر را پذیرفته‌اند.
        """
        analysis = self.analyze_impact()
        return list(analysis["affected_models"].keys()) if "affected_models" in analysis else []

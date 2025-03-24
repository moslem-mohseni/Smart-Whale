from typing import Dict, Any
import numpy as np
from ..metrics.performance_metrics import PerformanceMetrics


class QualityOptimizer:
    """
    ماژول بهینه‌سازی کیفیت پردازش مدل‌های فدراسیونی بر اساس متریک‌های ارزیابی و بازخورد کاربران.
    """

    def __init__(self):
        """
        مقداردهی اولیه و تنظیم متغیرهای بهینه‌سازی.
        """
        self.performance_metrics = PerformanceMetrics()
        self.optimization_history: Dict[str, Dict[str, Any]] = {}

    def optimize_model_quality(self, model_id: str, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        تحلیل متریک‌های کیفیت و بهینه‌سازی پارامترهای مدل.
        :param model_id: شناسه مدل فدراسیونی.
        :param quality_metrics: داده‌های ارزیابی کیفیت مدل شامل دقت، انسجام و سازگاری.
        :return: دیکشنری شامل تنظیمات بهینه‌سازی اعمال‌شده.
        """
        # تحلیل متریک‌های کیفیت
        adjusted_params = self._analyze_quality_metrics(quality_metrics)

        # ذخیره تاریخچه بهینه‌سازی برای مدل
        self.optimization_history[model_id] = {
            "quality_metrics": quality_metrics,
            "adjusted_params": adjusted_params
        }

        return adjusted_params

    def _analyze_quality_metrics(self, quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        پردازش داده‌های متریک‌های کیفیت و استخراج پیشنهادات بهینه‌سازی.
        :param quality_metrics: داده‌های کیفیت پردازش.
        :return: دیکشنری شامل تنظیمات پیشنهادی بهینه‌سازی.
        """
        adjustments = {}

        # کاهش نرخ یادگیری در صورت افت کیفیت
        if quality_metrics["semantic_score"] < 0.8:
            adjustments["learning_rate"] = "reduce"

        # افزایش تنظیمات تطبیقی در صورت ناهماهنگی پاسخ‌ها
        if quality_metrics["coherence_score"] < 0.75:
            adjustments["adaptive_tuning"] = "increase"

        # بهبود پردازش زمینه در صورت عدم تطبیق با مکالمه
        if quality_metrics["context_score"] < 0.7:
            adjustments["context_weight"] = "increase"

        return adjustments

    def get_optimization_history(self, model_id: str) -> Dict[str, Any]:
        """
        دریافت تاریخچه بهینه‌سازی یک مدل.
        :param model_id: شناسه مدل.
        :return: دیکشنری شامل تنظیمات و داده‌های بهینه‌سازی قبلی مدل.
        """
        return self.optimization_history.get(model_id, {"error": "No optimization history available."})

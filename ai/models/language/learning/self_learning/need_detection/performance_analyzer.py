"""
PerformanceAnalyzer Module
---------------------------
این فایل مسئول تشخیص نیازهای یادگیری بر اساس تحلیل عملکرد مدل است.
کلاس PerformanceAnalyzer از NeedDetectorBase ارث می‌برد و با بررسی متریک‌های کلیدی عملکرد (accuracy، latency، f1_score و ...)
در صورت پایین‌تر بودن از آستانه‌های تعیین‌شده، نیازهای یادگیری مرتبط را شناسایی می‌کند.

نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا.
"""

import logging
from typing import Dict, Any, List, Optional

from .need_detector_base import NeedDetectorBase


class PerformanceAnalyzer(NeedDetectorBase):
    """
    PerformanceAnalyzer با تحلیل متریک‌های عملکردی مدل، نیازهای یادگیری مرتبط را شناسایی می‌کند.

    زیرکلاس NeedDetectorBase بوده و متد detect_needs را پیاده‌سازی می‌کند.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه PerformanceAnalyzer.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختصاصی تحلیل عملکرد. می‌تواند شامل آستانه‌ها و متریک‌های کلیدی باشد.
        """
        super().__init__(config=config)
        self.logger = logging.getLogger("PerformanceAnalyzer")
        # مثال: تعریف آستانه‌های پیش‌فرض
        self.default_thresholds = {
            "accuracy": 0.8,
            "latency": 2.0,
            "f1_score": 0.75
        }
        self.logger.info(f"[PerformanceAnalyzer] Initialized with config: {self.config}")

    def detect_needs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        شناسایی نیازهای یادگیری بر اساس متریک‌های عملکردی مدل.

        Args:
            input_data (Dict[str, Any]): دیکشنری شامل اطلاعاتی مانند:
                {
                  "evaluation_metrics": {
                      "accuracy": 0.78,
                      "latency": 2.5,
                      "f1_score": 0.72,
                      ...
                  },
                  "model_id": "model_123",
                  ...
                }

        Returns:
            List[Dict[str, Any]]: لیستی از نیازهای یادگیری شناسایی‌شده. هر آیتم به صورت:
                {
                  "need_type": "PERFORMANCE",
                  "metric": "accuracy",
                  "current_value": 0.78,
                  "threshold": 0.8,
                  "importance": "HIGH",
                  "details": "Accuracy below threshold"
                }
        """
        if not self.validate_input(input_data):
            self.logger.warning("[PerformanceAnalyzer] Invalid input data. No needs detected.")
            return []

        # استخراج متریک‌های عملکردی
        evaluation_metrics = input_data.get("evaluation_metrics", {})
        if not evaluation_metrics:
            self.logger.debug("[PerformanceAnalyzer] No evaluation_metrics found in input_data.")
            return []

        # ساختار خروجی
        detected_needs = []

        # بررسی هر متریک با آستانه مربوطه
        for metric_name, threshold_value in self.default_thresholds.items():
            current_value = evaluation_metrics.get(metric_name)
            if current_value is None:
                # ممکن است برخی متریک‌ها در evaluation_metrics نباشند
                continue

            # مثال: اگر accuracy < threshold یا latency > threshold
            # برای accuracy و f1_score => مقدار باید بالاتر از آستانه باشد
            # برای latency => مقدار باید پایین‌تر از آستانه باشد
            need_detected = False
            if metric_name == "latency":
                if current_value > threshold_value:
                    need_detected = True
            else:
                if current_value < threshold_value:
                    need_detected = True

            if need_detected:
                need_info = {
                    "need_type": "PERFORMANCE",
                    "metric": metric_name,
                    "current_value": current_value,
                    "threshold": threshold_value,
                    "importance": "HIGH",
                    "details": f"{metric_name.capitalize()} does not meet the threshold."
                }
                detected_needs.append(need_info)
                self.logger.debug(f"[PerformanceAnalyzer] Detected need: {need_info}")

        # انجام پردازش نهایی روی نیازها
        final_needs = self.post_process_needs(detected_needs)
        return final_needs

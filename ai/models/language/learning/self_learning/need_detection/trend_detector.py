"""
TrendDetector Module
----------------------
این فایل مسئول تشخیص روندهای داغ و جدید در داده‌های ورودی (مانند درخواست‌ها یا موضوعات) است.
این کلاس با استفاده از تحلیل فراوانی و رشد موضوعات، موضوعاتی را که به‌طور ناگهانی محبوب می‌شوند،
شناسایی می‌کند. این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from collections import Counter
from typing import Dict, Any, List, Optional

from .need_detector_base import NeedDetectorBase


class TrendDetector(NeedDetectorBase):
    """
    TrendDetector برای شناسایی روندهای داغ و جدید در داده‌های ورودی طراحی شده است.

    ورودی مورد انتظار:
      {
          "queries": List[str],       # لیستی از موضوعات/عبارات درخواست شده در یک بازه زمانی مشخص
          "baseline": Optional[Dict[str, float]]  # (اختیاری) دیکشنری شامل فراوانی‌های پیشین برای مقایسه (موضوع: فراوانی)
      }

    خروجی:
      لیستی از دیکشنری‌ها که هر کدام شامل اطلاعات روند به صورت زیر هستند:
          {
              "need_type": "TREND",
              "topic": <str>,
              "current_frequency": <int>,
              "baseline_frequency": <Optional[float]>,
              "growth_factor": <Optional[float]>,  # اگر baseline موجود باشد
              "details": <str>
          }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه TrendDetector.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختصاصی برای تنظیم آستانه‌های تشخیص روند.
                                              می‌تواند شامل "trend_threshold" (حداقل فراوانی) و
                                              "growth_factor_threshold" (حداقل ضریب رشد) باشد.
        """
        super().__init__(config=config)
        self.logger = logging.getLogger("TrendDetector")
        # آستانه پیش‌فرض: اگر تعداد درخواست‌های یک موضوع از این مقدار بالاتر باشد (بدون baseline)
        self.default_trend_threshold = float(self.config.get("trend_threshold", 5))
        # در صورت وجود baseline، ضریب رشد حداقل مورد نیاز (مثلاً 1.5 برابر رشد نسبت به دوره قبل)
        self.growth_factor_threshold = float(self.config.get("growth_factor_threshold", 1.5))
        self.logger.info(f"[TrendDetector] Initialized with trend_threshold={self.default_trend_threshold} and "
                         f"growth_factor_threshold={self.growth_factor_threshold}")

    def detect_needs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        شناسایی روندهای داغ بر اساس لیست درخواست‌های ورودی و (اختیاری) مقایسه با داده‌های پیشین.

        Args:
            input_data (Dict[str, Any]): داده‌های ورودی شامل:
                - "queries": List[str] (موضوعات یا عبارات)
                - "baseline": Optional[Dict[str, float]] (اختیاری: فراوانی‌های قبلی هر موضوع)

        Returns:
            List[Dict[str, Any]]: لیستی از موضوعات داغ همراه با جزئیات روند.
        """
        if not self.validate_input(input_data):
            self.logger.warning("[TrendDetector] Invalid input data.")
            return []

        queries = input_data.get("queries", [])
        if not queries:
            self.logger.info("[TrendDetector] No queries provided; no trends detected.")
            return []

        baseline: Optional[Dict[str, float]] = input_data.get("baseline")
        # محاسبه فراوانی فعلی موضوعات
        frequency_counter = Counter(queries)
        detected_trends = []

        for topic, current_freq in frequency_counter.items():
            trend_detected = False
            details = ""
            baseline_freq = None
            growth_factor = None

            if baseline and topic in baseline:
                baseline_freq = float(baseline[topic])
                if baseline_freq > 0:
                    growth_factor = current_freq / baseline_freq
                    if growth_factor >= self.growth_factor_threshold:
                        trend_detected = True
                        details = (f"Growth factor {growth_factor:.2f} exceeds threshold "
                                   f"{self.growth_factor_threshold}.")
                else:
                    # اگر baseline صفر است، تنها بررسی فراوانی فعلی
                    if current_freq >= self.default_trend_threshold:
                        trend_detected = True
                        details = (f"Current frequency {current_freq} exceeds threshold "
                                   f"{self.default_trend_threshold} with no prior baseline.")
            else:
                # در صورت عدم وجود baseline، فقط فراوانی فعلی مورد بررسی قرار می‌گیرد.
                if current_freq >= self.default_trend_threshold:
                    trend_detected = True
                    details = f"Current frequency {current_freq} meets or exceeds threshold {self.default_trend_threshold}."

            if trend_detected:
                need_info = {
                    "need_type": "TREND",
                    "topic": topic,
                    "current_frequency": current_freq,
                    "baseline_frequency": baseline_freq,
                    "growth_factor": growth_factor,
                    "details": details
                }
                detected_trends.append(need_info)
                self.logger.debug(f"[TrendDetector] Detected trend: {need_info}")

        final_trends = self.post_process_needs(detected_trends)
        self.logger.info(f"[TrendDetector] Total trends detected: {len(final_trends)}")
        return final_trends

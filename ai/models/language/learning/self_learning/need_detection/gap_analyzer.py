"""
GapAnalyzer Module
--------------------
این فایل مسئول شناسایی شکاف‌های دانشی مدل بر اساس مقایسه بین موضوعات مورد انتظار و موضوعات پوشش داده‌شده توسط مدل است.
با استفاده از داده‌های ورودی شامل لیست موضوعات مورد انتظار (expected_topics) و موضوعات پوشش داده‌شده (covered_topics)،
این کلاس میزان شکاف را محاسبه و در صورت بالا بودن نسبت شکاف نسبت به آستانه تعیین‌شده، نیاز به یادگیری در حوزه‌های گمشده را گزارش می‌دهد.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from typing import Dict, Any, List, Optional

from .need_detector_base import NeedDetectorBase


class GapAnalyzer(NeedDetectorBase):
    """
    GapAnalyzer برای شناسایی شکاف‌های دانشی مدل با مقایسه لیست‌های موضوعات مورد انتظار و پوشش داده‌شده طراحی شده است.

    ورودی مورد انتظار:
      {
        "expected_topics": List[str],
        "covered_topics": List[str],
        "coverage_threshold": Optional[float]  # درصد حداقل پوشش مورد انتظار (مثلاً 0.8)
      }

    خروجی:
      لیستی از دیکشنری‌هایی که هر کدام شامل اطلاعات شکاف دانشی به شکل زیر هستند:
        {
          "need_type": "GAP",
          "missing_topics": List[str],
          "coverage_ratio": float,
          "expected_count": int,
          "covered_count": int,
          "details": "Description of the gap"
        }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه GapAnalyzer.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختصاصی برای تنظیم آستانه‌های شکاف دانشی.
        """
        super().__init__(config=config)
        self.logger = logging.getLogger("GapAnalyzer")
        # آستانه پوشش پیش‌فرض: اگر پوشش موضوعات کمتر از 80 درصد باشد، شکاف وجود دارد.
        self.default_coverage_threshold = float(self.config.get("coverage_threshold", 0.8))
        self.logger.info(f"[GapAnalyzer] Initialized with default_coverage_threshold={self.default_coverage_threshold}")

    def detect_needs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        شناسایی شکاف‌های دانشی بر اساس لیست موضوعات مورد انتظار و پوشش داده‌شده.

        Args:
            input_data (Dict[str, Any]): داده‌های ورودی شامل:
                - expected_topics: لیستی از موضوعات مورد انتظار.
                - covered_topics: لیستی از موضوعات پوشش داده‌شده.
                - coverage_threshold (اختیاری): آستانه پوشش مورد انتظار (اگر ارائه شود جایگزین مقدار پیش‌فرض می‌شود).

        Returns:
            List[Dict[str, Any]]: لیستی از شکاف‌های دانشی شناسایی‌شده.
        """
        if not self.validate_input(input_data):
            self.logger.warning("[GapAnalyzer] Invalid input data.")
            return []

        expected_topics: List[str] = input_data.get("expected_topics", [])
        covered_topics: List[str] = input_data.get("covered_topics", [])
        if not expected_topics:
            self.logger.debug("[GapAnalyzer] No expected topics provided; no gap to detect.")
            return []

        # آستانه پوشش را از ورودی یا مقدار پیش‌فرض تنظیم می‌کنیم
        coverage_threshold = float(input_data.get("coverage_threshold", self.default_coverage_threshold))

        # محاسبه موضوعات مفقود
        missing_topics = [topic for topic in expected_topics if topic not in covered_topics]
        expected_count = len(expected_topics)
        covered_count = len(covered_topics)
        coverage_ratio = (expected_count - len(missing_topics)) / expected_count

        self.logger.debug(f"[GapAnalyzer] Expected topics: {expected_count}, Covered topics: {covered_count}, "
                          f"Missing topics: {missing_topics}, Coverage ratio: {coverage_ratio:.2f}")

        detected_needs = []
        if coverage_ratio < coverage_threshold:
            need_info = {
                "need_type": "GAP",
                "missing_topics": missing_topics,
                "coverage_ratio": coverage_ratio,
                "expected_count": expected_count,
                "covered_count": covered_count,
                "details": f"Coverage ratio ({coverage_ratio:.2f}) is below the threshold ({coverage_threshold:.2f})."
            }
            detected_needs.append(need_info)
            self.logger.info(f"[GapAnalyzer] Detected gap need: {need_info}")
        else:
            self.logger.info("[GapAnalyzer] No significant knowledge gap detected.")

        # پردازش نهایی (در صورت نیاز، می‌توان این لیست را مرتب یا فیلتر کرد)
        final_needs = self.post_process_needs(detected_needs)
        return final_needs


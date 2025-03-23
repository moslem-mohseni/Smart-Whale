"""
QueryAnalyzer Module
----------------------
این فایل مسئول تحلیل درخواست‌های متنی (query) کاربران برای شناسایی الگوهای تکراری،
ساختارهای پیچیده و نواقص احتمالی در پوشش دانشی مدل است.
این کلاس از NeedDetectorBase ارث‌بری می‌کند و با استفاده از تکنیک‌های آماری (مانند شمارش فراوانی و تحلیل طول)
الگوهای غیرمعمول را شناسایی کرده و نیازهای یادگیری مرتبط را گزارش می‌دهد.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from collections import Counter
from typing import Dict, Any, List, Optional

from .need_detector_base import NeedDetectorBase


class QueryAnalyzer(NeedDetectorBase):
    """
    QueryAnalyzer برای تحلیل درخواست‌های متنی (queries) ارائه شده توسط کاربران طراحی شده است.

    وظایف اصلی:
      - بررسی فراوانی تکرار درخواست‌ها برای شناسایی درخواست‌های پرتکرار.
      - تحلیل طول و پیچیدگی درخواست‌ها برای تشخیص درخواست‌های خارج از دامنه.
      - گزارش نیاز به بهبود یا توسعه دانش در حوزه‌های خاص بر مبنای الگوهای شناسایی‌شده.

    ورودی مورد انتظار:
      {
         "queries": List[str],            # لیستی از درخواست‌های متنی
         "frequency_threshold": Optional[int],  # حداقل تعداد تکرار برای شناسایی نیاز (پیش‌فرض: 3)
         "complexity_threshold": Optional[int]    # حداقل طول یا تعداد کلمات برای درخواست‌های پیچیده (پیش‌فرض: 5)
      }

    خروجی:
      لیستی از دیکشنری‌هایی که هر کدام شامل اطلاعات نیاز به بهبود حوزه‌های پرسشی هستند:
         {
           "need_type": "QUERY",
           "query": <str>,
           "frequency": <int>,
           "complexity": <int>,         # تعداد کلمات درخواست
           "details": <str>
         }
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه QueryAnalyzer.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختصاصی برای تنظیم آستانه‌ها.
                                               می‌تواند شامل "frequency_threshold" و "complexity_threshold" باشد.
        """
        super().__init__(config=config)
        self.logger = logging.getLogger("QueryAnalyzer")
        # آستانه‌های پیش‌فرض در صورت عدم ارائه در پیکربندی
        self.frequency_threshold = int(self.config.get("frequency_threshold", 3))
        self.complexity_threshold = int(self.config.get("complexity_threshold", 5))
        self.logger.info(f"[QueryAnalyzer] Initialized with frequency_threshold={self.frequency_threshold} and "
                         f"complexity_threshold={self.complexity_threshold}")

    def detect_needs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        شناسایی نیازهای یادگیری بر اساس تحلیل درخواست‌های متنی کاربران.

        Args:
            input_data (Dict[str, Any]): داده‌های ورودی شامل:
                - "queries": لیستی از درخواست‌های متنی.
                - "frequency_threshold" (اختیاری): حداقل تعداد تکرار برای شناسایی نیاز.
                - "complexity_threshold" (اختیاری): حداقل تعداد کلمات برای درخواست‌های پیچیده.

        Returns:
            List[Dict[str, Any]]: لیستی از نیازهای شناسایی‌شده در قالب دیکشنری.
        """
        if not self.validate_input(input_data):
            self.logger.warning("[QueryAnalyzer] Invalid input data.")
            return []

        queries: List[str] = input_data.get("queries", [])
        if not queries:
            self.logger.info("[QueryAnalyzer] No queries provided; no needs detected.")
            return []

        # به‌روزرسانی آستانه‌ها در صورت وجود در input_data
        freq_threshold = int(input_data.get("frequency_threshold", self.frequency_threshold))
        comp_threshold = int(input_data.get("complexity_threshold", self.complexity_threshold))

        # محاسبه فراوانی درخواست‌ها
        frequency_counter = Counter(queries)
        detected_needs = []

        for query, freq in frequency_counter.items():
            # تحلیل پیچیدگی: تعداد کلمات درخواست
            complexity = len(query.split())
            need_detected = False
            details = []

            # بررسی نیاز بر مبنای تکرار
            if freq >= freq_threshold:
                need_detected = True
                details.append(f"Frequency ({freq}) exceeds threshold ({freq_threshold}).")
            # بررسی نیاز بر مبنای پیچیدگی (در صورتی که درخواست پیچیده باشد ولی به درستی پاسخ داده نشده باشد)
            if complexity >= comp_threshold:
                need_detected = True
                details.append(f"Complexity ({complexity} words) exceeds threshold ({comp_threshold}).")

            if need_detected:
                need_info = {
                    "need_type": "QUERY",
                    "query": query,
                    "frequency": freq,
                    "complexity": complexity,
                    "details": " ".join(details)
                }
                detected_needs.append(need_info)
                self.logger.debug(f"[QueryAnalyzer] Detected query need: {need_info}")

        final_needs = self.post_process_needs(detected_needs)
        self.logger.info(f"[QueryAnalyzer] Total query needs detected: {len(final_needs)}")
        return final_needs


# نمونه تستی برای QueryAnalyzer
if __name__ == "__main__":
    import asyncio


    async def main():
        qa = QueryAnalyzer(config={"frequency_threshold": 3, "complexity_threshold": 5})
        sample_input = {
            "queries": [
                "What is quantum computing?",
                "What is quantum computing?",
                "Explain quantum mechanics",
                "What is AI?",
                "What is AI?",
                "What is AI?",
                "Define self learning",
                "Define self learning in AI",
                "Define self learning in AI"
            ]
        }
        result = qa.detect_needs(sample_input)
        print("Detected query needs:")
        for need in result:
            print(need)


    asyncio.run(main())

"""
SourceSelector Module
------------------------
این فایل مسئول انتخاب هوشمند منابع داده برای جمع‌آوری داده‌های آموزشی است.
کلاس SourceSelector بر اساس ورودی‌هایی مانند query، data_type و تنظیمات اختیاری، منبع مناسب (مانند WIKI، WEB، IMAGE یا GENERAL)
را انتخاب می‌کند و دلیل انتخاب را توضیح می‌دهد.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
import re
from typing import Dict, Any, Optional


class SourceSelector:
    """
    SourceSelector مسئول تعیین منبع مناسب جهت جمع‌آوری داده‌ها بر اساس پارامترهای ورودی است.

    ویژگی‌ها:
      - استفاده از الگوهای از پیش تعریف‌شده جهت تشخیص URL و واژگان کلیدی مرتبط با ویکی.
      - پذیرش پارامترهایی مانند query، data_type و params.
      - خروجی: دیکشنری شامل:
            - selected_source: منبع انتخاب‌شده (مانند "WIKI", "WEB", "IMAGE", یا "GENERAL")
            - rationale: توضیح درباره دلیل انتخاب منبع.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه SourceSelector.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل تنظیمات از پیش تعریف‌شده جهت انتخاب منبع.
                                              - "default_source": منبع پیش‌فرض (پیش‌فرض: "GENERAL")
                                              - "wiki_keywords": لیستی از کلمات کلیدی برای تشخیص منابع ویکی (پیش‌فرض: ["wikipedia", "wiki"])
        """
        self.logger = logging.getLogger("SourceSelector")
        self.config = config or {}
        self.default_source = self.config.get("default_source", "GENERAL")
        self.wiki_keywords = self.config.get("wiki_keywords", ["wikipedia", "wiki"])
        # الگوی شناسایی URL
        self.url_pattern = re.compile(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+")
        self.logger.info(f"[SourceSelector] Initialized with default_source={self.default_source} "
                         f"and wiki_keywords={self.wiki_keywords}")

    def select_source(self, query: str, data_type: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> \
    Dict[str, Any]:
        """
        انتخاب منبع مناسب بر اساس ورودی.

        Args:
            query (str): عبارت یا ورودی جستجو (مثلاً عنوان مقاله یا URL).
            data_type (Optional[str]): نوع داده مورد نیاز (مثلاً "TEXT", "IMAGE") – اختیاری.
            params (Optional[Dict[str, Any]]): پارامترهای اضافی جهت انتخاب منبع.

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - selected_source: منبع انتخاب‌شده.
                - rationale: توضیح درباره دلیل انتخاب.
        """
        rationale = []
        selected_source = self.default_source

        # بررسی وجود URL در query
        if self.url_pattern.search(query):
            selected_source = "WEB"
            rationale.append("Query contains URL pattern.")
        else:
            # بررسی وجود کلمات کلیدی مرتبط با ویکی در query
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in self.wiki_keywords):
                selected_source = "WIKI"
                rationale.append("Query contains wiki keywords.")
            else:
                # اگر data_type مشخص شده و به تصاویر اشاره دارد، انتخاب IMAGE
                if data_type and data_type.upper() == "IMAGE":
                    selected_source = "IMAGE"
                    rationale.append("Data type is IMAGE.")
                else:
                    selected_source = self.default_source
                    rationale.append("Default source selected.")

        self.logger.info(f"[SourceSelector] Selected source: {selected_source} | Rationale: {' '.join(rationale)}")
        return {
            "selected_source": selected_source,
            "rationale": " ".join(rationale)
        }


# Example usage for testing (final version intended for production)
if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.DEBUG)

    selector = SourceSelector(config={
        "default_source": "GENERAL",
        "wiki_keywords": ["wikipedia", "wiki"]
    })

    test_query = "Learn more about Wikipedia and its history."
    result = selector.select_source(test_query, data_type="TEXT")
    print(json.dumps(result, indent=2, ensure_ascii=False))

"""
DataCleaner Module
--------------------
این فایل مسئول تمیزسازی و استانداردسازی داده‌های ورودی برای فرآیندهای آموزشی سیستم خودآموزی است.
این کلاس داده‌های خام را پردازش کرده و نویز، نویسه‌های ناخواسته، فاصله‌های اضافی و خطاهای نگارشی را حذف می‌کند.
همچنین قابلیت نرمال‌سازی متنی (مانند تبدیل به حروف کوچک، حذف علائم نگارشی غیرضروری و ...) را فراهم می‌کند.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import re
import logging
from typing import Any, Dict, Optional

from . import quality_evaluator
from ..base.base_component import BaseComponent  # فرض بر این است که ساختار پوشه‌ی base در مسیر صحیح قرار دارد


class DataCleaner(BaseComponent):
    """
    DataCleaner مسئول تمیزسازی و پیش‌پردازش داده‌های ورودی جهت آموزش مدل است.

    امکانات:
      - حذف نویز، فاصله‌های اضافی، علائم نگارشی غیرضروری.
      - نرمال‌سازی متن: تبدیل به حروف کوچک، حذف کاراکترهای غیرمتنی.
      - آماده‌سازی داده‌ها برای پردازش‌های بعدی (مانند ارزیابی کیفیت و یکپارچه‌سازی دانش).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="data_cleaner", config=config)
        self.logger = logging.getLogger("DataCleaner")
        # تنظیمات پیش‌فرض برای تمیزسازی داده
        self.remove_punctuation = self.config.get("remove_punctuation", True)
        self.lowercase = self.config.get("lowercase", True)
        self.extra_space_pattern = re.compile(r"\s+")
        # الگوی حذف علائم نگارشی
        self.punctuation_pattern = re.compile(r"[^\w\s]")
        self.logger.info(
            f"[DataCleaner] Initialized with remove_punctuation={self.remove_punctuation}, lowercase={self.lowercase}")

    def clean_text(self, raw_text: str) -> str:
        """
        تمیزسازی و نرمال‌سازی متن ورودی.

        Args:
            raw_text (str): متن خام ورودی.

        Returns:
            str: متن تمیزشده و نرمال‌شده.
        """
        self.logger.debug(f"[DataCleaner] Starting cleaning text: {raw_text}")
        cleaned_text = raw_text.strip()
        # حذف فاصله‌های اضافی
        cleaned_text = self.extra_space_pattern.sub(" ", cleaned_text)
        # تبدیل به حروف کوچک در صورت فعال بودن
        if self.lowercase:
            cleaned_text = cleaned_text.lower()
        # حذف علائم نگارشی در صورت فعال بودن
        if self.remove_punctuation:
            cleaned_text = self.punctuation_pattern.sub("", cleaned_text)
        self.logger.debug(f"[DataCleaner] Cleaned text: {cleaned_text}")
        return cleaned_text

    def clean_data(self, data: Any) -> Any:
        """
        تمیزسازی داده ورودی. این متد می‌تواند برای انواع مختلف داده (متنی یا دیکشنری) استفاده شود.

        Args:
            data (Any): داده ورودی؛ ممکن است یک متن (str) یا یک دیکشنری باشد.

        Returns:
            Any: داده تمیزشده؛ اگر ورودی متن باشد، متن تمیز شده و در غیر این صورت داده به همان صورت برگردانده می‌شود.
        """
        if isinstance(data, str):
            return self.clean_text(data)
        elif isinstance(data, dict):
            # تمیزسازی تمامی مقادیر متنی در دیکشنری
            cleaned_data = {}
            for key, value in data.items():
                if isinstance(value, str):
                    cleaned_data[key] = self.clean_text(value)
                else:
                    cleaned_data[key] = value
            return cleaned_data
        else:
            # برای انواع دیگر داده، بدون تغییر برمی‌گردانیم
            return data

    def validate_cleaning(self, original: str, cleaned: str) -> bool:
        """
        ارزیابی کیفیت تمیزسازی: بررسی اینکه آیا تغییرات معناداری انجام شده است.

        Args:
            original (str): متن اصلی.
            cleaned (str): متن پس از تمیزسازی.

        Returns:
            bool: True اگر تمیزسازی موفقیت‌آمیز به نظر برسد، در غیر این صورت False.
        """
        # به عنوان نمونه، اگر طول متن تغییر کرده باشد، نتیجه تمیزسازی مثبت در نظر گرفته می‌شود.
        if len(original) != len(cleaned):
            return True
        return False

    def process(self, raw_input: Any) -> Dict[str, Any]:
        """
        پردازش کامل داده خام و برگرداندن خروجی شامل متن تمیزشده و ارزیابی کیفیت آن.

        Args:
            raw_input (Any): داده ورودی که می‌تواند متنی یا دیکشنری باشد.

        Returns:
            Dict[str, Any]: شامل:
                - cleaned_data: داده تمیزشده.
                - quality: نتیجه ارزیابی تمیزسازی (مثلاً True/False).
        """
        cleaned = self.clean_data(raw_input)
        quality = False
        if isinstance(raw_input, str):
            quality = self.validate_cleaning(raw_input, cleaned)
        elif isinstance(raw_input, dict):
            # به سادگی کیفیت را بر اساس اولین کلید متنی ارزیابی می‌کنیم
            for key, value in raw_input.items():
                if isinstance(value, str):
                    quality = self.validate_cleaning(value, cleaned.get(key, ""))
                    break
        return {
            "cleaned_data": cleaned,
            "quality": quality
        }


# نمونه تستی برای DataCleaner
if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level=logging.DEBUG)

    dc = DataCleaner(config={"remove_punctuation": True, "lowercase": True})
    raw_text = "   This is a Sample TEXT! With extra    spaces, and punctuation...   "
    result = dc.process(raw_text)
    print(json.dumps(result, indent=2, ensure_ascii=False))

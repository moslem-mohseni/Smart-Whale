"""
QualityEvaluator Module
-------------------------
این فایل مسئول ارزیابی کیفیت داده‌های پردازش‌شده (مثلاً متنی) برای اطمینان از صحت و اعتبار آن‌ها در فرآیند خودآموزی است.
این کلاس به عنوان یک ابزار نظارتی عمل می‌کند و بر مبنای معیارهایی مانند تعداد کلمات و نسبت حروف الفبا به کل کاراکترها، یک امتیاز کیفیت بین 0 تا 1 محاسبه می‌کند.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌ها و کارایی بالا پیاده‌سازی شده است.
"""

import re
import logging
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class QualityEvaluator(BaseComponent):
    """
    QualityEvaluator مسئول محاسبه کیفیت داده‌های متنی بر اساس معیارهای ساده مانند تعداد کلمات و نسبت حروف الفبا به کل کاراکترها است.

    امکانات:
      - محاسبه تعداد کلمات.
      - محاسبه نسبت حروف الفبا به کل کاراکترها (alpha_ratio).
      - تعیین امتیاز کیفیت (score) بین 0 تا 1.
      - ارائه جزئیات ارزیابی جهت بهبود کیفیت داده‌ها.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه QualityEvaluator.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "min_word_count": حداقل تعداد کلمات مورد نیاز برای داده با کیفیت (پیش‌فرض: 10).
        """
        super().__init__(component_type="quality_evaluator", config=config)
        self.logger = logging.getLogger("QualityEvaluator")
        self.min_word_count = int(self.config.get("min_word_count", 10))
        self.logger.info(f"[QualityEvaluator] Initialized with min_word_count={self.min_word_count}")

    def evaluate_quality(self, text: str) -> Dict[str, Any]:
        """
        ارزیابی کیفیت یک متن ورودی.

        Args:
            text (str): متن ورودی.

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - score (float): امتیاز کیفیت بین 0 تا 1.
                - word_count (int): تعداد کلمات موجود در متن.
                - alpha_ratio (float): نسبت تعداد حروف الفبا به کل کاراکترها.
                - details (str): توضیحات مربوط به ارزیابی کیفیت.
        """
        if not text:
            self.logger.warning("[QualityEvaluator] Received empty text.")
            return {"score": 0.0, "word_count": 0, "alpha_ratio": 0.0, "details": "Empty text"}

        # محاسبه تعداد کلمات
        words = text.split()
        word_count = len(words)
        # محاسبه کل کاراکترها و تعداد کاراکترهای الفبایی
        total_chars = len(text)
        alpha_chars = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0.0

        # ارزیابی کیفیت: اگر تعداد کلمات کمتر از حداقل تعیین‌شده باشد، کیفیت صفر است.
        if word_count < self.min_word_count:
            score = 0.0
            details = f"Word count ({word_count}) below minimum required ({self.min_word_count})."
        else:
            # به عنوان یک شاخص ساده، کیفیت برابر با نسبت حروف الفبا به کل کاراکترها در نظر گرفته می‌شود.
            score = alpha_ratio
            details = "Quality based on alphabetic ratio and sufficient word count."

        result = {
            "score": round(score, 3),
            "word_count": word_count,
            "alpha_ratio": round(alpha_ratio, 3),
            "details": details
        }
        self.logger.debug(f"[QualityEvaluator] Evaluation result: {result}")
        return result


# نمونه تستی برای QualityEvaluator
if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(level=logging.DEBUG)


    async def main():
        evaluator = QualityEvaluator(config={"min_word_count": 10})
        sample_text = "  This is a sample text! It contains multiple words, numbers 123, and punctuation...  "
        quality_result = evaluator.evaluate_quality(sample_text)
        print("Quality Evaluation:", quality_result)


    asyncio.run(main())

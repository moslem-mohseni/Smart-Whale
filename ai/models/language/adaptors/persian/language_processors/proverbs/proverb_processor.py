# persian/language_processors/proverbs/proverb_processor.py
"""
ماژول proverb_processor.py

این فایل زیرسیستم پردازش ضرب‌المثل زبان فارسی را پیاده‌سازی می‌کند.
کلاس PersianProverbProcessor به عنوان نقطه ورود اصلی برای پردازش ضرب‌المثل عمل می‌کند و وظایف زیر را انجام می‌دهد:
  - دریافت متن ورودی (خام)
  - نرمال‌سازی متن با استفاده از توابع موجود در proverb_nlp
  - استخراج کلمات کلیدی از ضرب‌المثل
  - جستجو برای ضرب‌المثل‌های مشابه با استفاده از روش‌های مبتنی بر شباهت متنی و برداری
  - جمع‌آوری متریک‌های عملکردی (در صورت وجود)
  - یکپارچه‌سازی خروجی نهایی به‌صورت دیکشنری شامل اطلاعات ضرب‌المثل، نسخه‌ها و متریک‌ها
"""

import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

# واردات ابزارهای پردازش زبان ضرب‌المثل از پوشه‌های مربوطه
from .proverb_data import ProverbDataAccess
from .proverb_nlp import normalize_proverb, extract_keywords, calculate_text_similarity
from .proverb_vector import ProverbVectorManager
# فرض کنید در صورت نیاز فایل proverb_metrics.py و proverb_integration.py نیز موجود هستند.
from .proverb_metrics import ProverbMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PersianProverbProcessor:
    """
    پردازشگر ضرب‌المثل فارسی

    این کلاس وظیفه هماهنگی بین اجزای زیرسیستم ضرب‌المثل را بر عهده دارد.
    از کلاس‌های ProverbDataAccess، proverb_nlp، ProverbVectorManager، ProverbMetrics و ProverbIntegration استفاده می‌کند.
    خروجی نهایی به صورت یک دیکشنری یکپارچه شامل اطلاعات ضرب‌المثل، کلمات کلیدی، ضرب‌المثل‌های تشخیص داده شده،
    متریک‌های جمع‌آوری‌شده و زمان تحلیل ارائه می‌شود.
    """

    def __init__(self,
                 data_access: Optional[ProverbDataAccess] = None,
                 vector_manager: Optional[ProverbVectorManager] = None,
                 metrics: Optional[ProverbMetrics] = None
                 ):
        # تنظیم ابزارهای دسترسی به داده
        self.data_access = data_access if data_access is not None else ProverbDataAccess()
        # بارگذاری ضرب‌المثل‌ها و نسخه‌ها از پایگاه داده
        self.proverbs = self.data_access.load_proverbs()
        self.variants = {}  # می‌توان از یک متد جداگانه برای بارگذاری نسخه‌ها استفاده کرد

        # ابزارهای پردازش زبان (NLP)
        # توابع موجود در proverb_nlp به عنوان توابع مستقل تعریف شده‌اند

        # مدیریت بردارهای معنایی
        self.vector_manager = vector_manager if vector_manager is not None else ProverbVectorManager()

        # جمع‌آوری متریک‌ها (در صورت وجود پیاده‌سازی)
        self.metrics = metrics if metrics is not None else ProverbMetrics()

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.info("PersianProverbProcessor با موفقیت مقداردهی اولیه شد.")

    def process_text(self, text: str) -> Dict[str, Any]:
        """
        پردازش متن ضرب‌المثل ورودی.

        Args:
            text (str): متن ورودی (خام)

        Returns:
            Dict[str, Any]: دیکشنری یکپارچه شامل:
                - original_text: متن اصلی ورودی
                - normalized_text: متن نرمال‌شده
                - keywords: لیست کلمات کلیدی استخراج‌شده
                - detected_proverbs: لیستی از ضرب‌المثل‌های تشخیص داده شده با جزئیات (شبیه‌سازی شده)
                - metrics: متریک‌های جمع‌آوری‌شده
                - analysis_time: زمان کل تحلیل (ثانیه)
        """
        start_time = datetime.now()

        # گام 1: نرمال‌سازی متن ضرب‌المثل
        normalized_text = normalize_proverb(text)
        self.logger.debug(f"متن نرمال‌شده: {normalized_text}")

        # گام 2: استخراج کلمات کلیدی از متن
        keywords = extract_keywords(normalized_text)
        self.logger.debug(f"کلمات کلیدی استخراج‌شده: {keywords}")

        # گام 3: تشخیص ضرب‌المثل‌های مشابه بر اساس شباهت متنی
        detected_proverbs = []
        for pid, p_data in self.proverbs.items():
            stored_proverb = p_data.get("proverb", "")
            similarity = calculate_text_similarity(normalized_text, stored_proverb)
            # استفاده از آستانه شباهت به عنوان نمونه (قابل تنظیم از طریق config)
            if similarity >= 0.7:
                detected_proverbs.append({
                    "proverb_id": pid,
                    "text": stored_proverb,
                    "meaning": p_data.get("meaning", ""),
                    "similarity": similarity
                })

        # گام 4: جستجوی برداری (در صورت نیاز)
        # تولید بردار معنایی از متن ورودی و جستجوی برداری در vector store
        semantic_vector = self.vector_manager.create_semantic_vector(normalized_text, "")
        vector_search_results = []
        try:
            # اجرای جستجوی برداری به صورت همزمان (synchronous برای سادگی)
            vector_search_results = asyncio.run(self.vector_manager.search_vectors(semantic_vector, top_k=3))
        except Exception as e:
            self.logger.error(f"خطا در جستجوی برداری: {e}")

        # ادغام نتایج تشخیص متنی و برداری (می‌توان آن‌ها را ترکیب کرد)
        for res in vector_search_results:
            pid = res.get("id", "")
            # اگر ضرب‌المثل از طریق متن قبلاً تشخیص داده نشده باشد
            if not any(dp.get("proverb_id") == pid for dp in detected_proverbs):
                # فرض می‌کنیم نتیجه برداری شامل فاصله و تبدیل به شباهت می‌شود
                detected_proverbs.append({
                    "proverb_id": pid,
                    "text": self.proverbs.get(pid, {}).get("proverb", ""),
                    "meaning": self.proverbs.get(pid, {}).get("meaning", ""),
                    "similarity": 1.0 - min(res.get("distance", 0), 1.0)
                })

        # مرتب‌سازی ضرب‌المثل‌های تشخیص داده شده بر اساس شباهت
        detected_proverbs = sorted(detected_proverbs, key=lambda x: x.get("similarity", 0), reverse=True)

        # گام 5: ثبت متریک‌ها
        analysis_time = (datetime.now() - start_time).total_seconds()
        try:
            self.metrics.collect_proverb_metrics(
                text_length=len(normalized_text),
                proverb_count=len(detected_proverbs),
                processing_time=analysis_time
            )
        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری متریک‌ها: {e}")

        result = {
            "original_text": text,
            "normalized_text": normalized_text,
            "keywords": keywords,
            "detected_proverbs": detected_proverbs,
            "metrics": self.metrics.get_metrics_snapshot(),
            "analysis_time": analysis_time
        }
        self.logger.info("پردازش ضرب‌المثل تکمیل شد.")
        return result


# نمونه تست مستقل
if __name__ == "__main__":
    import asyncio

    sample_text = "این یک متن نمونه است که در آن ضرب‌المثل 'هر که بامش بیش برفش بیشتر' وجود دارد."
    processor = PersianProverbProcessor()
    result = processor.process_text(sample_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

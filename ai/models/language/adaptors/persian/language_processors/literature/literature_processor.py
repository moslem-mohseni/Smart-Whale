# persian/language_processors/literature/literature_processor.py
"""
ماژول literature_processor.py

این فایل زیرسیستم پردازش متون ادبی فارسی را به عنوان رابط اصلی ارائه می‌دهد.
وظایف اصلی:
  - دریافت متن ورودی ادبی
  - پیش‌پردازش و نرمال‌سازی متن با استفاده از literature_nlp
  - تحلیل ادبی متن با استفاده از literature_analysis
  - ایجاد بردار معنایی متن با استفاده از literature_vector
  - ثبت و گزارش متریک‌های پردازش ادبی با استفاده از literature_metrics
  - ذخیره و مدیریت داده‌های ادبی (در صورت نیاز) با استفاده از literature_data
  - ارائه خروجی یکپارچه شامل تحلیل‌های ادبی، بردار معنایی، کلمات کلیدی و متریک‌ها
"""

import logging
import time
import json
from typing import Dict, Any

from .literature_nlp import normalize_literature, extract_keywords_literature
from .literature_analysis import LiteratureAnalysis
from .literature_vector import LiteratureVectorManager
from .literature_metrics import LiteratureMetrics
from .literature_data import LiteratureDataAccess
from .literature_services import LiteratureServices
from .literature_config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteratureProcessor:
    def __init__(self,
                 smart_model: Any = None,
                 teacher_model: Any = None,
                 vector_manager: Any = None,
                 metrics: Any = None,
                 data_access: Any = None,
                 services: Any = None):
        """
        سازنده LiteratureProcessor

        Args:
            smart_model: مدل هوشمند برای تحلیل ادبی (اختیاری)
            teacher_model: مدل معلم برای تحلیل ادبی (اختیاری)
            vector_manager: شیء مدیریت بردار معنایی ادبی (LiteratureVectorManager)
            metrics: شیء مدیریت متریک‌های پردازش ادبی (LiteratureMetrics)
            data_access: شیء دسترسی به داده‌های ادبی (LiteratureDataAccess)
            services: شیء خدمات تکمیلی (LiteratureServices)
        """
        self.smart_model = smart_model
        self.teacher_model = teacher_model

        # ایجاد نمونه LiteratureAnalysis با مدل‌های هوشمند (در صورت وجود)
        self.analysis = LiteratureAnalysis(smart_model=smart_model,
                                           teacher_model=teacher_model)

        # استفاده از LiteratureVectorManager برای ایجاد و جستجوی بردارهای معنایی
        self.vector_manager = vector_manager if vector_manager is not None else LiteratureVectorManager()
        # مدیریت متریک‌های پردازش متون ادبی
        self.metrics = metrics if metrics is not None else LiteratureMetrics()
        # دسترسی به داده‌های ادبی (مثلاً برای ذخیره‌سازی یا بازیابی)
        self.data_access = data_access if data_access is not None else LiteratureDataAccess()
        # خدمات تکمیلی مانند پشتیبان‌گیری و تاریخچه جستجو
        self.services = services if services is not None else LiteratureServices()

    def process(self, raw_text: str) -> Dict[str, Any]:
        """
        پردازش یک متن ادبی و ارائه خروجی یکپارچه شامل:
          - متن اصلی و نرمال‌شده
          - کلمات کلیدی استخراج‌شده
          - نتایج تحلیل ادبی (سبک، وزن، قالب، آرایه‌های ادبی)
          - بردار معنایی متن
          - متریک‌های عملکرد
          - زمان پردازش

        Args:
            raw_text: متن ورودی ادبی

        Returns:
            دیکشنری یکپارچه شامل نتایج تحلیل و پردازش.
        """
        start_time = time.time()

        # گام 1: پیش‌پردازش – نرمال‌سازی متن با استفاده از literature_nlp
        normalized_text = normalize_literature(raw_text)
        # استخراج کلمات کلیدی
        keywords = extract_keywords_literature(normalized_text)

        # گام 2: تحلیل ادبی – تحلیل سبک، وزن، قالب و آرایه‌های ادبی
        analysis_result = self.analysis.analyze(raw_text)

        # گام 3: ایجاد بردار معنایی – استفاده از LiteratureVectorManager
        semantic_vector = self.vector_manager.get_text_vector(normalized_text)

        # گام 4: ثبت متریک‌های پردازش
        processing_time = time.time() - start_time
        self.metrics.collect_literary_metrics(text_length=len(normalized_text),
                                              literary_text_count=1,
                                              processing_time=processing_time)

        # در صورت نیاز می‌توان نتایج تحلیل را در پایگاه داده ذخیره کرد (با استفاده از LiteratureDataAccess)
        # self.data_access.store_literary_corpus_item({...})

        # ترکیب خروجی نهایی
        result = {
            "original_text": raw_text,
            "normalized_text": normalized_text,
            "keywords": keywords,
            "analysis": analysis_result,
            "semantic_vector": semantic_vector,
            "metrics_snapshot": self.metrics.get_metrics_snapshot(),
            "processing_time": processing_time,
            "timestamp": time.time()
        }

        logger.info("متن ادبی با موفقیت پردازش شد.")
        return result


if __name__ == "__main__":
    sample_text = """هر آنکه آفتاب عشق در دل دارد،
از تاریکی شب بی‌خبر است.
در میان پرده‌های ظلمت،
نور امید تابان می‌شود."""
    processor = LiteratureProcessor()
    output = processor.process(sample_text)
    print(json.dumps(output, ensure_ascii=False, indent=2))

# persian/language_processors/domain/domain_processor.py

"""
ماژول domain_processor.py

این فایل شامل کلاس DomainProcessor است که مسئول هماهنگ‌سازی بین لایه‌های داده، تحلیل و متریک‌های حوزه است.
این کلاس یک خروجی یکپارچه (دیکشنری) ارائه می‌دهد که شامل اطلاعات حوزه، مفاهیم، روابط، پاسخ به پرسش‌های حوزه‌ای
و متریک‌های مربوطه می‌باشد.
"""

import asyncio
import time
import json
import logging

# واردات تنظیمات اختصاصی حوزه
from .domain_config import DOMAIN_CONFIG

# واردات لایه داده (DomainDataAccess)
from .domain_data import DomainDataAccess
# فرض می‌کنیم فایل‌های domain_analysis و domain_metrics به ترتیب پیاده‌سازی شده‌اند.
# در این مثال از آن‌ها استفاده می‌کنیم:
from .domain_analysis import DomainAnalysis  # این ماژول باید توابع تحلیل حوزه را داشته باشد
from .domain_metrics import DomainMetrics  # مشابه سایر متریک‌های سیستم

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DomainProcessor:
    """
    کلاس DomainProcessor مسئول هماهنگ‌سازی بین لایه‌های خدمات (داده و تحلیل) و متریک‌ها است.
    خروجی این کلاس شامل اطلاعات جامع حوزه (domain_info)، مفاهیم، روابط، پاسخ به پرسش‌های حوزه‌ای
    و متریک‌های عملکردی می‌باشد.
    """

    def __init__(self):
        # نمونه‌سازی از لایه داده
        self.data_access = DomainDataAccess()
        # نمونه‌سازی از لایه تحلیل (فرض می‌شود پیاده‌سازی شده است)
        self.analysis = DomainAnalysis()
        # نمونه‌سازی از متریک‌های حوزه
        self.metrics = DomainMetrics()
        # تنظیمات اختصاصی حوزه
        self.config = DOMAIN_CONFIG

    async def process_request(self, text: str) -> dict:
        """
        پردازش درخواست حوزه‌ای بر اساس متن ورودی.
        خروجی یک دیکشنری یکپارچه شامل:
          - domain_info: اطلاعات حوزه (در صورت یافتن)
          - concepts: لیست مفاهیم مرتبط با حوزه
          - relations: روابط مرتبط با مفاهیم حوزه
          - answer: پاسخ به پرسش حوزه‌ای (در صورت وجود)
          - metrics: آمار عملکرد به دست آمده
          - processing_time: زمان پردازش درخواست (به ثانیه)

        Args:
            text (str): متن ورودی (مثلاً پرسش یا توضیح حوزه)

        Returns:
            dict: خروجی یکپارچه حوزه
        """
        start_time = time.time()
        self.metrics.record_request()

        # مرحله ۱: یافتن حوزه مرتبط با متن
        try:
            detected_domains = await self.analysis.find_domain_for_text(text)
        except Exception as e:
            logger.error(f"خطا در یافتن حوزه با روش تحلیل: {e}")
            detected_domains = []

        if detected_domains:
            top_domain = detected_domains[0]
            domain_code = top_domain.get("domain_code", "")
        else:
            domain_code = None

        # مرحله ۲: بارگذاری اطلاعات حوزه از پایگاه داده
        domain_info = {}
        if domain_code:
            try:
                domains = await self.data_access.load_domains()
                # یافتن حوزه با کد مشخص شده
                for d in domains.values():
                    if d.get("domain_code") == domain_code:
                        domain_info = d
                        break
            except Exception as e:
                logger.error(f"خطا در بارگذاری اطلاعات حوزه: {e}")

        # مرحله ۳: دریافت مفاهیم مرتبط با حوزه
        domain_concepts = []
        try:
            concepts = await self.data_access.load_concepts()
            if domain_info:
                domain_id = domain_info.get("domain_id")
                domain_concepts = [c for c in concepts.values() if c.get("domain_id") == domain_id]
        except Exception as e:
            logger.error(f"خطا در بارگذاری مفاهیم: {e}")

        # مرحله ۴: دریافت روابط مرتبط با مفاهیم حوزه
        domain_relations = []
        try:
            relations = await self.data_access.load_relations()
            concept_ids = {c.get("concept_id") for c in domain_concepts}
            domain_relations = [r for r in relations.values() if r.get("source_concept_id") in concept_ids
                                or r.get("target_concept_id") in concept_ids]
        except Exception as e:
            logger.error(f"خطا در بارگذاری روابط: {e}")

        # مرحله ۵: پاسخ به پرسش حوزه‌ای
        answer = {}
        if domain_code:
            try:
                answer = await self.analysis.answer_domain_question(text, domain_code)
            except Exception as e:
                logger.error(f"خطا در پاسخ‌دهی به پرسش حوزه‌ای: {e}")
                answer = {"answer": "در حال حاضر پاسخ در دسترس نیست.", "confidence": 0.0, "domain_code": domain_code}
        else:
            answer = {"answer": "هیچ حوزه مرتبط یافت نشد.", "confidence": 0.0, "domain_code": None}

        # مرحله ۶: ثبت زمان پردازش
        processing_time = time.time() - start_time
        self.metrics.record_processing_time(processing_time)

        # مرحله ۷: جمع‌آوری متریک‌ها
        metrics_snapshot = self.metrics.get_metrics_snapshot()

        result = {
            "domain_info": domain_info,
            "concepts": domain_concepts,
            "relations": domain_relations,
            "answer": answer,
            "metrics": metrics_snapshot,
            "processing_time": round(processing_time, 4)
        }

        return result


# نمونه تست مستقل
if __name__ == "__main__":
    async def main():
        processor = DomainProcessor()
        sample_text = "در مورد پزشکی عمومی و روش‌های تشخیص بیماری‌های قلبی توضیح دهید."
        result = await processor.process_request(sample_text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    asyncio.run(main())

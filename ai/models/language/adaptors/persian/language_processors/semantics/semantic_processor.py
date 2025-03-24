"""
ماژول semantic_processor.py

این فایل هماهنگ‌کننده نهایی زیرسیستم تحلیل معنایی متون فارسی است.
کلاس SemanticProcessor مسئول هماهنگی بین لایه‌های خدمات (SemanticServices) و متریک‌ها (SemanticMetrics) می‌باشد.
این کلاس یک خروجی یکپارچه به صورت دیکشنری ارائه می‌دهد که شامل نتایج تحلیل معنایی و متریک‌های عملکرد است.
"""

import json
import logging
import time
from typing import Dict, Any

from .semantic_services import SemanticServices
from .semantic_metrics import SemanticMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticProcessor:
    def __init__(self):
        self.logger = logger
        self.semantic_services = SemanticServices()
        self.semantic_metrics = SemanticMetrics()

    def process(self, text: str) -> Dict[str, Any]:
        start_time = time.time()
        # تحلیل معنایی متن
        analysis_result = self.semantic_services.analyze_text(text)
        processing_time = time.time() - start_time

        # ثبت متریک‌های تحلیل معنایی
        self.semantic_metrics.collect_semantic_metrics(
            text_length=len(text),
            processing_time=processing_time,
            source=analysis_result.source
        )

        result = {
            "analysis": analysis_result,
            "metrics": self.semantic_metrics.get_metrics_snapshot(),
            "processing_time": processing_time
        }
        self.logger.info(f"Semantic processing completed in {processing_time:.4f} seconds.")
        return result


if __name__ == "__main__":
    processor = SemanticProcessor()
    sample_text = "این یک متن نمونه برای تحلیل معنایی است. هدف این متن استخراج هدف، احساسات، موضوعات و بردار معنایی آن می‌باشد."
    result = processor.process(sample_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))

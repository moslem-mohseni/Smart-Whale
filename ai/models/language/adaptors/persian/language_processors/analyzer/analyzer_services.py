# language_processors/analyzer/analyzer_services.py

"""
ماژول analyzer_services.py

این فایل شامل کلاس AnalyzerServices است که مسئول اجرای منطق تجاری اصلی برای آنالیز
متون فارسی می‌باشد. این کلاس با استفاده از زیرسیستم‌های مربوط به گرامر، لهجه، معناشناسی و حوزه،
نتایج جداگانه هر کدام را دریافت و در یک نتیجه‌ی کلی تلفیق می‌کند.

ویژگی‌های کلیدی:
  - فراخوانی پردازشگرهای مربوط به گرامر (GrammarProcessor)، لهجه (DialectProcessor)، معناشناسی (SemanticProcessor)
    و حوزه (DomainProcessor) برای تحلیل عمیق متن.
  - تلفیق نتایج دریافت شده با استفاده از وزن‌های تعیین‌شده در ANALYZER_CONFIG.
  - برگرداندن یک شیء AnalysisResult (تعریف شده در analyzer_models.py) حاوی امتیازهای جزئی و کلی و
    گزارش‌های تکمیلی.
"""

import logging
import time
from typing import Dict, Any

from ..grammar.grammar_processor import GrammarProcessor
from ..dialects.processor import DialectProcessor
from ..semantics.semantic_processor import SemanticProcessor
from ..domain.domain_processor import DomainProcessor

from .analyzer_config import ANALYZER_CONFIG
from .analyzer_models import AnalysisResult

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnalyzerServices:
    """
    کلاس AnalyzerServices مسئول تحلیل جامع متون فارسی با تلفیق نتایج زیرسیستم‌های مختلف است.
    """

    def __init__(self):
        self.logger = logger
        self.grammar_processor = GrammarProcessor()
        self.dialect_processor = DialectProcessor()
        self.semantic_processor = SemanticProcessor()
        self.domain_processor = DomainProcessor()
        self.config = ANALYZER_CONFIG
        self.weights = self.config.get("analysis_weights", {
            "grammar": 0.3,
            "dialect": 0.2,
            "semantic": 0.4,
            "domain": 0.1
        })
        self.logger.info("AnalyzerServices initialized with weights: %s", self.weights)

    def analyze_text_deeply(self, text: str) -> AnalysisResult:
        """
        تحلیل عمیق متن با استفاده از پردازش‌های گرامری، لهجه، معناشناسی و حوزه.

        Args:
            text (str): متن ورودی جهت آنالیز

        Returns:
            AnalysisResult: شیء حاوی امتیازهای زیرسیستم‌ها و امتیاز کلی به همراه جزئیات.
        """
        start_time = time.time()

        # دریافت نتایج تحلیل از هر زیرسیستم
        try:
            grammar_res = self.grammar_processor.process(text)
            self.logger.info("نتیجه تحلیل گرامر دریافت شد.")
        except Exception as e:
            self.logger.error("خطا در تحلیل گرامر: %s", e)
            grammar_res = {"score": 0.0, "details": {}}

        try:
            dialect_res = self.dialect_processor.detect(text)
            # فرض می‌کنیم که نتیجه لهجه دارای فیلد 'confidence' به عنوان امتیاز باشد.
            dialect_score = dialect_res.get("confidence", 0.0)
            dialect_details = dialect_res
            self.logger.info("نتیجه تحلیل لهجه دریافت شد.")
        except Exception as e:
            self.logger.error("خطا در تحلیل لهجه: %s", e)
            dialect_score = 0.0
            dialect_details = {}

        try:
            semantic_res = self.semantic_processor.analyze(text)
            self.logger.info("نتیجه تحلیل معناشناسی دریافت شد.")
        except Exception as e:
            self.logger.error("خطا در تحلیل معناشناسی: %s", e)
            semantic_res = {"score": 0.0, "details": {}}

        try:
            domain_res = self.domain_processor.process(text)
            self.logger.info("نتیجه تحلیل حوزه دریافت شد.")
        except Exception as e:
            self.logger.error("خطا در تحلیل حوزه: %s", e)
            domain_res = {"score": 0.0, "details": {}}

        # استخراج امتیازهای هر بخش (فرض بر این است که هر نتیجه شامل کلید 'score' باشد)
        grammar_score = grammar_res.get("score", 0.0)
        dialect_score = dialect_score  # از تحلیل لهجه
        semantic_score = semantic_res.get("score", 0.0)
        domain_score = domain_res.get("score", 0.0)

        # محاسبه امتیاز کلی با استفاده از وزن‌های تعیین‌شده
        overall_score = (
            grammar_score * self.weights.get("grammar", 0.3) +
            dialect_score * self.weights.get("dialect", 0.2) +
            semantic_score * self.weights.get("semantic", 0.4) +
            domain_score * self.weights.get("domain", 0.1)
        )

        # ترکیب جزئیات تحلیل
        details = {
            "grammar": grammar_res.get("details", {}),
            "dialect": dialect_details,
            "semantic": semantic_res.get("details", {}),
            "domain": domain_res.get("details", {}),
            "processing_time": round(time.time() - start_time, 3)
        }

        self.logger.info("تحلیل متن به پایان رسید. امتیاز کلی: %s", overall_score)

        return AnalysisResult(
            grammar_score=grammar_score,
            dialect_score=dialect_score,
            semantic_score=semantic_score,
            domain_score=domain_score,
            overall_score=overall_score,
            details=details
        )

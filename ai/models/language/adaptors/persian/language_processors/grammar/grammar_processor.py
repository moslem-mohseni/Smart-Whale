# persian/language_processors/grammar/grammar_processor.py
"""
ماژول grammar_processor.py

این فایل زیرسیستم پردازش گرامر زبان فارسی را پیاده‌سازی می‌کند.
کلاس GrammarProcessor به عنوان رابط اصلی زیرسیستم گرامر عمل کرده و وظایف زیر را انجام می‌دهد:
  - دریافت متن فارسی (به عنوان ورودی)
  - نرمال‌سازی متن با استفاده از ابزارهای موجود در پوشه utils
  - تحلیل قواعد گرامری از طریق RuleManager
  - استخراج و تحلیل برچسب‌های دستوری (POS) از طریق POSAnalyzer
  - اعمال اصلاحات گرامری با استفاده از CorrectionEngine
  - جمع‌آوری متریک‌های عملکردی از طریق GrammarMetrics

خروجی نهایی شامل یک دیکشنری یکپارچه است که موارد زیر را در بر می‌گیرد:
  - original_text: متن اصلی ورودی
  - normalized_text: متن نرمال‌شده
  - grammar_errors: لیست خطاهای گرامری به همراه پیشنهادات اصلاح
  - pos_analysis: نتایج تحلیل برچسب‌های دستوری (POS)
  - corrected_text: متن پس از اعمال اصلاحات
  - metrics: نمای کلی از متریک‌های جمع‌آوری‌شده
  - analysis_time: زمان کل تحلیل به ثانیه
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..utils.text_normalization import TextNormalizer
from ..utils.tokenization import Tokenizer
from .rule_manager import RuleManager
from .pos_analyzer import POSAnalyzer
from .correction_engine import CorrectionEngine
from .metrics import GrammarMetrics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GrammarProcessor:
    """
    پردازشگر گرامر فارسی

    این کلاس زیرسیستمی برای تحلیل گرامر متون فارسی است که وظایف زیر را انجام می‌دهد:
      - نرمال‌سازی متن
      - تحلیل قواعد گرامری (استخراج خطاها) با استفاده از RuleManager
      - تحلیل برچسب‌های دستوری (POS) با استفاده از POSAnalyzer
      - اعمال اصلاحات بر مبنای قواعد موجود از طریق CorrectionEngine
      - جمع‌آوری و گزارش متریک‌های عملکردی از طریق GrammarMetrics

    جریان داده:
      - ورودی: متن فارسی (خام)
      - پردازش:
           1. نرمال‌سازی متن
           2. استخراج خطاهای گرامری
           3. تحلیل POS متن
           4. اعمال اصلاحات
           5. جمع‌آوری متریک‌ها
      - خروجی: دیکشنری یکپارچه شامل متن اصلی، متن نرمال‌شده، خطاهای گرامری، متن اصلاح‌شده، تحلیل POS، متریک‌ها و زمان تحلیل.
    """

    def __init__(self,
                 language: str = "persian",
                 normalizer: Optional[TextNormalizer] = None,
                 tokenizer: Optional[Tokenizer] = None,
                 rule_manager: Optional[RuleManager] = None,
                 pos_analyzer: Optional[POSAnalyzer] = None,
                 correction_engine: Optional[CorrectionEngine] = None,
                 metrics: Optional[GrammarMetrics] = None):
        self.language = language
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # ابزارهای پردازش اولیه
        self.normalizer = normalizer if normalizer is not None else TextNormalizer()
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()

        # اجزای زیرسیستم گرامر
        self.rule_manager = rule_manager if rule_manager is not None else RuleManager()
        self.pos_analyzer = pos_analyzer if pos_analyzer is not None else POSAnalyzer(language=self.language)
        self.correction_engine = correction_engine if correction_engine is not None else CorrectionEngine()
        self.metrics = metrics if metrics is not None else GrammarMetrics()

        self.logger.info("GrammarProcessor با موفقیت مقداردهی اولیه شد.")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        تحلیل گرامری متن فارسی.

        Args:
            text (str): متن ورودی (خام)

        Returns:
            Dict[str, Any]: دیکشنری شامل:
                - original_text: متن اصلی ورودی
                - normalized_text: متن نرمال‌شده
                - grammar_errors: لیست خطاهای گرامری به همراه پیشنهادات اصلاح
                - pos_analysis: نتایج تحلیل برچسب‌های دستوری (POS)
                - corrected_text: متن پس از اعمال اصلاحات
                - metrics: نمای کلی از متریک‌های جمع‌آوری‌شده
                - analysis_time: زمان کل تحلیل (ثانیه)
        """
        start_time = datetime.now()

        # مرحله ۱: نرمال‌سازی متن
        try:
            normalized_text = self.normalizer.normalize(text)
            self.logger.debug("متن نرمال‌شده.")
        except Exception as e:
            self.logger.error(f"خطا در نرمال‌سازی متن: {e}")
            normalized_text = text

        # مرحله ۲: استخراج خطاهای گرامری با استفاده از RuleManager
        try:
            # استفاده از متد analyze (به‌روزرسانی‌شده در RuleManager)
            grammar_errors = self.rule_manager.analyze(normalized_text)
            self.logger.debug(f"تعداد خطاهای گرامری یافت‌شده: {len(grammar_errors)}")
        except Exception as e:
            self.logger.error(f"خطا در تحلیل قواعد گرامری: {e}")
            grammar_errors = []

        # مرحله ۳: تحلیل POS متن با استفاده از POSAnalyzer
        try:
            pos_analysis = self.pos_analyzer.analyze(normalized_text)
            self.logger.debug("تحلیل POS انجام شد.")
        except Exception as e:
            self.logger.error(f"خطا در تحلیل POS: {e}")
            pos_analysis = []

        # مرحله ۴: اعمال اصلاحات گرامری با استفاده از CorrectionEngine
        try:
            correction_result = self.correction_engine.correct_text(normalized_text)
            corrected_text = correction_result.get("corrected_text", normalized_text)
            self.logger.debug("اصلاحات گرامری اعمال شدند.")
        except Exception as e:
            self.logger.error(f"خطا در اعمال اصلاحات گرامری: {e}")
            corrected_text = normalized_text

        # مرحله ۵: محاسبه زمان تحلیل و جمع‌آوری متریک‌ها
        analysis_time = (datetime.now() - start_time).total_seconds()
        try:
            self.metrics.collect_grammar_analysis_metrics(
                text_length=len(normalized_text),
                error_count=len(grammar_errors),
                confidence=0.0,  # در صورت وجود میزان اطمینان تحلیل، مقداردهی شود
                source="rule_based",
                processing_time=analysis_time
            )
            self.logger.debug("متریک‌های تحلیل گرامر جمع‌آوری شدند.")
        except Exception as e:
            self.logger.error(f"خطا در جمع‌آوری متریک‌ها: {e}")

        result = {
            "original_text": text,
            "normalized_text": normalized_text,
            "grammar_errors": grammar_errors,
            "pos_analysis": pos_analysis,
            "corrected_text": corrected_text,
            "metrics": self.metrics.get_metrics_snapshot(),
            "analysis_time": analysis_time
        }
        self.logger.info("تحلیل گرامری تکمیل شد.")
        return result

    def get_rules(self) -> List[Dict[str, Any]]:
        """
        دریافت قواعد گرامری موجود.

        Returns:
            List[Dict[str, Any]]: لیستی از قواعد گرامری.
        """
        return self.rule_manager.get_all_rules()

    def update_rules(self) -> List[Dict[str, Any]]:
        """
        به‌روزرسانی قواعد گرامری.

        Returns:
            List[Dict[str, Any]]: لیست به‌روز شده قواعد گرامری.
        """
        return self.rule_manager.update_rules()

    def get_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار عملکرد تحلیل گرامری.

        Returns:
            Dict[str, Any]: دیکشنری شامل آمارهای جمع‌آوری‌شده.
        """
        return self.metrics.get_metrics_snapshot()

    def export_knowledge(self, filename: str = "grammar_knowledge.json") -> str:
        """
        خروجی گرفتن از دانش گرامری ذخیره‌شده.

        Args:
            filename (str): نام فایل خروجی.

        Returns:
            str: پیام نتیجه عملیات.
        """
        return self.rule_manager.export_rules(filename)

    def import_knowledge(self, filename: str = "grammar_knowledge.json") -> str:
        """
        وارد کردن دانش گرامری از فایل.

        Args:
            filename (str): نام فایل ورودی.

        Returns:
            str: پیام نتیجه عملیات.
        """
        return self.rule_manager.import_rules(filename)


if __name__ == "__main__":
    sample_text = "این یک متن نمونه است که نیاز به بررسی گرامری دارد. ممکن است برخی از اشتباهات نگارشی داشته باشد."
    processor = GrammarProcessor()
    analysis_result = processor.analyze(sample_text)
    import json
    print(json.dumps(analysis_result, ensure_ascii=False, indent=4))

# language_processors/analyzer/analyzer_models.py

"""
ماژول analyzer_models.py

در این فایل، مدل‌های داده‌ای مورد نیاز در زیرسیستم آنالیز (analyzer) تعریف می‌شوند.
این مدل‌ها برای ذخیره و تبادل اطلاعات بین بخش‌های مختلف آنالیز (خروجی زیرسیستم‌های دیگر
و سرویس‌های داخلی ماژول) استفاده می‌شوند.

مثال‌هایی از مدل‌ها:
  - AnalyzerInput: ورودی آنالیز (متن و سایر پارامترها)
  - AnalysisResult: نتیجه نهایی آنالیز شامل امتیازات زیرسیستم‌های مختلف و جزئیات
"""

from typing import Dict, Any, Optional


class AnalyzerInput:
    """
    کلاسی برای نگهداری ورودی آنالیز. می‌توان پارامترهای مختلفی (متن، تنظیمات سفارشی، متادیتا و غیره) را در اینجا ذخیره کرد.
    """

    def __init__(self, text: str, extra_config: Optional[Dict[str, Any]] = None):
        """
        سازنده AnalyzerInput

        Args:
            text (str): متن ورودی جهت آنالیز
            extra_config (Optional[Dict[str, Any]]): پیکربندی اضافی یا داده‌های متفرقه
        """
        self.text = text
        self.extra_config = extra_config if extra_config else {}

    def __repr__(self):
        return f"<AnalyzerInput text='{self.text[:30]}...' extra_config={self.extra_config}>"


class AnalysisResult:
    """
    کلاس اصلی برای نگهداری خروجی آنالیز نهایی.
    این کلاس می‌تواند نتایج جزئی و امتیاز کلی را شامل شود.
    """

    def __init__(
        self,
        grammar_score: float = 0.0,
        dialect_score: float = 0.0,
        semantic_score: float = 0.0,
        domain_score: float = 0.0,
        overall_score: float = 0.0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        سازنده AnalysisResult

        Args:
            grammar_score (float): امتیاز گرامری
            dialect_score (float): امتیاز لهجه
            semantic_score (float): امتیاز معنایی
            domain_score (float): امتیاز مربوط به حوزه
            overall_score (float): امتیاز کلی آنالیز (مثلاً میانگین یا ترکیبی از سایر امتیازات)
            details (Optional[Dict[str, Any]]): جزئیات تکمیلی (گزارش‌های هر زیرسیستم، خطاها، توضیحات و ...)
        """
        self.grammar_score = grammar_score
        self.dialect_score = dialect_score
        self.semantic_score = semantic_score
        self.domain_score = domain_score
        self.overall_score = overall_score
        self.details = details if details else {}

    def __repr__(self):
        return (
            f"<AnalysisResult grammar={self.grammar_score}, dialect={self.dialect_score}, "
            f"semantic={self.semantic_score}, domain={self.domain_score}, overall={self.overall_score}, "
            f"details_keys={list(self.details.keys())}>"
        )

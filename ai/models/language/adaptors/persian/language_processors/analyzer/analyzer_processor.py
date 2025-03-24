# language_processors/analyzer/analyzer_processor.py

"""
ماژول analyzer_processor.py

این فایل هماهنگ‌کننده‌ی نهایی زیرسیستم آنالیز است.
کلاس AnalyzerProcessor جریان کامل پردازش یک درخواست آنالیز را مدیریت می‌کند:
  - دریافت ورودی (متن مورد آنالیز)
  - فراخوانی سرویس‌های اصلی آنالیز (از طریق AnalyzerServices)
  - ثبت و گزارش متریک‌های عملکرد (با استفاده از AnalyzerMetrics)
  - ذخیره نتایج آنالیز در پایگاه داده و کش (از طریق AnalyzerDataAccess) در صورت نیاز
  - برگرداندن خروجی یکپارچه شامل امتیازات زیرسیستم‌های مختلف و جزئیات

این فایل نقطه ورود سطح بالا برای پردازش درخواست‌های آنالیز است.
"""

import logging
import time
from typing import Dict, Any

from .analyzer_services import AnalyzerServices
from .analyzer_metrics import AnalyzerMetrics
from .analyzer_data import AnalyzerDataAccess
from .analyzer_models import AnalysisResult, AnalyzerInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnalyzerProcessor:
    """
    کلاس AnalyzerProcessor، نقطه ورود اصلی زیرسیستم آنالیز است.
    """
    def __init__(self):
        self.logger = logger
        self.services = AnalyzerServices()
        self.metrics = AnalyzerMetrics()
        self.data_access = AnalyzerDataAccess()
        self.logger.info("AnalyzerProcessor initialized successfully.")

    def process_analysis(self, text: str, save_report: bool = False) -> Dict[str, Any]:
        """
        پردازش یک درخواست آنالیز متن به صورت یکپارچه.

        مراحل:
          1. دریافت متن ورودی و (در صورت نیاز) نرمال‌سازی آن توسط سرویس‌های داخلی.
          2. فراخوانی متد analyze_text_deeply از AnalyzerServices جهت دریافت نتایج
             تحلیل زیرسیستم‌های گرامر، لهجه، معناشناسی و حوزه.
          3. ثبت متریک‌های مربوط به پردازش (زمان اجرا، موفقیت یا خطا، وضعیت کش).
          4. (اختیاری) ذخیره گزارش نهایی آنالیز در پایگاه داده و کش.
          5. بازگرداندن خروجی نهایی به صورت یک دیکشنری شامل امتیازهای جداگانه و کلی.

        Args:
            text (str): متن ورودی جهت آنالیز.
            save_report (bool, optional): اگر True باشد، گزارش آنالیز ذخیره می‌شود (پیش‌فرض: False).

        Returns:
            Dict[str, Any]: خروجی یکپارچه آنالیز شامل امتیازات و جزئیات تحلیل.
        """
        start_time = time.time()
        try:
            self.logger.info("شروع پردازش آنالیز متن.")
            # دریافت نتیجه تحلیل از سرویس‌های زیرسیستم
            analysis_result: AnalysisResult = self.services.analyze_text_deeply(text)
            processing_time = time.time() - start_time
            self.metrics.record_request(processing_time, success=True, cache_hit=False)

            result_dict = {
                "grammar_score": analysis_result.grammar_score,
                "dialect_score": analysis_result.dialect_score,
                "semantic_score": analysis_result.semantic_score,
                "domain_score": analysis_result.domain_score,
                "overall_score": analysis_result.overall_score,
                "details": analysis_result.details,
                "processing_time": round(processing_time, 3)
            }
            self.logger.info("پردازش آنالیز با موفقیت به پایان رسید.")

            # ذخیره گزارش در صورت درخواست
            if save_report:
                saved = self.data_access.store_analysis_report(text, result_dict)
                result_dict["report_saved"] = saved

            return result_dict

        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.record_request(processing_time, success=False, cache_hit=False)
            self.logger.error("خطا در پردازش آنالیز: %s", e)
            return {
                "error": str(e),
                "processing_time": round(processing_time, 3)
            }

    def get_processor_metrics(self) -> Dict[str, Any]:
        """
        دریافت متریک‌های عملکرد پردازشگر آنالیز.

        Returns:
            Dict[str, Any]: آمار و متریک‌های جمع‌آوری‌شده توسط AnalyzerMetrics.
        """
        return self.metrics.get_metrics()

    def get_saved_reports(self) -> Dict[str, Any]:
        """
        دریافت تمامی گزارش‌های ذخیره‌شده آنالیز.

        Returns:
            Dict[str, Any]: شامل لیست گزارش‌های ذخیره‌شده و آمارهای مربوط به آن.
        """
        reports = self.data_access.get_all_analysis_reports()
        stats = self.data_access.get_statistics()
        return {"reports": reports, "statistics": stats}

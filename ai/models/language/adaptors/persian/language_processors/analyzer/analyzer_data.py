# language_processors/analyzer/analyzer_data.py

"""
ماژول analyzer_data.py

این فایل مسئول دسترسی به داده‌های مرتبط با زیرسیستم آنالیز است.
عملیات اصلی این فایل عبارتند از:
  - ایجاد و به‌روزرسانی جداول پایگاه داده (برای ذخیره گزارش‌های تحلیل)
  - خواندن، درج و به‌روزرسانی نتایج آنالیز از/به پایگاه داده
  - استفاده از کش (Redis) برای دسترسی سریع به نتایج موقت
  - در صورت عدم وجود داده در کش یا پایگاه داده، استفاده از مقادیر پیش‌فرض و ذخیره آن‌ها

این ماژول از زیرساخت‌های موجود (RedisAdapter، CacheManager، ClickHouseDB و FileManagementService) استفاده می‌کند.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ai.models.language.infrastructure.caching.redis_adapter import RedisAdapter
from ai.models.language.infrastructure.caching.cache_manager import CacheManager
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.file_management.file_service import FileManagementService

from ...config import CONFIG

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AnalyzerDataAccess:
    """
    کلاس AnalyzerDataAccess مسئول مدیریت داده‌های زیرسیستم آنالیز است.
    این کلاس شامل توابعی برای خواندن و نوشتن گزارش‌های تحلیل (analysis reports)
    در پایگاه داده و کش می‌باشد.
    """

    def __init__(self, language: str = "persian"):
        self.language = language
        self.logger = logger
        self.redis = RedisAdapter()
        self.cache_manager = CacheManager()
        self.database = ClickHouseDB()
        self.file_service = FileManagementService()
        self.statistics = {
            "analysis_reports_retrieved": 0,
            "analysis_reports_inserted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": time.time()
        }
        self._setup_database()
        self.logger.info("AnalyzerDataAccess initialized successfully.")

    def _setup_database(self) -> None:
        """
        ایجاد جداول مورد نیاز برای ذخیره گزارش‌های آنالیز در پایگاه داده.
        در اینجا یک جدول اصلی به نام analysis_reports ایجاد می‌شود.
        """
        try:
            self.database.execute_query("""
                CREATE TABLE IF NOT EXISTS analysis_reports (
                    report_id String,
                    text String,
                    analysis_result String,
                    timestamp DateTime DEFAULT now()
                ) ENGINE = MergeTree()
                ORDER BY (report_id, timestamp)
            """)
            self.logger.info("جدول analysis_reports در پایگاه داده ایجاد شد یا قبلاً وجود داشت.")
        except Exception as e:
            self.logger.error(f"خطا در ایجاد جدول analysis_reports: {e}")

    def get_analysis_report(self, text: str) -> Optional[Dict[str, Any]]:
        """
        دریافت گزارش آنالیز مربوط به متن داده شده از کش یا پایگاه داده.
        ابتدا تلاش می‌شود نتیجه در کش بازیابی شود. در صورت عدم موفقیت، از پایگاه داده خوانده می‌شود.
        اگر در هیچ‌یک موجود نباشد، None برگردانده می‌شود.

        Args:
            text (str): متن ورودی جهت آنالیز

        Returns:
            Optional[Dict[str, Any]]: گزارش آنالیز به صورت دیکشنری یا None
        """
        normalized_text = text.strip()  # فرض بر اینکه متن ورودی پیش‌نیاز نرمال‌سازی شده است
        cache_key = f"analysis_report:{hash(normalized_text)}"
        cached = self.cache_manager.get_cached_result(cache_key)
        if cached:
            self.statistics["cache_hits"] += 1
            self.logger.info("گزارش آنالیز از کش بازیابی شد.")
            return json.loads(cached)

        # اگر در کش نباشد، از پایگاه داده بخوانید
        try:
            query = f"SELECT * FROM analysis_reports WHERE text = '{normalized_text}' LIMIT 1"
            result = self.database.execute_query(query)
            if result and len(result) > 0:
                report = result[0]
                # تبدیل رشته JSON به دیکشنری
                analysis = json.loads(report.get("analysis_result", "{}"))
                self.cache_manager.cache_result(cache_key, json.dumps(analysis), CONFIG.get("analysis_cache_ttl", 3600))
                self.statistics["analysis_reports_retrieved"] += 1
                self.logger.info("گزارش آنالیز از پایگاه داده بازیابی و در کش ذخیره شد.")
                return analysis
        except Exception as e:
            self.logger.error(f"خطا در دریافت گزارش آنالیز از پایگاه داده: {e}")
        self.statistics["cache_misses"] += 1
        return None

    def store_analysis_report(self, text: str, analysis_result: Dict[str, Any]) -> bool:
        """
        ذخیره گزارش آنالیز مربوط به یک متن در پایگاه داده و کش.
        این تابع گزارش را به صورت JSON در جدول analysis_reports ذخیره کرده و نتیجه را در کش نیز ثبت می‌کند.

        Args:
            text (str): متن ورودی که آنالیز شده است.
            analysis_result (Dict[str, Any]): نتیجه آنالیز به صورت دیکشنری.

        Returns:
            bool: True در صورت موفقیت در ذخیره‌سازی، False در غیر این صورت.
        """
        normalized_text = text.strip()
        report_id = f"report_{abs(hash(normalized_text + str(time.time()))) % 100000}"
        analysis_json = json.dumps(analysis_result)
        data = {
            "report_id": report_id,
            "text": normalized_text,
            "analysis_result": analysis_json,
            "timestamp": time.time()
        }
        try:
            self.database.insert_data("analysis_reports", data)
            cache_key = f"analysis_report:{hash(normalized_text)}"
            self.cache_manager.cache_result(cache_key, analysis_json, CONFIG.get("analysis_cache_ttl", 3600))
            self.statistics["analysis_reports_inserted"] += 1
            self.logger.info("گزارش آنالیز با موفقیت ذخیره و در کش ثبت شد.")
            return True
        except Exception as e:
            self.logger.error(f"خطا در ذخیره گزارش آنالیز: {e}")
            return False

    def get_all_analysis_reports(self) -> List[Dict[str, Any]]:
        """
        دریافت تمامی گزارش‌های ذخیره‌شده آنالیز از پایگاه داده.

        Returns:
            List[Dict[str, Any]]: لیستی از گزارش‌های آنالیز به صورت دیکشنری.
        """
        try:
            results = self.database.execute_query("SELECT * FROM analysis_reports")
            reports = []
            if results:
                for row in results:
                    report = {
                        "report_id": row.get("report_id"),
                        "text": row.get("text"),
                        "analysis_result": json.loads(row.get("analysis_result", "{}")),
                        "timestamp": row.get("timestamp")
                    }
                    reports.append(report)
            return reports
        except Exception as e:
            self.logger.error(f"خطا در دریافت تمامی گزارش‌های آنالیز: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        دریافت آمار عملکرد دسترسی به داده‌های آنالیز.

        Returns:
            Dict[str, Any]: شامل تعداد گزارش‌های خوانده شده، درج شده، ضرب کش و زمان شروع.
        """
        uptime = time.time() - self.statistics["start_time"]
        stats = {
            "analysis_reports_retrieved": self.statistics["analysis_reports_retrieved"],
            "analysis_reports_inserted": self.statistics["analysis_reports_inserted"],
            "cache_hits": self.statistics["cache_hits"],
            "cache_misses": self.statistics["cache_misses"],
            "uptime_seconds": uptime,
        }
        return stats

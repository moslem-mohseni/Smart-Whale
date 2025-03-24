"""
مدیریت جمع‌آورنده‌های مختلف داده و بهینه‌سازی استفاده از آن‌ها
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Type, Set, Union

from ai.core.messaging import DataType, DataSource
from ai.data.collectors.base.collector import BaseCollector
from ai.data.collectors.text.specialized.wiki_collector import WikiCollector
from ai.data.collectors.text.web_collector.general_crawler import GeneralWebCrawler

logger = logging.getLogger(__name__)


class CollectorManager:
    """
    مدیریت جمع‌آورنده‌های داده و بهینه‌سازی استفاده از آن‌ها
    """

    def __init__(self):
        """
        مقداردهی اولیه مدیر جمع‌آورنده‌ها
        """
        # ثبت انواع جمع‌آورنده‌ها
        self.collector_classes: Dict[str, Type[BaseCollector]] = {}
        self._register_collector_classes()

        # نمونه‌های فعال جمع‌آورنده‌ها
        self.active_collectors: Dict[str, BaseCollector] = {}

        # وضعیت‌های جمع‌آورنده‌ها
        self.collector_statuses: Dict[str, Dict[str, Any]] = {}

        # منابع در حال استفاده
        self.in_use_sources: Set[str] = set()

    def _register_collector_classes(self):
        """
        ثبت کلاس‌های جمع‌آورنده
        """
        # ثبت جمع‌آورنده‌های متن
        self.collector_classes["wiki"] = WikiCollector
        self.collector_classes["web"] = GeneralWebCrawler

        # سایر جمع‌آورنده‌ها در اینجا ثبت می‌شوند

        logger.info(f"✅ {len(self.collector_classes)} کلاس جمع‌آورنده ثبت شد")

    async def get_collector(
            self,
            collector_type: str,
            source_id: str,
            **params
    ) -> Optional[BaseCollector]:
        """
        دریافت یک نمونه جمع‌آورنده (ایجاد نمونه جدید یا استفاده مجدد از نمونه موجود)

        :param collector_type: نوع جمع‌آورنده (wiki, web, ...)
        :param source_id: شناسه منبع داده
        :param params: پارامترهای ایجاد جمع‌آورنده
        :return: نمونه جمع‌آورنده یا None در صورت عدم وجود
        """
        # بررسی وجود کلاس جمع‌آورنده
        if collector_type not in self.collector_classes:
            logger.error(f"❌ کلاس جمع‌آورنده '{collector_type}' یافت نشد")
            return None

        # شناسه یکتا برای این نمونه
        collector_key = f"{collector_type}:{source_id}"

        # بررسی وجود نمونه فعال
        if collector_key in self.active_collectors:
            logger.info(f"✅ استفاده مجدد از جمع‌آورنده '{collector_key}'")
            return self.active_collectors[collector_key]

        try:
            # ایجاد نمونه جدید
            collector_class = self.collector_classes[collector_type]
            collector = collector_class(**params)

            # ثبت در نمونه‌های فعال
            self.active_collectors[collector_key] = collector

            # ثبت وضعیت
            self.collector_statuses[collector_key] = {
                "type": collector_type,
                "source_id": source_id,
                "created_at": self._get_timestamp(),
                "status": "created",
                "params": params
            }

            # ثبت منبع در حال استفاده
            self.in_use_sources.add(source_id)

            logger.info(f"✅ جمع‌آورنده جدید '{collector_key}' ایجاد شد")

            return collector

        except Exception as e:
            logger.exception(f"❌ خطا در ایجاد جمع‌آورنده '{collector_type}': {str(e)}")
            return None

    async def release_collector(self, collector_type: str, source_id: str) -> bool:
        """
        آزادسازی یک جمع‌آورنده

        :param collector_type: نوع جمع‌آورنده
        :param source_id: شناسه منبع داده
        :return: نتیجه آزادسازی
        """
        collector_key = f"{collector_type}:{source_id}"

        if collector_key in self.active_collectors:
            try:
                # دریافت جمع‌آورنده
                collector = self.active_collectors[collector_key]

                # توقف جمع‌آوری
                await collector.stop_collection()

                # حذف از نمونه‌های فعال
                del self.active_collectors[collector_key]

                # به‌روزرسانی وضعیت
                if collector_key in self.collector_statuses:
                    self.collector_statuses[collector_key]["status"] = "released"
                    self.collector_statuses[collector_key]["released_at"] = self._get_timestamp()

                # حذف از منابع در حال استفاده
                if source_id in self.in_use_sources:
                    self.in_use_sources.remove(source_id)

                logger.info(f"✅ جمع‌آورنده '{collector_key}' آزاد شد")
                return True

            except Exception as e:
                logger.exception(f"❌ خطا در آزادسازی جمع‌آورنده '{collector_key}': {str(e)}")
                return False
        else:
            logger.warning(f"⚠ جمع‌آورنده '{collector_key}' یافت نشد")
            return False

    async def collect_data(
            self,
            data_type: Union[DataType, str],
            data_source: Union[DataSource, str],
            parameters: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        جمع‌آوری داده با استفاده از جمع‌آورنده مناسب

        :param data_type: نوع داده مورد نیاز
        :param data_source: منبع داده
        :param parameters: پارامترهای جمع‌آوری
        :return: داده‌های جمع‌آوری‌شده یا None در صورت خطا
        """
        # تبدیل Enum به رشته
        if isinstance(data_type, DataType):
            data_type_str = data_type.value
        else:
            data_type_str = data_type

        if isinstance(data_source, DataSource):
            data_source_str = data_source.value
        else:
            data_source_str = data_source

        # انتخاب نوع جمع‌آورنده بر اساس منبع و نوع داده
        collector_type = self._get_collector_type(data_source_str, data_type_str)
        if not collector_type:
            logger.error(f"❌ جمع‌آورنده مناسب برای منبع '{data_source_str}' و نوع '{data_type_str}' یافت نشد")
            return None

        # شناسه منبع
        source_id = parameters.get("query", parameters.get("title", parameters.get("url", "")))

        # آماده‌سازی پارامترهای جمع‌آورنده
        collector_params = self._prepare_collector_params(collector_type, parameters)

        try:
            # دریافت جمع‌آورنده
            collector = await self.get_collector(collector_type, source_id, **collector_params)
            if not collector:
                return None

            # جمع‌آوری داده
            data = await collector.collect_data()

            # آزادسازی جمع‌آورنده
            await self.release_collector(collector_type, source_id)

            return data

        except Exception as e:
            logger.exception(f"❌ خطا در جمع‌آوری داده: {str(e)}")

            # آزادسازی جمع‌آورنده در صورت خطا
            await self.release_collector(collector_type, source_id)

            return None

    def _get_collector_type(self, data_source: str, data_type: str) -> Optional[str]:
        """
        تعیین نوع جمع‌آورنده بر اساس منبع و نوع داده

        :param data_source: منبع داده
        :param data_type: نوع داده
        :return: نوع جمع‌آورنده یا None در صورت عدم وجود
        """
        # نگاشت منابع داده به انواع جمع‌آورنده
        source_to_collector = {
            "wiki": "wiki",
            "web": "web",
            "twitter": "twitter",
            "telegram": "telegram",
            "youtube": "youtube",
            "aparat": "aparat"
        }

        # برگرداندن نوع جمع‌آورنده مناسب
        if data_source in source_to_collector:
            collector_type = source_to_collector[data_source]
            if collector_type in self.collector_classes:
                return collector_type

        # نوع پیش‌فرض بر اساس نوع داده
        if data_type == "text":
            return "web"

        return None

    def _prepare_collector_params(self, collector_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        آماده‌سازی پارامترهای مناسب برای هر نوع جمع‌آورنده

        :param collector_type: نوع جمع‌آورنده
        :param parameters: پارامترهای درخواست
        :return: پارامترهای آماده‌شده برای ایجاد جمع‌آورنده
        """
        collector_params = {}

        if collector_type == "wiki":
            # پارامترهای WikiCollector
            collector_params["language"] = parameters.get("language", "fa")
            collector_params["max_length"] = parameters.get("max_length", 5000)
            # تنظیم عنوان
            collector_params["title"] = parameters.get("title", parameters.get("query", ""))

        elif collector_type == "web":
            # پارامترهای GeneralWebCrawler
            collector_params["source_name"] = "WebCrawler"
            # تنظیم URL
            start_url = parameters.get("url", parameters.get("query", ""))
            collector_params["start_url"] = start_url
            # تنظیم تعداد صفحات
            collector_params["max_pages"] = parameters.get("max_pages", 3)

        # پارامترهای سایر انواع جمع‌آورنده‌ها در اینجا اضافه می‌شوند

        return collector_params

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
collector_manager = CollectorManager()

"""
سرویس ارسال درخواست‌های جمع‌آوری داده به ماژول Data
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    create_data_request, DataRequest, kafka_service, TopicManager,
    DATA_REQUESTS_TOPIC
)

logger = logging.getLogger(__name__)


class DataService:
    """
    سرویس ارسال درخواست‌های داده از Balance به ماژول Data

    این سرویس وظیفه فرمت‌بندی درخواست‌ها، ارسال به ماژول Data، و مدیریت
    درخواست‌های دسته‌ای را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس داده
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.request_topic = DATA_REQUESTS_TOPIC
        self.request_counter = 0
        self._is_initialized = False

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس و اطمینان از وجود موضوعات مورد نیاز
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوع درخواست‌ها
        await self.topic_manager.ensure_topic_exists(self.request_topic)

        self._is_initialized = True
        logger.info(f"✅ سرویس داده آماده به کار است (موضوع: {self.request_topic})")

    async def request_data(
            self,
            model_id: str,
            query: str,
            data_type: Union[DataType, str] = DataType.TEXT,
            source_type: Optional[Union[DataSource, str]] = None,
            priority: Union[Priority, int] = Priority.MEDIUM,
            request_source: Union[RequestSource, str] = RequestSource.USER,
            **params
    ) -> Dict[str, Any]:
        """
        ارسال درخواست جمع‌آوری داده از یک منبع مشخص

        :param model_id: شناسه مدل درخواست‌کننده
        :param query: عبارت جستجو (URL یا عنوان مقاله)
        :param data_type: نوع داده مورد درخواست (TEXT, IMAGE, VIDEO, AUDIO, ...)
        :param source_type: نوع منبع (WEB, WIKI, TWITTER, TELEGRAM, ...)
        :param priority: اولویت درخواست (CRITICAL تا BACKGROUND)
        :param request_source: منبع درخواست (USER، MODEL، یا SYSTEM)
        :param params: سایر پارامترهای خاص منبع
        :return: اطلاعات درخواست ارسال شده
        """
        try:
            # اطمینان از آماده‌سازی اولیه
            await self.initialize()

            # تبدیل پارامترهای رشته‌ای به enum
            if isinstance(data_type, str):
                try:
                    data_type = DataType(data_type)
                except ValueError:
                    data_type = DataType.TEXT
                    logger.warning(f"⚠ نوع داده '{data_type}' نامعتبر است، از TEXT استفاده می‌شود")

            if source_type and isinstance(source_type, str):
                try:
                    source_type = DataSource(source_type)
                except ValueError:
                    source_type = None
                    logger.warning(f"⚠ نوع منبع '{source_type}' نامعتبر است، از تشخیص خودکار استفاده می‌شود")

            # تبدیل اولویت به enum
            if isinstance(priority, int):
                try:
                    priority = Priority(priority)
                except ValueError:
                    priority = Priority.MEDIUM

            # تبدیل منبع درخواست به enum
            if isinstance(request_source, str):
                try:
                    request_source = RequestSource(request_source)
                except ValueError:
                    request_source = RequestSource.USER

            # ایجاد موضوع نتایج برای مدل
            response_topic = await self.topic_manager.ensure_model_topic(model_id)

            # تنظیم پارامترهای خاص برای هر منبع
            parameters = {"query": query, **params}

            if source_type == DataSource.WIKI:
                # پارامترهای ویژه ویکی‌پدیا
                parameters["title"] = query  # عنوان مقاله
                if "language" not in parameters:
                    parameters["language"] = "fa"  # زبان پیش‌فرض

            elif source_type == DataSource.WEB:
                # پارامترهای ویژه وب
                if query.startswith("http"):
                    parameters["url"] = query  # آدرس وب مستقیم
                else:
                    parameters["search_term"] = query  # عبارت جستجو

            # افزایش شمارنده درخواست
            self.request_counter += 1

            # ایجاد درخواست
            request = create_data_request(
                model_id=model_id,
                data_type=data_type,
                data_source=source_type,
                parameters=parameters,
                priority=priority,
                response_topic=response_topic,
                operation=OperationType.FETCH_DATA,
                request_source=request_source
            )

            # ارسال درخواست
            success = await self.kafka_service.send_data_request(request, self.request_topic)

            if success:
                logger.info(f"✅ درخواست داده برای مدل '{model_id}' با موفقیت ارسال شد (منبع: {source_type})")
                return {
                    "status": "request_sent",
                    "request_id": request.metadata.request_id,
                    "timestamp": request.metadata.timestamp,
                    "model_id": model_id,
                    "data_type": data_type.value if isinstance(data_type, DataType) else data_type,
                    "source_type": source_type.value if isinstance(source_type, DataSource) else "auto",
                    "response_topic": response_topic,
                    "estimated_time": self._estimate_processing_time(source_type, parameters)
                }
            else:
                error_msg = f"❌ خطا در ارسال درخواست داده برای مدل '{model_id}'"
                logger.error(error_msg)
                return {"status": "error", "error": error_msg}

        except Exception as e:
            error_msg = f"❌ خطا در ارسال درخواست داده: {str(e)}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}

    async def request_batch(
            self,
            model_id: str,
            queries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        ارسال چندین درخواست به صورت دسته‌ای

        :param model_id: شناسه مدل درخواست‌کننده
        :param queries: لیست درخواست‌ها با پارامترهای مربوطه
        :return: لیست نتایج ارسال
        """
        results = []
        for query_info in queries:
            # استخراج پارامترهای هر درخواست
            query = query_info.get("query", "")
            data_type = query_info.get("data_type", DataType.TEXT)
            source_type = query_info.get("source_type", None)
            priority = query_info.get("priority", Priority.MEDIUM)
            request_source = query_info.get("request_source", RequestSource.USER)

            # حذف پارامترهای اصلی برای جداسازی پارامترهای اضافی
            params = {k: v for k, v in query_info.items()
                      if k not in ["query", "data_type", "source_type", "priority", "request_source"]}

            # ارسال درخواست
            result = await self.request_data(
                model_id=model_id,
                query=query,
                data_type=data_type,
                source_type=source_type,
                priority=priority,
                request_source=request_source,
                **params
            )

            results.append(result)

        return results

    def _estimate_processing_time(
            self,
            source_type: Optional[Union[DataSource, str]],
            parameters: Dict[str, Any]
    ) -> int:
        """
        تخمین زمان پردازش بر اساس نوع منبع و پارامترها (به ثانیه)

        :param source_type: نوع منبع داده
        :param parameters: پارامترهای درخواست
        :return: زمان تخمینی به ثانیه
        """
        # تبدیل رشته به enum
        if isinstance(source_type, str):
            try:
                source_type = DataSource(source_type)
            except ValueError:
                source_type = None

        if source_type == DataSource.WIKI:
            return 3  # زمان تخمینی برای ویکی‌پدیا

        elif source_type == DataSource.WEB:
            max_pages = int(parameters.get("max_pages", 3))
            return max_pages * 2  # زمان تخمینی برای صفحات وب

        elif source_type in [DataSource.TWITTER, DataSource.TELEGRAM]:
            count = int(parameters.get("count", 10))
            return max(5, count // 5)  # زمان تخمینی برای شبکه‌های اجتماعی

        elif source_type == DataSource.YOUTUBE or source_type == DataSource.APARAT:
            return 10  # زمان تخمینی برای ویدیو

        else:
            return 5  # زمان تخمینی پیش‌فرض


# نمونه Singleton برای استفاده در سراسر سیستم
data_service = DataService()

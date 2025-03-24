"""
سرویس اصلی جمع‌آوری داده از منابع مختلف بر اساس درخواست‌های دریافتی
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Type

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    DataRequest, DataResponse, create_data_response,
    kafka_service, TopicManager, DATA_REQUESTS_TOPIC,
    MESSAGE_STATUS_SUCCESS, MESSAGE_STATUS_ERROR
)
from ai.data.collectors.base.collector import BaseCollector
from ai.data.collectors.text.specialized.wiki_collector import WikiCollector
from ai.data.collectors.text.web_collector.general_crawler import GeneralWebCrawler

logger = logging.getLogger(__name__)


class DataCollectorService:
    """
    سرویس اصلی جمع‌آوری داده که درخواست‌ها را دریافت و به جمع‌آورنده‌های مناسب هدایت می‌کند
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس جمع‌آوری داده
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.request_topic = DATA_REQUESTS_TOPIC
        self._is_initialized = False
        self._shutdown_event = asyncio.Event()

        # ثبت جمع‌آورنده‌های پشتیبانی‌شده
        self.collectors: Dict[DataSource, Dict[DataType, Type[BaseCollector]]] = {}
        self._register_collectors()

        # نمونه‌های فعال جمع‌آورنده‌ها
        self.active_collectors: Dict[str, BaseCollector] = {}

    def _register_collectors(self):
        """
        ثبت انواع جمع‌آورنده‌های پشتیبانی‌شده
        """
        # تنظیم جمع‌آورنده‌های متن
        text_collectors = {
            DataSource.WIKI: WikiCollector,
            DataSource.WEB: GeneralWebCrawler,
            # در اینجا سایر جمع‌آورنده‌های متنی اضافه می‌شوند
        }

        # ثبت جمع‌آورنده‌های متن
        for source, collector_class in text_collectors.items():
            if source not in self.collectors:
                self.collectors[source] = {}
            self.collectors[source][DataType.TEXT] = collector_class

        # در اینجا می‌توان سایر انواع جمع‌آورنده‌ها را برای انواع داده دیگر ثبت کرد
        # مانند عکس، ویدیو، صوت و...

        logger.info(f"✅ {len(text_collectors)} جمع‌آورنده‌ی متن ثبت شد")

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس جمع‌آوری داده
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوع درخواست‌های داده
        await self.topic_manager.ensure_topic_exists(self.request_topic)

        self._is_initialized = True
        logger.info(f"✅ سرویس جمع‌آوری داده آماده به کار است (موضوع: {self.request_topic})")

    async def run(self):
        """
        اجرای سرویس جمع‌آوری داده و شروع گوش دادن به درخواست‌ها
        """
        await self.initialize()

        # اشتراک در موضوع درخواست‌های داده
        group_id = "data-collector-service"
        await self.kafka_service.subscribe(self.request_topic, group_id, self.process_request)

        logger.info(f"🔔 در حال گوش دادن به درخواست‌های داده روی موضوع {self.request_topic}")

        try:
            # انتظار برای درخواست توقف
            await self._shutdown_event.wait()
        finally:
            # پاکسازی منابع
            await self.shutdown()

    async def shutdown(self):
        """
        توقف سرویس جمع‌آوری داده و آزادسازی منابع
        """
        if not self._is_initialized:
            return

        logger.info("🛑 در حال توقف سرویس جمع‌آوری داده...")

        # بستن همه جمع‌آورنده‌های فعال
        for collector_key, collector in list(self.active_collectors.items()):
            await collector.stop_collection()

        # قطع اتصال از کافکا
        await self.kafka_service.disconnect()

        self._is_initialized = False
        self._shutdown_event.set()

        logger.info("✅ سرویس جمع‌آوری داده با موفقیت متوقف شد")

    async def process_request(self, request_data: Dict[str, Any]):
        """
        پردازش درخواست دریافتی از موضوع Kafka

        :param request_data: داده‌های درخواست
        """
        try:
            # تبدیل به نمونه DataRequest
            request = DataRequest.from_dict(request_data)

            # استخراج اطلاعات درخواست
            operation = request.payload.operation
            model_id = request.payload.model_id
            data_type = request.payload.data_type
            data_source = request.payload.data_source
            parameters = request.payload.parameters
            response_topic = request.payload.response_topic

            # بررسی عملیات
            if isinstance(operation, str):
                try:
                    operation = OperationType(operation)
                except ValueError:
                    operation = OperationType.FETCH_DATA

            # پردازش انواع عملیات
            if operation == OperationType.FETCH_DATA:
                # جمع‌آوری داده
                await self.collect_data(request, response_topic)
            else:
                # سایر عملیات
                logger.warning(f"⚠ عملیات '{operation}' پشتیبانی نمی‌شود")

                # ارسال پاسخ خطا
                error_response = create_data_response(
                    request_id=request.metadata.request_id,
                    model_id=model_id,
                    status=MESSAGE_STATUS_ERROR,
                    error_message=f"عملیات '{operation}' پشتیبانی نمی‌شود"
                )

                await self.kafka_service.send_data_response(error_response, response_topic)

        except Exception as e:
            logger.exception(f"❌ خطا در پردازش درخواست: {str(e)}")

    async def collect_data(self, request: DataRequest, response_topic: str):
        """
        جمع‌آوری داده بر اساس درخواست

        :param request: درخواست داده
        :param response_topic: موضوع پاسخ
        """
        # استخراج اطلاعات درخواست
        model_id = request.payload.model_id
        data_type = request.payload.data_type
        data_source = request.payload.data_source
        parameters = request.payload.parameters

        # تبدیل رشته‌ها به Enum
        if isinstance(data_type, str):
            try:
                data_type = DataType(data_type)
            except ValueError:
                data_type = DataType.TEXT

        if isinstance(data_source, str) and data_source:
            try:
                data_source = DataSource(data_source)
            except ValueError:
                data_source = self._detect_data_source(parameters)
        elif not data_source:
            data_source = self._detect_data_source(parameters)

        # بررسی وجود جمع‌آورنده مناسب
        if data_source not in self.collectors or data_type not in self.collectors[data_source]:
            error_message = f"جمع‌آورنده مناسب برای نوع داده '{data_type}' از منبع '{data_source}' یافت نشد"
            logger.error(f"❌ {error_message}")

            # ارسال پاسخ خطا
            error_response = create_data_response(
                request_id=request.metadata.request_id,
                model_id=model_id,
                status=MESSAGE_STATUS_ERROR,
                error_message=error_message
            )

            await self.kafka_service.send_data_response(error_response, response_topic)
            return

        try:
            # ایجاد نمونه جمع‌آورنده
            collector_class = self.collectors[data_source][data_type]
            collector_key = f"{model_id}:{data_source.value}:{data_type.value}"

            # تنظیم پارامترهای خاص برای هر نوع جمع‌آورنده
            collector_params = self._prepare_collector_params(data_source, parameters)

            # ایجاد نمونه جمع‌آورنده
            collector = collector_class(**collector_params)
            self.active_collectors[collector_key] = collector

            # زمان شروع
            import time
            start_time = time.time()

            # جمع‌آوری داده
            data = await collector.collect_data()

            # زمان پایان
            processing_time = time.time() - start_time

            # حذف از جمع‌آورنده‌های فعال
            if collector_key in self.active_collectors:
                del self.active_collectors[collector_key]

            # بررسی نتیجه
            if data:
                # ساخت پاسخ
                response = create_data_response(
                    request_id=request.metadata.request_id,
                    model_id=model_id,
                    status=MESSAGE_STATUS_SUCCESS,
                    data=data,
                    data_type=data_type,
                    data_source=data_source,
                    metrics={
                        "processing_time_ms": round(processing_time * 1000, 2)
                    }
                )

                # ارسال پاسخ
                await self.kafka_service.send_data_response(response, response_topic)
                logger.info(f"✅ داده با موفقیت برای مدل '{model_id}' از منبع '{data_source}' جمع‌آوری و ارسال شد")
            else:
                # ارسال پاسخ خطا
                error_message = f"داده‌ای از منبع '{data_source}' یافت نشد"
                error_response = create_data_response(
                    request_id=request.metadata.request_id,
                    model_id=model_id,
                    status=MESSAGE_STATUS_ERROR,
                    error_message=error_message,
                    data_type=data_type,
                    data_source=data_source
                )

                await self.kafka_service.send_data_response(error_response, response_topic)
                logger.warning(f"⚠ {error_message}")

        except Exception as e:
            logger.exception(f"❌ خطا در جمع‌آوری داده: {str(e)}")

            # ارسال پاسخ خطا
            error_response = create_data_response(
                request_id=request.metadata.request_id,
                model_id=model_id,
                status=MESSAGE_STATUS_ERROR,
                error_message=f"خطا در جمع‌آوری داده: {str(e)}",
                data_type=data_type,
                data_source=data_source
            )

            await self.kafka_service.send_data_response(error_response, response_topic)

    def _detect_data_source(self, parameters: Dict[str, Any]) -> DataSource:
        """
        تشخیص خودکار منبع داده بر اساس پارامترهای درخواست

        :param parameters: پارامترهای درخواست
        :return: منبع داده تشخیص داده شده
        """
        query = parameters.get("query", "")

        # بررسی پارامترهای خاص
        if "title" in parameters:
            return DataSource.WIKI
        elif "url" in parameters or query.startswith("http"):
            return DataSource.WEB
        elif "hashtag" in parameters or "username" in parameters:
            return DataSource.TWITTER
        elif "channel" in parameters:
            return DataSource.TELEGRAM
        elif "video_id" in parameters:
            return DataSource.YOUTUBE

        # تشخیص بر اساس query
        if query.startswith("http"):
            return DataSource.WEB

        # پیش‌فرض
        return DataSource.WEB

    def _prepare_collector_params(self, data_source: DataSource, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        آماده‌سازی پارامترهای مناسب برای هر نوع جمع‌آورنده

        :param data_source: منبع داده
        :param parameters: پارامترهای درخواست
        :return: پارامترهای آماده‌شده برای ایجاد جمع‌آورنده
        """
        collector_params = {}

        if data_source == DataSource.WIKI:
            # پارامترهای WikiCollector
            collector_params["language"] = parameters.get("language", "fa")
            collector_params["max_length"] = parameters.get("max_length", 5000)
            # تنظیم عنوان
            collector_params["title"] = parameters.get("title", parameters.get("query", ""))

        elif data_source == DataSource.WEB:
            # پارامترهای GeneralWebCrawler
            collector_params["source_name"] = "WebCrawler"
            # تنظیم URL
            start_url = parameters.get("url", parameters.get("query", ""))
            collector_params["start_url"] = start_url
            # تنظیم تعداد صفحات
            collector_params["max_pages"] = parameters.get("max_pages", 3)

        # پارامترهای سایر منابع در اینجا اضافه می‌شوند

        return collector_params


# نمونه Singleton برای استفاده در سراسر سیستم
data_collector_service = DataCollectorService()

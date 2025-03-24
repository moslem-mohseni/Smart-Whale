"""
سرویس مدیریت درخواست‌های داده از ماژول بالانس

این سرویس وظیفه ارسال درخواست‌های داده به ماژول بالانس و دریافت
نتایج جمع‌آوری داده را بر عهده دارد.
"""
import json
import logging
import asyncio
from uuid import uuid4
from typing import Dict, Any, List, Callable, Optional, Set, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    create_data_request, DataRequest, DataResponse,
    kafka_service, TopicManager, MODELS_REQUESTS_TOPIC
)

logger = logging.getLogger(__name__)


class DataRequestService:
    """
    سرویس مدیریت درخواست‌های داده از ماژول بالانس

    این سرویس وظیفه فرمت‌بندی درخواست‌های داده، ارسال به ماژول بالانس و
    دریافت نتایج جمع‌آوری داده را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس درخواست داده
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.models_requests_topic = MODELS_REQUESTS_TOPIC
        self.request_counter = 0
        self.pending_requests: Dict[str, Dict[str, Any]] = {}  # درخواست‌های در حال انتظار
        self.response_handlers: Dict[str, Callable] = {}  # پردازشگرهای پاسخ داده بر اساس مدل
        self._is_initialized = False

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس و اطمینان از وجود موضوعات مورد نیاز
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوع درخواست‌های مدل
        await self.topic_manager.ensure_topic_exists(self.models_requests_topic)

        self._is_initialized = True
        logger.info(f"✅ سرویس درخواست داده آماده به کار است (موضوع: {self.models_requests_topic})")

    async def request_data(
            self,
            model_id: str,
            query: str,
            data_type: Union[DataType, str] = DataType.TEXT,
            source_type: Optional[Union[DataSource, str]] = None,
            priority: Union[Priority, int] = Priority.MEDIUM,
            request_source: Union[RequestSource, str] = RequestSource.MODEL,
            **params
    ) -> Dict[str, Any]:
        """
        ارسال درخواست جمع‌آوری داده به ماژول بالانس

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
                    request_source = RequestSource.MODEL

            # اطمینان از وجود موضوع نتایج برای مدل
            response_topic = await self.topic_manager.ensure_model_topic(model_id)

            # تنظیم پارامترهای درخواست
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

            # ثبت در لیست درخواست‌های در حال انتظار
            request_id = request.metadata.request_id
            self.pending_requests[request_id] = {
                "request_id": request_id,
                "model_id": model_id,
                "timestamp": request.metadata.timestamp,
                "query": query,
                "data_type": data_type.value if isinstance(data_type, DataType) else data_type,
                "source_type": source_type.value if isinstance(source_type, DataSource) else "auto",
                "parameters": parameters,
                "status": "pending",
                "response": None
            }

            # ارسال درخواست
            success = await self.kafka_service.send_message(self.models_requests_topic, request.to_dict())

            if success:
                logger.info(
                    f"✅ درخواست داده '{request_id}' از مدل '{model_id}' با موفقیت ارسال شد (منبع: {source_type})")

                # به‌روزرسانی وضعیت درخواست
                self.pending_requests[request_id]["status"] = "sent"

                # تنظیم تایمر برای اتمام خودکار درخواست پس از مهلت زمانی
                timeout = params.get("timeout", 120)  # مهلت زمانی پیش‌فرض: 120 ثانیه
                asyncio.create_task(self._expire_request(request_id, timeout))

                return {
                    "status": "request_sent",
                    "request_id": request_id,
                    "timestamp": request.metadata.timestamp,
                    "model_id": model_id,
                    "data_type": data_type.value if isinstance(data_type, DataType) else data_type,
                    "source_type": source_type.value if isinstance(source_type, DataSource) else "auto",
                    "estimated_time": self._estimate_processing_time(source_type, parameters)
                }
            else:
                error_msg = f"❌ خطا در ارسال درخواست داده از مدل '{model_id}'"
                logger.error(error_msg)

                # حذف از لیست درخواست‌های در حال انتظار
                if request_id in self.pending_requests:
                    self.pending_requests[request_id]["status"] = "failed"

                return {"status": "error", "error": error_msg, "request_id": request_id}

        except Exception as e:
            error_msg = f"❌ خطا در ارسال درخواست داده: {str(e)}"
            logger.exception(error_msg)
            return {"status": "error", "error": error_msg}

    async def request_batch(
            self,
            model_id: str,
            queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        ارسال چندین درخواست به صورت دسته‌ای

        :param model_id: شناسه مدل درخواست‌کننده
        :param queries: لیست درخواست‌ها با پارامترهای مربوطه
        :return: نتیجه عملیات دسته‌ای
        """
        results = []
        successful = 0
        failed = 0

        batch_id = str(uuid4())

        for idx, query_info in enumerate(queries):
            # استخراج پارامترهای هر درخواست
            query = query_info.get("query", "")
            if not query:
                logger.warning(f"⚠ درخواست شماره {idx + 1} فاقد پارامتر 'query' است و نادیده گرفته می‌شود")
                failed += 1
                continue

            data_type = query_info.get("data_type", DataType.TEXT)
            source_type = query_info.get("source_type", None)
            priority = query_info.get("priority", Priority.MEDIUM)
            request_source = query_info.get("request_source", RequestSource.MODEL)

            # حذف پارامترهای اصلی برای جداسازی پارامترهای اضافی
            params = {k: v for k, v in query_info.items()
                      if k not in ["query", "data_type", "source_type", "priority", "request_source"]}

            # افزودن شناسه دسته
            params["batch_id"] = batch_id
            params["batch_index"] = idx

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
            if result.get("status") == "request_sent":
                successful += 1
            else:
                failed += 1

        return {
            "batch_id": batch_id,
            "status": "batch_sent",
            "total_requests": len(queries),
            "successful": successful,
            "failed": failed,
            "model_id": model_id,
            "requests": results
        }

    async def subscribe_to_data_responses(
            self,
            model_id: str,
            handler: Callable
    ) -> Dict[str, Any]:
        """
        اشتراک در موضوع پاسخ‌های داده برای یک مدل

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده پاسخ‌های داده
        :return: نتیجه عملیات اشتراک
        """
        await self.initialize()

        # دریافت موضوع نتایج مدل
        response_topic = await self.topic_manager.ensure_model_topic(model_id)

        # ثبت پردازشگر پاسخ برای مدل
        self.response_handlers[model_id] = handler

        # تعریف پردازشگر پیام‌های دریافتی
        async def response_handler(message_data: Dict[str, Any]):
            # تبدیل به نمونه DataResponse
            try:
                response = DataResponse.from_dict(message_data)

                # بررسی وجود درخواست در لیست درخواست‌های در حال انتظار
                request_id = response.metadata.request_id
                if request_id in self.pending_requests:
                    # به‌روزرسانی وضعیت درخواست
                    self.pending_requests[request_id]["status"] = "received"
                    self.pending_requests[request_id]["response"] = response.to_dict()

                # فراخوانی پردازشگر اختصاصی مدل
                await handler(response.to_dict())

                logger.info(f"📥 پاسخ داده برای درخواست '{request_id}' از مدل '{model_id}' دریافت شد")
            except Exception as e:
                logger.error(f"❌ خطا در پردازش پاسخ داده: {str(e)}")

        # اشتراک در موضوع نتایج مدل
        group_id = f"model-{model_id}-responses"
        await self.kafka_service.subscribe(response_topic, group_id, response_handler)

        logger.info(f"✅ مدل '{model_id}' با موفقیت در موضوع پاسخ‌های داده مشترک شد")

        return {
            "model_id": model_id,
            "status": "subscribed",
            "topic": response_topic
        }

    async def get_request_status(
            self,
            request_id: str
    ) -> Dict[str, Any]:
        """
        دریافت وضعیت یک درخواست داده

        :param request_id: شناسه درخواست
        :return: اطلاعات وضعیت درخواست
        """
        if request_id not in self.pending_requests:
            return {
                "request_id": request_id,
                "status": "not_found",
                "error": "درخواست با این شناسه یافت نشد"
            }

        return self.pending_requests[request_id]

    async def cancel_request(
            self,
            request_id: str
    ) -> Dict[str, Any]:
        """
        لغو یک درخواست داده در حال انتظار

        :param request_id: شناسه درخواست
        :return: نتیجه عملیات لغو
        """
        if request_id not in self.pending_requests:
            return {
                "request_id": request_id,
                "status": "not_found",
                "error": "درخواست با این شناسه یافت نشد"
            }

        # بررسی وضعیت فعلی درخواست
        current_status = self.pending_requests[request_id]["status"]
        if current_status in ["received", "expired", "cancelled", "failed"]:
            return {
                "request_id": request_id,
                "status": "invalid_operation",
                "current_status": current_status,
                "error": f"درخواست در وضعیت '{current_status}' قابل لغو نیست"
            }

        # به‌روزرسانی وضعیت درخواست
        self.pending_requests[request_id]["status"] = "cancelled"
        logger.info(f"🛑 درخواست '{request_id}' لغو شد")

        return {
            "request_id": request_id,
            "status": "cancelled",
            "previous_status": current_status
        }

    async def _expire_request(self, request_id: str, timeout: int):
        """
        تابع داخلی برای پایان دادن به درخواست پس از گذشت مهلت زمانی

        :param request_id: شناسه درخواست
        :param timeout: مدت زمان انتظار (ثانیه)
        """
        await asyncio.sleep(timeout)

        if request_id in self.pending_requests:
            current_status = self.pending_requests[request_id]["status"]
            if current_status == "sent" or current_status == "pending":
                self.pending_requests[request_id]["status"] = "expired"
                logger.warning(f"⏱ درخواست '{request_id}' به دلیل عدم دریافت پاسخ در مهلت مقرر منقضی شد")

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

        elif source_type in [DataSource.YOUTUBE, DataSource.APARAT]:
            return 10  # زمان تخمینی برای ویدیو

        else:
            return 5  # زمان تخمینی پیش‌فرض

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
data_request_service = DataRequestService()

"""
سرویس مدیریت ارتباط با مدل‌ها و پردازش درخواست‌های آن‌ها
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Set, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    DataRequest, DataResponse, is_valid_data_request,
    kafka_service, TopicManager, MODELS_REQUESTS_TOPIC
)
from ai.balance.services.data_service import data_service

logger = logging.getLogger(__name__)


class ModelService:
    """
    سرویس مدیریت ارتباط با مدل‌ها و پردازش درخواست‌های آن‌ها

    این سرویس وظیفه دریافت درخواست‌های مدل‌ها، ارسال آن‌ها به ماژول Data
    و مدیریت اشتراک مدل‌ها در موضوعات کافکا را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس مدل
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.models_requests_topic = MODELS_REQUESTS_TOPIC
        self.registered_models: Set[str] = set()
        self.model_handlers: Dict[str, Callable] = {}
        self._is_initialized = False

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس و اطمینان از وجود موضوعات مورد نیاز
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوع درخواست‌های مدل‌ها
        await self.topic_manager.ensure_topic_exists(self.models_requests_topic)

        # آماده‌سازی سرویس داده
        await data_service.initialize()

        self._is_initialized = True
        logger.info(f"✅ سرویس مدل آماده به کار است (موضوع: {self.models_requests_topic})")

    async def register_model(self, model_id: str, handler: Optional[Callable] = None):
        """
        ثبت یک مدل در سیستم و ایجاد موضوع اختصاصی برای آن

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده پیام‌های ارسال شده به مدل (اختیاری)
        """
        await self.initialize()

        # اطمینان از وجود موضوع نتایج برای مدل
        model_topic = await self.topic_manager.ensure_model_topic(model_id)

        # ثبت مدل در لیست مدل‌های ثبت‌شده
        self.registered_models.add(model_id)

        # ثبت پردازشگر اختصاصی (اگر وجود داشته باشد)
        if handler:
            self.model_handlers[model_id] = handler

        logger.info(f"✅ مدل '{model_id}' با موفقیت ثبت شد (موضوع: {model_topic})")

        return {
            "model_id": model_id,
            "status": "registered",
            "topic": model_topic
        }

    async def unregister_model(self, model_id: str):
        """
        حذف ثبت یک مدل از سیستم

        :param model_id: شناسه مدل
        """
        if model_id not in self.registered_models:
            logger.warning(f"⚠ مدل '{model_id}' قبلاً ثبت نشده است")
            return {
                "model_id": model_id,
                "status": "not_registered"
            }

        # حذف از مدل‌های ثبت‌شده
        self.registered_models.remove(model_id)

        # حذف پردازشگر اختصاصی
        if model_id in self.model_handlers:
            del self.model_handlers[model_id]

        logger.info(f"✅ ثبت مدل '{model_id}' با موفقیت حذف شد")

        return {
            "model_id": model_id,
            "status": "unregistered"
        }

    async def process_model_request(self, request_data: Dict[str, Any]):
        """
        پردازش درخواست دریافتی از مدل

        :param request_data: داده‌های درخواست
        """
        try:
            # اعتبارسنجی درخواست
            if not is_valid_data_request(request_data):
                logger.error("❌ فرمت درخواست مدل نامعتبر است")
                return

            # ساخت نمونه DataRequest
            request = DataRequest.from_dict(request_data)

            # استخراج اطلاعات اصلی
            model_id = request.payload.model_id
            operation = request.payload.operation
            data_type = request.payload.data_type
            data_source = request.payload.data_source
            parameters = request.payload.parameters

            # بررسی ثبت مدل
            if model_id not in self.registered_models:
                await self.register_model(model_id)

            # پردازش عملیات
            if operation == OperationType.FETCH_DATA:
                # درخواست جمع‌آوری داده
                query = parameters.get("query", "")
                if not query:
                    logger.error(f"❌ پارامتر 'query' در درخواست مدل '{model_id}' یافت نشد")
                    return

                # پیش‌فرض منبع درخواست به MODEL
                request_source = request.metadata.request_source
                if not request_source or request_source == RequestSource.USER.value:
                    request_source = RequestSource.MODEL

                # ارسال درخواست به سرویس داده
                await data_service.request_data(
                    model_id=model_id,
                    query=query,
                    data_type=data_type,
                    source_type=data_source,
                    priority=request.metadata.priority,
                    request_source=request_source,
                    **parameters
                )
            else:
                logger.warning(f"⚠ عملیات '{operation}' برای مدل '{model_id}' پشتیبانی نمی‌شود")

        except Exception as e:
            logger.exception(f"❌ خطا در پردازش درخواست مدل: {str(e)}")

    async def subscribe_to_model_requests(self):
        """
        اشتراک در موضوع درخواست‌های مدل‌ها برای دریافت و پردازش درخواست‌ها
        """
        await self.initialize()

        # تعریف پردازشگر پیام‌های دریافتی
        async def request_handler(message_data: Dict[str, Any]):
            await self.process_model_request(message_data)

        # اشتراک در موضوع درخواست‌های مدل‌ها
        group_id = "balance-model-service"
        await self.kafka_service.subscribe(self.models_requests_topic, group_id, request_handler)

        logger.info(f"✅ اشتراک در موضوع درخواست‌های مدل‌ها ({self.models_requests_topic}) انجام شد")

    async def subscribe_to_model_results(self, model_id: str, handler: Optional[Callable] = None):
        """
        اشتراک در موضوع نتایج یک مدل برای دریافت و پردازش نتایج جمع‌آوری داده

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده نتایج (اختیاری)
        """
        # ثبت مدل در صورت نیاز
        if model_id not in self.registered_models:
            await self.register_model(model_id, handler)
        elif handler:
            # به‌روزرسانی پردازشگر
            self.model_handlers[model_id] = handler

        # دریافت موضوع نتایج مدل
        model_topic = self.topic_manager.get_model_result_topic(model_id)

        # تعریف پردازشگر پیام‌های دریافتی
        async def response_handler(message_data: Dict[str, Any]):
            # اگر پردازشگر اختصاصی وجود دارد، از آن استفاده می‌شود
            if model_id in self.model_handlers and self.model_handlers[model_id]:
                await self.model_handlers[model_id](message_data)
            else:
                # پردازش پیش‌فرض
                logger.info(f"📥 نتیجه داده برای مدل '{model_id}' دریافت شد")

        # اشتراک در موضوع نتایج مدل
        group_id = f"balance-model-{model_id}"
        await self.kafka_service.subscribe(model_topic, group_id, response_handler)

        logger.info(f"✅ اشتراک در موضوع نتایج مدل '{model_id}' ({model_topic}) انجام شد")

    async def forward_result_to_model(self, model_id: str, result_data: Dict[str, Any]):
        """
        ارسال مستقیم نتیجه به یک مدل

        :param model_id: شناسه مدل
        :param result_data: داده‌های نتیجه
        """
        # ثبت مدل در صورت نیاز
        if model_id not in self.registered_models:
            await self.register_model(model_id)

        # دریافت موضوع نتایج مدل
        model_topic = self.topic_manager.get_model_result_topic(model_id)

        # ارسال به موضوع مدل
        success = await self.kafka_service.send_message(model_topic, result_data)

        if success:
            logger.info(f"✅ نتیجه با موفقیت به مدل '{model_id}' ارسال شد")
        else:
            logger.error(f"❌ خطا در ارسال نتیجه به مدل '{model_id}'")

        return success


# نمونه Singleton برای استفاده در سراسر سیستم
model_service = ModelService()

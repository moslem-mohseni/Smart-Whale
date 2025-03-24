"""
سرویس مدیریت یکپارچه پیام‌رسانی در ماژول Models

این سرویس وظیفه هماهنگی بین سرویس‌های فدراسیون، کاربر و درخواست داده
در ماژول Models را بر عهده دارد.
"""
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Set, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    kafka_service, TopicManager, MODELS_REQUESTS_TOPIC,
    MODELS_FEDERATION_TOPIC, USER_REQUESTS_TOPIC, USER_RESPONSES_TOPIC,
    DATA_REQUESTS_TOPIC, BALANCE_METRICS_TOPIC, BALANCE_EVENTS_TOPIC
)

from .federation_service import federation_service
from .user_service import user_service
from .data_request_service import data_request_service

logger = logging.getLogger(__name__)


class MessagingService:
    """
    سرویس مدیریت یکپارچه پیام‌رسانی در ماژول Models

    این سرویس وظیفه راه‌اندازی، نظارت و مدیریت تمام ارتباطات پیام‌رسانی
    در ماژول Models را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس پیام‌رسانی
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.federation_service = federation_service
        self.user_service = user_service
        self.data_request_service = data_request_service
        self.registered_models: Set[str] = set()
        self._is_initialized = False
        self._shutdown_event = asyncio.Event()

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس پیام‌رسانی و راه‌اندازی همه سرویس‌های وابسته
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوعات اصلی
        await self.topic_manager.initialize_all_topics()

        # راه‌اندازی سرویس‌های وابسته
        await self.federation_service.initialize()
        await self.user_service.initialize()
        await self.data_request_service.initialize()

        self._is_initialized = True
        logger.info("✅ سرویس پیام‌رسانی ماژول Models آماده به کار است")

    async def register_model(
            self,
            model_id: str,
            federation_handler: Optional[Callable] = None,
            data_handler: Optional[Callable] = None,
            user_handler: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        ثبت یک مدل در تمام سرویس‌های پیام‌رسانی

        :param model_id: شناسه مدل
        :param federation_handler: پردازشگر پیام‌های فدراسیونی (اختیاری)
        :param data_handler: پردازشگر پاسخ‌های داده (اختیاری)
        :param user_handler: پردازشگر درخواست‌های کاربر (اختیاری)
        :return: نتیجه ثبت مدل
        """
        await self.initialize()

        results = {}

        # ثبت در سرویس فدراسیون
        federation_result = await self.federation_service.register_model(model_id, federation_handler)
        await self.federation_service.subscribe_to_federation(model_id, federation_handler)
        results["federation"] = federation_result

        # ثبت در سرویس درخواست داده
        if data_handler:
            data_result = await self.data_request_service.subscribe_to_data_responses(model_id, data_handler)
            results["data"] = data_result

        # ثبت در سرویس کاربر
        if user_handler:
            user_result = await self.user_service.subscribe_to_user_requests(model_id, user_handler)
            results["user"] = user_result

        # ثبت در لیست مدل‌های ثبت‌شده
        self.registered_models.add(model_id)

        logger.info(f"✅ مدل '{model_id}' با موفقیت در تمام سرویس‌های پیام‌رسانی ثبت شد")

        return {
            "model_id": model_id,
            "status": "registered",
            "services": results
        }

    async def request_data(
            self,
            model_id: str,
            query: str,
            data_type: Union[DataType, str] = DataType.TEXT,
            source_type: Optional[Union[DataSource, str]] = None,
            priority: Union[Priority, int] = Priority.MEDIUM,
            **params
    ) -> Dict[str, Any]:
        """
        ارسال درخواست داده از طریق سرویس درخواست داده

        :param model_id: شناسه مدل درخواست‌کننده
        :param query: عبارت جستجو
        :param data_type: نوع داده مورد درخواست
        :param source_type: نوع منبع داده
        :param priority: اولویت درخواست
        :param params: سایر پارامترهای خاص منبع
        :return: نتیجه ارسال درخواست
        """
        # اطمینان از ثبت مدل
        if model_id not in self.registered_models:
            await self.register_model(model_id)

        # استفاده از سرویس درخواست داده
        return await self.data_request_service.request_data(
            model_id=model_id,
            query=query,
            data_type=data_type,
            source_type=source_type,
            priority=priority,
            request_source=RequestSource.MODEL,
            **params
        )

    async def request_batch_data(
            self,
            model_id: str,
            queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        ارسال درخواست‌های دسته‌ای داده

        :param model_id: شناسه مدل درخواست‌کننده
        :param queries: لیست درخواست‌ها
        :return: نتیجه ارسال درخواست‌های دسته‌ای
        """
        # اطمینان از ثبت مدل
        if model_id not in self.registered_models:
            await self.register_model(model_id)

        # استفاده از سرویس درخواست داده
        return await self.data_request_service.request_batch(
            model_id=model_id,
            queries=queries
        )

    async def share_knowledge(
            self,
            source_model_id: str,
            target_model_id: Optional[str] = None,
            knowledge_data: Dict[str, Any] = None,
            knowledge_type: str = "general",
            privacy_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        اشتراک‌گذاری دانش از یک مدل با سایر مدل‌ها

        :param source_model_id: شناسه مدل منبع
        :param target_model_id: شناسه مدل هدف (None برای همه مدل‌ها)
        :param knowledge_data: داده‌های دانش
        :param knowledge_type: نوع دانش
        :param privacy_level: سطح حریم خصوصی
        :return: نتیجه اشتراک‌گذاری
        """
        # اطمینان از ثبت مدل
        if source_model_id not in self.registered_models:
            await self.register_model(source_model_id)

        # استفاده از سرویس فدراسیون
        return await self.federation_service.share_knowledge(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            knowledge_data=knowledge_data,
            knowledge_type=knowledge_type,
            privacy_level=privacy_level
        )

    async def request_collaboration(
            self,
            source_model_id: str,
            problem_data: Dict[str, Any],
            target_models: Optional[List[str]] = None,
            collaboration_type: str = "general",
            priority: Union[Priority, int] = Priority.MEDIUM,
            timeout: int = 60
    ) -> Dict[str, Any]:
        """
        درخواست همکاری از سایر مدل‌ها

        :param source_model_id: شناسه مدل درخواست‌کننده
        :param problem_data: داده‌های مسئله
        :param target_models: لیست مدل‌های هدف
        :param collaboration_type: نوع همکاری
        :param priority: اولویت درخواست
        :param timeout: مهلت زمانی (ثانیه)
        :return: نتیجه درخواست همکاری
        """
        # اطمینان از ثبت مدل
        if source_model_id not in self.registered_models:
            await self.register_model(source_model_id)

        # استفاده از سرویس فدراسیون
        return await self.federation_service.request_collaboration(
            source_model_id=source_model_id,
            problem_data=problem_data,
            target_models=target_models,
            collaboration_type=collaboration_type,
            priority=priority,
            timeout=timeout
        )

    async def respond_to_collaboration(
            self,
            source_model_id: str,
            target_model_id: str,
            request_id: str,
            response_data: Dict[str, Any],
            status: str = "success"
    ) -> Dict[str, Any]:
        """
        پاسخ به درخواست همکاری

        :param source_model_id: شناسه مدل پاسخ‌دهنده
        :param target_model_id: شناسه مدل درخواست‌کننده
        :param request_id: شناسه درخواست
        :param response_data: داده‌های پاسخ
        :param status: وضعیت پاسخ
        :return: نتیجه ارسال پاسخ
        """
        # اطمینان از ثبت مدل‌ها
        if source_model_id not in self.registered_models:
            await self.register_model(source_model_id)

        # استفاده از سرویس فدراسیون
        return await self.federation_service.respond_to_collaboration(
            source_model_id=source_model_id,
            target_model_id=target_model_id,
            request_id=request_id,
            response_data=response_data,
            status=status
        )

    async def stream_response(
            self,
            session_id: str,
            response_chunk: str,
            is_final: bool = False,
            chunk_id: Optional[int] = None,
            thinking: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        استریم بخشی از پاسخ به کاربر

        :param session_id: شناسه نشست
        :param response_chunk: بخش پاسخ
        :param is_final: آیا بخش نهایی است
        :param chunk_id: شناسه بخش (اختیاری)
        :param thinking: فرآیند تفکر (اختیاری)
        :param metadata: اطلاعات اضافی (اختیاری)
        :return: نتیجه استریم
        """
        # استفاده از سرویس کاربر
        return await self.user_service.stream_response(
            session_id=session_id,
            response_chunk=response_chunk,
            is_final=is_final,
            chunk_id=chunk_id,
            thinking=thinking,
            metadata=metadata
        )

    async def send_thinking_process(
            self,
            session_id: str,
            thinking_data: str,
            is_final: bool = False,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        ارسال فرآیند تفکر مدل به کاربر

        :param session_id: شناسه نشست
        :param thinking_data: داده‌های فرآیند تفکر
        :param is_final: آیا بخش نهایی است
        :param metadata: اطلاعات اضافی (اختیاری)
        :return: نتیجه ارسال
        """
        # استفاده از سرویس کاربر
        return await self.user_service.send_thinking_process(
            session_id=session_id,
            thinking_data=thinking_data,
            is_final=is_final,
            metadata=metadata
        )

    async def register_user_session(
            self,
            session_id: str,
            model_id: str,
            user_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        ثبت یک نشست کاربر

        :param session_id: شناسه نشست
        :param model_id: شناسه مدل
        :param user_id: شناسه کاربر (اختیاری)
        :param metadata: اطلاعات اضافی (اختیاری)
        :return: نتیجه ثبت نشست
        """
        # اطمینان از ثبت مدل
        if model_id not in self.registered_models:
            await self.register_model(model_id)

        # استفاده از سرویس کاربر
        return await self.user_service.register_user_session(
            session_id=session_id,
            model_id=model_id,
            user_id=user_id,
            metadata=metadata
        )

    async def end_user_session(
            self,
            session_id: str,
            reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        پایان دادن به نشست کاربر

        :param session_id: شناسه نشست
        :param reason: دلیل پایان (اختیاری)
        :return: نتیجه پایان نشست
        """
        # استفاده از سرویس کاربر
        return await self.user_service.end_user_session(
            session_id=session_id,
            reason=reason
        )

    async def publish_metric(
            self,
            model_id: str,
            metric_name: str,
            metric_data: Dict[str, Any]
    ) -> bool:
        """
        انتشار یک متریک مربوط به مدل

        :param model_id: شناسه مدل
        :param metric_name: نام متریک
        :param metric_data: داده‌های متریک
        :return: نتیجه انتشار
        """
        await self.initialize()

        # افزودن شناسه مدل به داده‌های متریک
        extended_data = {
            "model_id": model_id,
            "source": "models",
            "timestamp": self._get_timestamp(),
            **metric_data
        }

        # آماده‌سازی پیام متریک
        message_data = {
            "metric_name": metric_name,
            "timestamp": self._get_timestamp(),
            "data": extended_data
        }

        # انتشار متریک
        return await self.kafka_service.send_message(BALANCE_METRICS_TOPIC, message_data)

    async def publish_event(
            self,
            model_id: str,
            event_type: str,
            event_data: Dict[str, Any]
    ) -> bool:
        """
        انتشار یک رویداد مربوط به مدل

        :param model_id: شناسه مدل
        :param event_type: نوع رویداد
        :param event_data: داده‌های رویداد
        :return: نتیجه انتشار
        """
        await self.initialize()

        # افزودن شناسه مدل به داده‌های رویداد
        extended_data = {
            "model_id": model_id,
            "source": "models",
            "timestamp": self._get_timestamp(),
            **event_data
        }

        # آماده‌سازی پیام رویداد
        message_data = {
            "event_type": event_type,
            "timestamp": self._get_timestamp(),
            "data": extended_data
        }

        # انتشار رویداد
        return await self.kafka_service.send_message(BALANCE_EVENTS_TOPIC, message_data)

    async def run(self):
        """
        شروع سرویس پیام‌رسانی و اجرای حلقه اصلی آن
        """
        await self.initialize()

        try:
            logger.info("🚀 سرویس پیام‌رسانی ماژول Models در حال اجرا است")

            # انتظار برای درخواست توقف
            await self._shutdown_event.wait()

        except Exception as e:
            logger.exception(f"❌ خطا در سرویس پیام‌رسانی: {str(e)}")
        finally:
            # پاکسازی منابع
            await self.shutdown()

    async def shutdown(self):
        """
        توقف سرویس پیام‌رسانی و آزادسازی منابع
        """
        if not self._is_initialized:
            return

        logger.info("🛑 در حال توقف سرویس پیام‌رسانی...")

        # قطع اتصال از کافکا
        await self.kafka_service.disconnect()

        self._is_initialized = False
        self._shutdown_event.set()

        logger.info("✅ سرویس پیام‌رسانی با موفقیت متوقف شد")

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
messaging_service = MessagingService()

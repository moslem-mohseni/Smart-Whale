"""
سرویس مدیریت یکپارچه پیام‌رسانی در ماژول Balance
"""
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    kafka_service, TopicManager, MODELS_REQUESTS_TOPIC,
    DATA_REQUESTS_TOPIC, BALANCE_METRICS_TOPIC, BALANCE_EVENTS_TOPIC
)
from ai.balance.services.data_service import data_service
from ai.balance.services.model_service import model_service

logger = logging.getLogger(__name__)


class MessagingService:
    """
    سرویس مدیریت یکپارچه پیام‌رسانی در ماژول Balance

    این سرویس وظیفه راه‌اندازی، نظارت و مدیریت تمام ارتباطات پیام‌رسانی
    در ماژول Balance را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس پیام‌رسانی
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.data_service = data_service
        self.model_service = model_service
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
        await self.data_service.initialize()
        await self.model_service.initialize()

        # اشتراک در موضوعات اصلی
        await self.model_service.subscribe_to_model_requests()

        self._is_initialized = True
        logger.info("✅ سرویس پیام‌رسانی آماده به کار است")

    async def register_model(self, model_id: str, handler: Optional[Callable] = None):
        """
        ثبت یک مدل در سیستم و اشتراک در موضوع نتایج آن

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده نتایج (اختیاری)
        """
        await self.initialize()

        # ثبت مدل در سرویس مدل
        result = await self.model_service.register_model(model_id, handler)

        # اشتراک در موضوع نتایج مدل
        await self.model_service.subscribe_to_model_results(model_id, handler)

        return result

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
        # اطمینان از ثبت مدل
        if model_id not in self.model_service.registered_models:
            await self.register_model(model_id)

        # ارسال درخواست به سرویس داده
        return await self.data_service.request_data(
            model_id=model_id,
            query=query,
            data_type=data_type,
            source_type=source_type,
            priority=priority,
            request_source=request_source,
            **params
        )

    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """
        انتشار یک رویداد در سیستم

        :param event_type: نوع رویداد
        :param event_data: داده‌های رویداد
        :return: نتیجه انتشار
        """
        await self.initialize()

        # افزودن نوع رویداد به داده‌ها
        message_data = {
            "event_type": event_type,
            "timestamp": self._get_timestamp(),
            "data": event_data
        }

        # انتشار رویداد
        return await self.kafka_service.send_message(BALANCE_EVENTS_TOPIC, message_data)

    async def publish_metric(self, metric_name: str, metric_data: Dict[str, Any]) -> bool:
        """
        انتشار یک متریک در سیستم

        :param metric_name: نام متریک
        :param metric_data: داده‌های متریک
        :return: نتیجه انتشار
        """
        await self.initialize()

        # افزودن نام متریک به داده‌ها
        message_data = {
            "metric_name": metric_name,
            "timestamp": self._get_timestamp(),
            "data": metric_data
        }

        # انتشار متریک
        return await self.kafka_service.send_message(BALANCE_METRICS_TOPIC, message_data)

    async def run(self):
        """
        شروع سرویس پیام‌رسانی و اجرای حلقه اصلی آن
        """
        await self.initialize()

        try:
            logger.info("🚀 سرویس پیام‌رسانی در حال اجرا است")

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

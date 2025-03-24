"""
سرویس پیام‌رسانی برای ماژول Data
"""
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    DataRequest, DataResponse, kafka_service, TopicManager,
    DATA_REQUESTS_TOPIC, BALANCE_METRICS_TOPIC, MESSAGE_STATUS_SUCCESS
)
from ai.data.services.data_collector_service import data_collector_service

logger = logging.getLogger(__name__)


class MessagingService:
    """
    سرویس مدیریت یکپارچه پیام‌رسانی در ماژول Data

    این سرویس وظیفه راه‌اندازی، نظارت و مدیریت تمام ارتباطات پیام‌رسانی
    در ماژول Data را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس پیام‌رسانی
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.data_collector_service = data_collector_service
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

        # راه‌اندازی سرویس جمع‌آوری داده
        await self.data_collector_service.initialize()

        self._is_initialized = True
        logger.info("✅ سرویس پیام‌رسانی ماژول Data آماده به کار است")

    async def run(self):
        """
        شروع سرویس پیام‌رسانی و اجرای سرویس جمع‌آوری داده
        """
        await self.initialize()

        try:
            logger.info("🚀 سرویس پیام‌رسانی ماژول Data در حال اجرا است")

            # اجرای سرویس جمع‌آوری داده
            collector_task = asyncio.create_task(self.data_collector_service.run())

            # انتظار برای درخواست توقف
            await self._shutdown_event.wait()

            # لغو وظیفه جمع‌آوری داده
            collector_task.cancel()

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

        # توقف سرویس جمع‌آوری داده
        await self.data_collector_service.shutdown()

        # قطع اتصال از کافکا
        await self.kafka_service.disconnect()

        self._is_initialized = False
        self._shutdown_event.set()

        logger.info("✅ سرویس پیام‌رسانی با موفقیت متوقف شد")

    async def send_result_to_model(self, model_id: str, data: Dict[str, Any], request_id: str,
                                   response_topic: str) -> bool:
        """
        ارسال نتیجه جمع‌آوری داده به مدل

        :param model_id: شناسه مدل
        :param data: داده‌های جمع‌آوری‌شده
        :param request_id: شناسه درخواست اصلی
        :param response_topic: موضوع پاسخ
        :return: نتیجه ارسال
        """
        try:
            # ساخت پاسخ
            response = DataResponse()
            response.metadata.request_id = request_id
            response.metadata.source = "data"
            response.metadata.destination = model_id

            response.payload.status = MESSAGE_STATUS_SUCCESS
            response.payload.data = data

            # ارسال به موضوع مدل
            success = await self.kafka_service.send_data_response(response, response_topic)

            if success:
                logger.info(f"✅ نتیجه با موفقیت به مدل '{model_id}' ارسال شد")
            else:
                logger.error(f"❌ خطا در ارسال نتیجه به مدل '{model_id}'")

            return success

        except Exception as e:
            logger.exception(f"❌ خطا در ارسال نتیجه به مدل: {str(e)}")
            return False

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
            "source": "data",
            "timestamp": self._get_timestamp(),
            "data": metric_data
        }

        # انتشار متریک
        return await self.kafka_service.send_message(BALANCE_METRICS_TOPIC, message_data)

    async def process_test_request(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        پردازش درخواست تست مستقیم (برای آزمایش بدون کافکا)

        :param query_params: پارامترهای درخواست
        :return: نتیجه جمع‌آوری داده
        """
        try:
            await self.initialize()

            # ساخت درخواست آزمایشی
            model_id = query_params.get("model_id", "test_model")
            data_type_str = query_params.get("data_type", "text")
            source_type_str = query_params.get("source_type", "web")
            query = query_params.get("query", "")

            # تبدیل رشته‌ها به Enum
            try:
                data_type = DataType(data_type_str)
            except ValueError:
                data_type = DataType.TEXT

            try:
                data_source = DataSource(source_type_str) if source_type_str else None
            except ValueError:
                data_source = None

            # نیاز به تشخیص خودکار منبع
            if not data_source:
                data_source = self.data_collector_service._detect_data_source(query_params)

            # آماده‌سازی پارامترهای خاص
            collector_params = query_params.copy()
            collector_params["query"] = query

            # جمع‌آوری داده
            result = await self.data_collector_service.collect_data(
                DataRequest(
                    payload={
                        "model_id": model_id,
                        "data_type": data_type,
                        "data_source": data_source,
                        "parameters": collector_params,
                        "response_topic": ""
                    }
                ),
                ""
            )

            return {
                "status": "success" if result else "error",
                "model_id": model_id,
                "data_type": data_type_str,
                "source_type": data_source.value if data_source else "unknown",
                "query": query,
                "data": result
            }

        except Exception as e:
            logger.exception(f"❌ خطا در پردازش درخواست تست: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
messaging_service = MessagingService()

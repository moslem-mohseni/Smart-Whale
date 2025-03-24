"""
سرویس ارسال نتایج جمع‌آوری داده به مدل‌ها
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union

from ai.core.messaging import (
    DataType, DataSource, OperationType, Priority, RequestSource,
    DataResponse, create_data_response,
    kafka_service, TopicManager, MESSAGE_STATUS_SUCCESS, MESSAGE_STATUS_ERROR
)

logger = logging.getLogger(__name__)


class ResultService:
    """
    سرویس ارسال نتایج جمع‌آوری داده به مدل‌ها
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس نتایج
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self._is_initialized = False

        # نگهداری سابقه نتایج ارسال‌شده
        self.sent_results: Dict[str, Dict[str, Any]] = {}

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس نتایج
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        self._is_initialized = True
        logger.info("✅ سرویس نتایج آماده به کار است")

    async def send_result(
            self,
            model_id: str,
            request_id: str,
            data: Any,
            data_type: Union[DataType, str] = DataType.TEXT,
            data_source: Optional[Union[DataSource, str]] = None,
            metrics: Optional[Dict[str, Any]] = None,
            additional_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        ارسال نتیجه جمع‌آوری داده به مدل

        :param model_id: شناسه مدل
        :param request_id: شناسه درخواست
        :param data: داده‌های جمع‌آوری‌شده
        :param data_type: نوع داده
        :param data_source: منبع داده
        :param metrics: متریک‌های جمع‌آوری
        :param additional_info: اطلاعات تکمیلی
        :return: نتیجه ارسال
        """
        try:
            await self.initialize()

            # دریافت موضوع نتایج مدل
            model_topic = self.topic_manager.get_model_result_topic(model_id)

            # ساخت پاسخ
            response = create_data_response(
                request_id=request_id,
                model_id=model_id,
                status=MESSAGE_STATUS_SUCCESS,
                data=data,
                data_type=data_type,
                data_source=data_source,
                metrics=metrics or {}
            )

            # افزودن اطلاعات تکمیلی
            if additional_info:
                response.payload.additional_info = additional_info

            # ارسال به موضوع مدل
            success = await self.kafka_service.send_data_response(response, model_topic)

            if success:
                # ثبت نتیجه ارسال‌شده
                result_key = f"{model_id}:{request_id}"
                self.sent_results[result_key] = {
                    "model_id": model_id,
                    "request_id": request_id,
                    "timestamp": self._get_timestamp(),
                    "status": "sent",
                    "data_type": data_type.value if isinstance(data_type, DataType) else data_type,
                }

                logger.info(f"✅ نتیجه با موفقیت به مدل '{model_id}' ارسال شد (درخواست: {request_id})")
            else:
                logger.error(f"❌ خطا در ارسال نتیجه به مدل '{model_id}' (درخواست: {request_id})")

            return success

        except Exception as e:
            logger.exception(f"❌ خطا در ارسال نتیجه به مدل: {str(e)}")
            return False

    async def send_error(
            self,
            model_id: str,
            request_id: str,
            error_message: str,
            error_code: Optional[str] = None,
            data_type: Union[DataType, str] = DataType.TEXT,
            data_source: Optional[Union[DataSource, str]] = None
    ) -> bool:
        """
        ارسال پیام خطا به مدل

        :param model_id: شناسه مدل
        :param request_id: شناسه درخواست
        :param error_message: پیام خطا
        :param error_code: کد خطا (اختیاری)
        :param data_type: نوع داده درخواست‌شده
        :param data_source: منبع داده درخواست‌شده
        :return: نتیجه ارسال
        """
        try:
            await self.initialize()

            # دریافت موضوع نتایج مدل
            model_topic = self.topic_manager.get_model_result_topic(model_id)

            # ساخت پاسخ خطا
            response = create_data_response(
                request_id=request_id,
                model_id=model_id,
                status=MESSAGE_STATUS_ERROR,
                error_message=error_message,
                data_type=data_type,
                data_source=data_source
            )

            # افزودن کد خطا
            if error_code:
                response.payload.error_code = error_code

            # ارسال به موضوع مدل
            success = await self.kafka_service.send_data_response(response, model_topic)

            if success:
                # ثبت خطای ارسال‌شده
                error_key = f"{model_id}:{request_id}"
                self.sent_results[error_key] = {
                    "model_id": model_id,
                    "request_id": request_id,
                    "timestamp": self._get_timestamp(),
                    "status": "error",
                    "error_message": error_message
                }

                logger.info(f"✅ پیام خطا با موفقیت به مدل '{model_id}' ارسال شد (درخواست: {request_id})")
            else:
                logger.error(f"❌ خطا در ارسال پیام خطا به مدل '{model_id}' (درخواست: {request_id})")

            return success

        except Exception as e:
            logger.exception(f"❌ خطا در ارسال پیام خطا به مدل: {str(e)}")
            return False

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
result_service = ResultService()

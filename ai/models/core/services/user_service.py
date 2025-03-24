"""
سرویس مدیریت ارتباط با کاربر و استریم پاسخ‌ها

این سرویس وظیفه مدیریت ارتباط با کاربر، ثبت و مدیریت نشست‌ها و
استریم کردن پاسخ‌های نهایی مدل به کاربر را بر عهده دارد.
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Set, Union

from ai.core.messaging import (
    kafka_service, TopicManager, Priority, RequestSource,
    USER_REQUESTS_TOPIC, USER_RESPONSES_TOPIC
)

logger = logging.getLogger(__name__)


class UserService:
    """
    سرویس مدیریت ارتباط با کاربر و استریم پاسخ‌ها

    این سرویس وظیفه مدیریت نشست‌های کاربر، دریافت درخواست‌های کاربر،
    و استریم کردن پاسخ‌های مدل به کاربر را بر عهده دارد.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس کاربر
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.user_requests_topic = USER_REQUESTS_TOPIC
        self.user_responses_topic = USER_RESPONSES_TOPIC
        self.active_sessions: Dict[str, Dict[str, Any]] = {}  # نگهداری نشست‌های فعال
        self.request_handlers: Dict[str, Callable] = {}  # پردازشگرهای درخواست کاربر بر اساس مدل
        self.session_models: Dict[str, str] = {}  # مدل مربوط به هر نشست
        self._is_initialized = False

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس و اطمینان از وجود موضوعات مورد نیاز
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوعات اصلی کاربر
        await self.topic_manager.ensure_topic_exists(self.user_requests_topic)
        await self.topic_manager.ensure_topic_exists(self.user_responses_topic)

        self._is_initialized = True
        logger.info(
            f"✅ سرویس کاربر آماده به کار است (موضوعات: {self.user_requests_topic}, {self.user_responses_topic})")

    async def register_user_session(
            self,
            session_id: str,
            model_id: str,
            user_id: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        ثبت یک نشست جدید کاربر

        :param session_id: شناسه یکتای نشست
        :param model_id: شناسه مدل مورد استفاده برای این نشست
        :param user_id: شناسه کاربر (اختیاری)
        :param metadata: اطلاعات اضافی نشست (اختیاری)
        :return: اطلاعات نشست ثبت شده
        """
        await self.initialize()

        # اطمینان از وجود موضوع پاسخ اختصاصی برای نشست
        session_topic = await self.topic_manager.ensure_session_topic(session_id)

        # ثبت اطلاعات نشست
        self.active_sessions[session_id] = {
            "session_id": session_id,
            "model_id": model_id,
            "user_id": user_id,
            "creation_time": self._get_timestamp(),
            "last_activity": self._get_timestamp(),
            "metadata": metadata or {},
            "response_topic": session_topic,
            "message_count": 0,
            "status": "active"
        }

        # ثبت مدل مربوط به نشست
        self.session_models[session_id] = model_id

        logger.info(f"✅ نشست کاربر '{session_id}' با مدل '{model_id}' ثبت شد")

        return {
            "session_id": session_id,
            "model_id": model_id,
            "status": "active",
            "response_topic": session_topic,
            "creation_time": self.active_sessions[session_id]["creation_time"]
        }

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
        استریم یک بخش از پاسخ به کاربر

        :param session_id: شناسه نشست کاربر
        :param response_chunk: بخشی از پاسخ برای استریم
        :param is_final: آیا این آخرین بخش پاسخ است
        :param chunk_id: شناسه بخش (برای ترتیب‌بندی، اختیاری)
        :param thinking: فرآیند تفکر مدل (استدلال، اختیاری)
        :param metadata: اطلاعات اضافی پاسخ (اختیاری)
        :return: نتیجه عملیات استریم
        """
        await self.initialize()

        # بررسی وجود نشست
        if session_id not in self.active_sessions:
            error_msg = f"❌ نشست '{session_id}' یافت نشد"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # به‌روزرسانی زمان آخرین فعالیت
        self.active_sessions[session_id]["last_activity"] = self._get_timestamp()

        # افزایش شمارنده پیام
        if chunk_id is None:
            self.active_sessions[session_id]["message_count"] += 1
            chunk_id = self.active_sessions[session_id]["message_count"]

        # آماده‌سازی پیام پاسخ
        message = {
            "session_id": session_id,
            "model_id": self.active_sessions[session_id]["model_id"],
            "timestamp": self._get_timestamp(),
            "response_chunk": response_chunk,
            "chunk_id": chunk_id,
            "is_final": is_final,
            "thinking": thinking,
            "metadata": metadata or {}
        }

        # دریافت موضوع پاسخ اختصاصی نشست
        topic = self.active_sessions[session_id]["response_topic"]

        # ارسال پیام
        success = await self.kafka_service.send_message(topic, message)

        # ارسال به موضوع عمومی پاسخ‌های کاربر نیز
        if success:
            await self.kafka_service.send_message(self.user_responses_topic, message)

        if success:
            logger.info(f"✅ بخش پاسخ {chunk_id} برای نشست '{session_id}' استریم شد")

            # در صورت پایان پاسخ، ثبت در لاگ
            if is_final:
                logger.info(f"🏁 پاسخ نهایی برای نشست '{session_id}' استریم شد")
        else:
            logger.error(f"❌ خطا در استریم بخش پاسخ {chunk_id} برای نشست '{session_id}'")

        return {
            "status": "sent" if success else "error",
            "session_id": session_id,
            "chunk_id": chunk_id,
            "is_final": is_final,
            "timestamp": message["timestamp"]
        }

    async def send_thinking_process(
            self,
            session_id: str,
            thinking_data: str,
            is_final: bool = False,
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        ارسال فرآیند تفکر مدل به کاربر (اختیاری)

        :param session_id: شناسه نشست کاربر
        :param thinking_data: داده‌های فرآیند تفکر (استدلال مدل)
        :param is_final: آیا این آخرین بخش فرآیند تفکر است
        :param metadata: اطلاعات اضافی (اختیاری)
        :return: نتیجه عملیات ارسال
        """
        await self.initialize()

        # بررسی وجود نشست
        if session_id not in self.active_sessions:
            error_msg = f"❌ نشست '{session_id}' یافت نشد"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # به‌روزرسانی زمان آخرین فعالیت
        self.active_sessions[session_id]["last_activity"] = self._get_timestamp()

        # آماده‌سازی پیام تفکر
        message = {
            "session_id": session_id,
            "model_id": self.active_sessions[session_id]["model_id"],
            "timestamp": self._get_timestamp(),
            "thinking": thinking_data,
            "is_final": is_final,
            "response_type": "thinking",
            "metadata": metadata or {}
        }

        # دریافت موضوع پاسخ اختصاصی نشست
        topic = self.active_sessions[session_id]["response_topic"]

        # ارسال پیام
        success = await self.kafka_service.send_message(topic, message)

        if success:
            logger.info(f"✅ فرآیند تفکر برای نشست '{session_id}' ارسال شد")
        else:
            logger.error(f"❌ خطا در ارسال فرآیند تفکر برای نشست '{session_id}'")

        return {
            "status": "sent" if success else "error",
            "session_id": session_id,
            "is_final": is_final,
            "timestamp": message["timestamp"]
        }

    async def receive_user_request(
            self,
            session_id: str,
            request_text: str,
            request_type: str = "text",
            metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        دریافت و پردازش درخواست کاربر

        :param session_id: شناسه نشست کاربر
        :param request_text: متن درخواست کاربر
        :param request_type: نوع درخواست (text, voice, image)
        :param metadata: اطلاعات اضافی درخواست (اختیاری)
        :return: نتیجه پردازش درخواست
        """
        await self.initialize()

        # بررسی وجود نشست
        if session_id not in self.active_sessions:
            error_msg = f"❌ نشست '{session_id}' یافت نشد"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # به‌روزرسانی زمان آخرین فعالیت
        self.active_sessions[session_id]["last_activity"] = self._get_timestamp()

        # دریافت مدل مربوط به نشست
        model_id = self.session_models.get(session_id)
        if not model_id:
            error_msg = f"❌ مدل مربوط به نشست '{session_id}' یافت نشد"
            logger.error(error_msg)
            return {"status": "error", "error": error_msg}

        # آماده‌سازی پیام درخواست
        message = {
            "session_id": session_id,
            "model_id": model_id,
            "timestamp": self._get_timestamp(),
            "request_text": request_text,
            "request_type": request_type,
            "metadata": metadata or {}
        }

        # ارسال به موضوع درخواست‌های کاربر
        success = await self.kafka_service.send_message(self.user_requests_topic, message)

        if success:
            logger.info(f"✅ درخواست کاربر برای نشست '{session_id}' و مدل '{model_id}' ثبت شد")
        else:
            logger.error(f"❌ خطا در ثبت درخواست کاربر برای نشست '{session_id}'")

        return {
            "status": "received" if success else "error",
            "session_id": session_id,
            "model_id": model_id,
            "timestamp": message["timestamp"]
        }

    async def subscribe_to_user_requests(
            self,
            model_id: str,
            handler: Callable
    ) -> Dict[str, Any]:
        """
        اشتراک یک مدل در موضوع درخواست‌های کاربر

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده درخواست‌های کاربر
        :return: نتیجه عملیات اشتراک
        """
        await self.initialize()

        # ثبت پردازشگر درخواست برای مدل
        self.request_handlers[model_id] = handler

        # تعریف پردازشگر پیام‌های دریافتی
        async def request_handler(message_data: Dict[str, Any]):
            # بررسی اینکه پیام مربوط به این مدل باشد
            if message_data.get("model_id") == model_id:
                # فراخوانی پردازشگر اختصاصی مدل
                await handler(message_data)

        # اشتراک در موضوع درخواست‌های کاربر
        group_id = f"model-{model_id}-user-requests"
        await self.kafka_service.subscribe(self.user_requests_topic, group_id, request_handler)

        logger.info(f"✅ مدل '{model_id}' با موفقیت در موضوع درخواست‌های کاربر مشترک شد")

        return {
            "model_id": model_id,
            "status": "subscribed",
            "topic": self.user_requests_topic
        }

    async def end_user_session(
            self,
            session_id: str,
            reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        پایان دادن به یک نشست کاربر

        :param session_id: شناسه نشست
        :param reason: دلیل پایان نشست (اختیاری)
        :return: نتیجه عملیات پایان نشست
        """
        # بررسی وجود نشست
        if session_id not in self.active_sessions:
            logger.warning(f"⚠ نشست '{session_id}' قبلاً یافت نشد یا پایان یافته است")
            return {
                "session_id": session_id,
                "status": "not_found"
            }

        # به‌روزرسانی وضعیت نشست
        self.active_sessions[session_id]["status"] = "ended"
        self.active_sessions[session_id]["end_time"] = self._get_timestamp()
        self.active_sessions[session_id]["end_reason"] = reason

        # ارسال پیام پایان نشست (اختیاری)
        try:
            # آماده‌سازی پیام پایان
            message = {
                "session_id": session_id,
                "model_id": self.active_sessions[session_id]["model_id"],
                "timestamp": self.active_sessions[session_id]["end_time"],
                "status": "session_ended",
                "reason": reason,
                "is_final": True
            }

            # دریافت موضوع پاسخ اختصاصی نشست
            topic = self.active_sessions[session_id]["response_topic"]

            # ارسال پیام
            await self.kafka_service.send_message(topic, message)
            await self.kafka_service.send_message(self.user_responses_topic, message)
        except Exception as e:
            logger.error(f"❌ خطا در ارسال پیام پایان نشست '{session_id}': {str(e)}")

        # نگهداری اطلاعات نشست برای مدت مشخص و سپس حذف آن (با تأخیر)
        asyncio.create_task(self._cleanup_session(session_id))

        logger.info(f"🏁 نشست '{session_id}' با موفقیت پایان یافت")

        return {
            "session_id": session_id,
            "status": "ended",
            "end_time": self.active_sessions[session_id]["end_time"],
            "reason": reason
        }

    async def get_session_info(
            self,
            session_id: str
    ) -> Dict[str, Any]:
        """
        دریافت اطلاعات نشست کاربر

        :param session_id: شناسه نشست
        :return: اطلاعات نشست
        """
        if session_id not in self.active_sessions:
            return {
                "session_id": session_id,
                "status": "not_found",
                "error": "نشست مورد نظر یافت نشد"
            }

        return self.active_sessions[session_id]

    async def _cleanup_session(self, session_id: str, delay: int = 3600):
        """
        تابع داخلی برای پاکسازی منابع نشست پس از گذشت زمان مشخص

        :param session_id: شناسه نشست
        :param delay: تأخیر پاکسازی به ثانیه (پیش‌فرض: 1 ساعت)
        """
        await asyncio.sleep(delay)

        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id].copy()
            del self.active_sessions[session_id]

            if session_id in self.session_models:
                del self.session_models[session_id]

            logger.info(f"🧹 منابع نشست '{session_id}' پاکسازی شد")

            return session_info

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
user_service = UserService()

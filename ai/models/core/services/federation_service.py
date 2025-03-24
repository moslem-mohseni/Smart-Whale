"""
سرویس مدیریت ارتباط فدراسیونی بین مدل‌ها

این سرویس مسئول مدیریت ارتباطات بین مدل‌ها، اشتراک‌گذاری دانش
و هماهنگی همکاری بین آن‌ها می‌باشد.
"""
import json
import logging
import asyncio
from typing import Dict, Any, List, Callable, Optional, Set, Union

from ai.core.messaging import (
    kafka_service, TopicManager, OperationType, Priority, RequestSource,
    MODELS_FEDERATION_TOPIC
)

logger = logging.getLogger(__name__)


class FederationService:
    """
    سرویس مدیریت ارتباط فدراسیونی بین مدل‌ها

    این سرویس امکان اشتراک‌گذاری دانش و همکاری بین مدل‌های مختلف
    را فراهم می‌آورد و از حفظ حریم خصوصی داده‌ها اطمینان حاصل می‌کند.
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس فدراسیون
        """
        self.kafka_service = kafka_service
        self.topic_manager = TopicManager(kafka_service)
        self.federation_topic = MODELS_FEDERATION_TOPIC
        self.registered_models: Set[str] = set()
        self.federation_handlers: Dict[str, Callable] = {}
        self.collaboration_requests: Dict[str, Dict[str, Any]] = {}  # نگهداری درخواست‌های همکاری فعال
        self._is_initialized = False

    async def initialize(self):
        """
        آماده‌سازی اولیه سرویس و اطمینان از وجود موضوعات مورد نیاز
        """
        if self._is_initialized:
            return

        # اتصال به کافکا
        await self.kafka_service.connect()

        # اطمینان از وجود موضوع فدراسیونی اصلی
        await self.topic_manager.ensure_topic_exists(self.federation_topic)

        self._is_initialized = True
        logger.info(f"✅ سرویس فدراسیون آماده به کار است (موضوع: {self.federation_topic})")

    async def register_model(self, model_id: str, handler: Optional[Callable] = None) -> Dict[str, Any]:
        """
        ثبت یک مدل در سیستم فدراسیون

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده پیام‌های فدراسیونی (اختیاری)
        :return: اطلاعات ثبت مدل
        """
        await self.initialize()

        # اطمینان از وجود موضوع فدراسیونی برای مدل
        federation_topic = await self.topic_manager.ensure_model_federation_topic(model_id)

        # ثبت مدل در لیست مدل‌های ثبت‌شده
        self.registered_models.add(model_id)

        # ثبت پردازشگر فدراسیونی اختصاصی
        if handler:
            self.federation_handlers[model_id] = handler

        logger.info(f"✅ مدل '{model_id}' با موفقیت در سیستم فدراسیون ثبت شد (موضوع: {federation_topic})")

        return {
            "model_id": model_id,
            "status": "registered",
            "federation_topic": federation_topic
        }

    async def unregister_model(self, model_id: str) -> Dict[str, Any]:
        """
        حذف ثبت یک مدل از سیستم فدراسیون

        :param model_id: شناسه مدل
        :return: نتیجه عملیات حذف ثبت
        """
        if model_id not in self.registered_models:
            logger.warning(f"⚠ مدل '{model_id}' قبلاً در سیستم فدراسیون ثبت نشده است")
            return {
                "model_id": model_id,
                "status": "not_registered"
            }

        # حذف از مدل‌های ثبت‌شده
        self.registered_models.remove(model_id)

        # حذف پردازشگر فدراسیونی
        if model_id in self.federation_handlers:
            del self.federation_handlers[model_id]

        # حذف درخواست‌های همکاری مربوطه
        to_remove = []
        for req_id, req_data in self.collaboration_requests.items():
            if req_data.get("source_model") == model_id or req_data.get("target_model") == model_id:
                to_remove.append(req_id)

        for req_id in to_remove:
            del self.collaboration_requests[req_id]

        logger.info(f"✅ ثبت مدل '{model_id}' با موفقیت از سیستم فدراسیون حذف شد")

        return {
            "model_id": model_id,
            "status": "unregistered"
        }

    async def share_knowledge(
            self,
            source_model_id: str,
            target_model_id: Optional[str] = None,
            knowledge_data: Dict[str, Any] = None,
            knowledge_type: str = "general",
            privacy_level: str = "standard"
    ) -> Dict[str, Any]:
        """
        اشتراک‌گذاری دانش از یک مدل با یک یا چند مدل دیگر

        :param source_model_id: شناسه مدل منبع
        :param target_model_id: شناسه مدل هدف (None برای همه مدل‌ها)
        :param knowledge_data: داده‌های دانش برای اشتراک‌گذاری
        :param knowledge_type: نوع دانش (general, domain_specific, etc.)
        :param privacy_level: سطح حریم خصوصی (standard, anonymized, encrypted)
        :return: نتیجه عملیات اشتراک‌گذاری
        """
        await self.initialize()

        # بررسی ثبت مدل منبع
        if source_model_id not in self.registered_models:
            await self.register_model(source_model_id)

        # آماده‌سازی پیام اشتراک‌گذاری دانش
        message = {
            "operation": "SHARE_KNOWLEDGE",
            "timestamp": self._get_timestamp(),
            "source_model": source_model_id,
            "target_model": target_model_id,
            "knowledge_type": knowledge_type,
            "privacy_level": privacy_level,
            "data": knowledge_data or {}
        }

        # انتخاب موضوع مناسب
        if target_model_id:
            # اشتراک‌گذاری با یک مدل خاص
            if target_model_id not in self.registered_models:
                await self.register_model(target_model_id)

            topic = self.topic_manager.get_model_federation_topic(target_model_id)
        else:
            # اشتراک‌گذاری با همه مدل‌ها
            topic = self.federation_topic

        # ارسال پیام
        success = await self.kafka_service.send_message(topic, message)

        if success:
            logger.info(f"✅ دانش از مدل '{source_model_id}' با موفقیت به اشتراک گذاشته شد")

            # اگر پیام به موضوع خاص مدل ارسال شده، دریافت را در لاگ ثبت می‌کنیم
            if target_model_id:
                logger.info(f"📤 دانش به مدل '{target_model_id}' ارسال شد")
        else:
            logger.error(f"❌ خطا در اشتراک‌گذاری دانش از مدل '{source_model_id}'")

        return {
            "status": "success" if success else "error",
            "source_model": source_model_id,
            "target_model": target_model_id,
            "knowledge_type": knowledge_type,
            "timestamp": message["timestamp"]
        }

    async def subscribe_to_federation(
            self,
            model_id: str,
            handler: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        اشتراک یک مدل در موضوع فدراسیون برای دریافت پیام‌های اشتراک‌گذاری دانش

        :param model_id: شناسه مدل
        :param handler: تابع پردازش‌کننده پیام‌های فدراسیونی (اختیاری)
        :return: نتیجه عملیات اشتراک
        """
        await self.initialize()

        # ثبت مدل در صورت نیاز
        if model_id not in self.registered_models:
            await self.register_model(model_id, handler)
        elif handler:
            # به‌روزرسانی پردازشگر
            self.federation_handlers[model_id] = handler

        # دریافت موضوع فدراسیونی مدل
        model_topic = self.topic_manager.get_model_federation_topic(model_id)

        # تعریف پردازشگر پیام‌های دریافتی
        async def federation_message_handler(message_data: Dict[str, Any]):
            # اگر پردازشگر اختصاصی وجود دارد، از آن استفاده می‌شود
            if model_id in self.federation_handlers and self.federation_handlers[model_id]:
                await self.federation_handlers[model_id](message_data)
            else:
                # پردازش پیش‌فرض
                operation = message_data.get("operation")
                source = message_data.get("source_model")

                if operation == "SHARE_KNOWLEDGE":
                    logger.info(f"📥 دانش از مدل '{source}' برای مدل '{model_id}' دریافت شد")
                elif operation == "REQUEST_COLLABORATION":
                    logger.info(f"📥 درخواست همکاری از مدل '{source}' برای مدل '{model_id}' دریافت شد")
                    # در اینجا می‌توان منطق پردازش درخواست همکاری را اضافه کرد
                elif operation == "COLLABORATION_RESPONSE":
                    logger.info(f"📥 پاسخ همکاری از مدل '{source}' برای مدل '{model_id}' دریافت شد")
                else:
                    logger.info(f"📥 پیام فدراسیونی نوع '{operation}' از مدل '{source}' دریافت شد")

        # اشتراک در موضوع فدراسیونی مدل
        group_id = f"federation-{model_id}"
        await self.kafka_service.subscribe(model_topic, group_id, federation_message_handler)

        # اشتراک در موضوع عمومی فدراسیون
        general_group_id = f"federation-general-{model_id}"
        await self.kafka_service.subscribe(self.federation_topic, general_group_id, federation_message_handler)

        logger.info(f"✅ اشتراک مدل '{model_id}' در سیستم فدراسیون انجام شد")

        return {
            "model_id": model_id,
            "status": "subscribed",
            "model_topic": model_topic,
            "general_topic": self.federation_topic
        }

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
        ارسال درخواست همکاری از یک مدل به یک یا چند مدل دیگر

        :param source_model_id: شناسه مدل درخواست‌کننده
        :param problem_data: داده‌های مسئله نیازمند همکاری
        :param target_models: لیست شناسه مدل‌های هدف (None برای همه مدل‌ها)
        :param collaboration_type: نوع همکاری درخواستی
        :param priority: اولویت درخواست
        :param timeout: مدت زمان انتظار برای پاسخ (ثانیه)
        :return: نتیجه عملیات درخواست همکاری
        """
        await self.initialize()

        # بررسی ثبت مدل منبع
        if source_model_id not in self.registered_models:
            await self.register_model(source_model_id)

        # ساخت شناسه یکتا برای درخواست همکاری
        request_id = f"collab_{source_model_id}_{self._get_timestamp()}_{len(self.collaboration_requests) + 1}"

        # آماده‌سازی پیام درخواست همکاری
        message = {
            "operation": "REQUEST_COLLABORATION",
            "request_id": request_id,
            "timestamp": self._get_timestamp(),
            "source_model": source_model_id,
            "collaboration_type": collaboration_type,
            "priority": priority.value if isinstance(priority, Priority) else priority,
            "problem_data": problem_data,
            "timeout": timeout
        }

        # ثبت درخواست در لیست درخواست‌های فعال
        self.collaboration_requests[request_id] = {
            "source_model": source_model_id,
            "target_models": target_models,
            "timestamp": message["timestamp"],
            "responses": {},
            "status": "pending"
        }

        success_count = 0
        error_count = 0

        # ارسال به مدل‌های هدف
        if target_models:
            for target_model_id in target_models:
                if target_model_id not in self.registered_models:
                    await self.register_model(target_model_id)

                topic = self.topic_manager.get_model_federation_topic(target_model_id)
                target_message = {**message, "target_model": target_model_id}

                success = await self.kafka_service.send_message(topic, target_message)
                if success:
                    success_count += 1
                    logger.info(f"📤 درخواست همکاری از مدل '{source_model_id}' به مدل '{target_model_id}' ارسال شد")
                else:
                    error_count += 1
                    logger.error(f"❌ خطا در ارسال درخواست همکاری به مدل '{target_model_id}'")
        else:
            # ارسال به موضوع عمومی فدراسیون برای دریافت توسط همه مدل‌ها
            success = await self.kafka_service.send_message(self.federation_topic, message)
            if success:
                success_count = 1
                logger.info(f"📤 درخواست همکاری از مدل '{source_model_id}' به تمام مدل‌ها ارسال شد")
            else:
                error_count = 1
                logger.error("❌ خطا در ارسال درخواست همکاری به موضوع عمومی فدراسیون")

        # بروزرسانی وضعیت درخواست
        self.collaboration_requests[request_id]["status"] = "sent"

        # تنظیم تایمر برای خاتمه خودکار درخواست بعد از مهلت زمانی
        asyncio.create_task(self._expire_collaboration_request(request_id, timeout))

        return {
            "request_id": request_id,
            "status": "sent" if success_count > 0 else "error",
            "source_model": source_model_id,
            "target_models": target_models,
            "success_count": success_count,
            "error_count": error_count,
            "timestamp": message["timestamp"]
        }

    async def respond_to_collaboration(
            self,
            source_model_id: str,
            target_model_id: str,
            request_id: str,
            response_data: Dict[str, Any],
            status: str = "success"
    ) -> Dict[str, Any]:
        """
        ارسال پاسخ به یک درخواست همکاری

        :param source_model_id: شناسه مدل پاسخ‌دهنده
        :param target_model_id: شناسه مدل درخواست‌کننده
        :param request_id: شناسه درخواست همکاری
        :param response_data: داده‌های پاسخ
        :param status: وضعیت پاسخ (success, partial, error)
        :return: نتیجه عملیات پاسخ به همکاری
        """
        await self.initialize()

        # بررسی ثبت مدل‌ها
        if source_model_id not in self.registered_models:
            await self.register_model(source_model_id)

        if target_model_id not in self.registered_models:
            await self.register_model(target_model_id)

        # آماده‌سازی پیام پاسخ همکاری
        message = {
            "operation": "COLLABORATION_RESPONSE",
            "request_id": request_id,
            "timestamp": self._get_timestamp(),
            "source_model": source_model_id,
            "target_model": target_model_id,
            "status": status,
            "data": response_data
        }

        # ارسال پاسخ به مدل درخواست‌کننده
        topic = self.topic_manager.get_model_federation_topic(target_model_id)
        success = await self.kafka_service.send_message(topic, message)

        if success:
            logger.info(f"✅ پاسخ همکاری از مدل '{source_model_id}' به مدل '{target_model_id}' ارسال شد")

            # به‌روزرسانی وضعیت درخواست در صورت وجود
            if request_id in self.collaboration_requests:
                self.collaboration_requests[request_id]["responses"][source_model_id] = {
                    "timestamp": message["timestamp"],
                    "status": status
                }
        else:
            logger.error(f"❌ خطا در ارسال پاسخ همکاری از مدل '{source_model_id}' به مدل '{target_model_id}'")

        return {
            "request_id": request_id,
            "status": "sent" if success else "error",
            "source_model": source_model_id,
            "target_model": target_model_id,
            "timestamp": message["timestamp"]
        }

    async def get_collaboration_status(
            self,
            request_id: str
    ) -> Dict[str, Any]:
        """
        دریافت وضعیت یک درخواست همکاری

        :param request_id: شناسه درخواست همکاری
        :return: اطلاعات وضعیت درخواست
        """
        if request_id not in self.collaboration_requests:
            return {
                "request_id": request_id,
                "status": "not_found",
                "error": "درخواست همکاری با این شناسه یافت نشد"
            }

        return {
            "request_id": request_id,
            "status": self.collaboration_requests[request_id]["status"],
            "source_model": self.collaboration_requests[request_id]["source_model"],
            "target_models": self.collaboration_requests[request_id]["target_models"],
            "timestamp": self.collaboration_requests[request_id]["timestamp"],
            "responses": self.collaboration_requests[request_id]["responses"],
            "response_count": len(self.collaboration_requests[request_id]["responses"])
        }

    async def _expire_collaboration_request(self, request_id: str, timeout: int):
        """
        تابع داخلی برای پایان دادن به درخواست همکاری پس از گذشت مهلت زمانی

        :param request_id: شناسه درخواست همکاری
        :param timeout: مدت زمان انتظار (ثانیه)
        """
        await asyncio.sleep(timeout)

        if request_id in self.collaboration_requests:
            if self.collaboration_requests[request_id]["status"] == "sent":
                self.collaboration_requests[request_id]["status"] = "expired"
                logger.info(f"⏱ درخواست همکاری '{request_id}' منقضی شد")

    def _get_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()


# نمونه Singleton برای استفاده در سراسر سیستم
federation_service = FederationService()

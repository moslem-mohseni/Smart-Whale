"""
BalanceConnector Module
-------------------------
این فایل مسئول هماهنگی با ماژول Balance جهت تخصیص منابع و ارسال درخواست‌های مرتبط با جمع‌آوری داده‌ها در سیستم خودآموزی است.
این کلاس پیام‌های استاندارد با ساختار metadata و payload را می‌سازد و از طریق Kafka Producer به ماژول Balance ارسال می‌کند.
پیام‌ها طبق استانداردهای تعریف‌شده برای ارتباط با Balance ساخته می‌شوند.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import uuid
from datetime import datetime
import logging
from typing import Any, Dict, Optional


class BalanceConnector:
    """
    BalanceConnector مسئول ارسال درخواست‌های منابع به ماژول Balance از طریق Kafka است.

    ویژگی‌ها:
      - استفاده از Kafka Producer و Topic Manager جهت ارسال پیام به موضوع BALANCE_REQUESTS_TOPIC.
      - ساخت پیام استاندارد با بخش‌های metadata و payload.
      - مدیریت خطا و ثبت لاگ‌های دقیق برای اطمینان از عملکرد صحیح.
    """

    def __init__(self, producer: Any, topic_manager: Any, response_topic: Optional[str] = None,
                 model_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه BalanceConnector.

        Args:
            producer (Any): شیء Kafka Producer (ناهمزمان) جهت ارسال پیام.
            topic_manager (Any): شیء مدیریت موضوعات (Topic Manager) جهت دریافت نام موضوع.
            response_topic (Optional[str]): موضوع پاسخ جهت دریافت نتایج از ماژول Balance.
            model_id (Optional[str]): شناسه مدل مرتبط (در صورت نیاز).
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل تنظیمات مانند سطح اولویت پیش‌فرض.
        """
        self.logger = logging.getLogger("BalanceConnector")
        self.producer = producer
        self.topic_manager = topic_manager
        self.response_topic = response_topic or "BALANCE_RESPONSE_TOPIC"
        self.model_id = model_id
        self.config = config or {}
        self.default_priority = int(self.config.get("default_priority", 1))
        self.logger.info(
            f"[BalanceConnector] Initialized with model_id={self.model_id} and response_topic={self.response_topic}")

    def _build_metadata(self, destination: str, priority: Optional[int] = None, request_source: str = "system") -> Dict[
        str, Any]:
        """
        ساخت بخش metadata برای پیام.

        Args:
            destination (str): مقصد پیام، برای ما همیشه "balance".
            priority (Optional[int]): سطح اولویت درخواست؛ در صورت عدم ارائه از مقدار پیش‌فرض استفاده می‌شود.
            request_source (str): منبع درخواست (معمولاً "system").

        Returns:
            Dict[str, Any]: بخش metadata پیام.
        """
        return {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source": "self_learning_acquisition",
            "destination": destination,
            "priority": priority if priority is not None else self.default_priority,
            "request_source": request_source
        }

    def _build_message(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        ساخت پیام استاندارد جهت درخواست منابع.

        Args:
            parameters (Dict[str, Any]): پارامترهای مربوط به درخواست منابع.

        Returns:
            Dict[str, Any]: پیام نهایی با ساختار استاندارد.
        """
        metadata = self._build_metadata(destination="balance")
        payload = {
            "operation": "REQUEST_RESOURCES",
            "parameters": parameters,
            "response_topic": self.response_topic
        }
        if self.model_id:
            payload["model_id"] = self.model_id
        return {
            "metadata": metadata,
            "payload": payload
        }

    async def request_resources(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست منابع به ماژول Balance از طریق Kafka.

        Args:
            parameters (Dict[str, Any]): پارامترهای مورد نیاز برای درخواست منابع.

        Returns:
            Dict[str, Any]: پاسخ ارسال درخواست شامل موفقیت یا خطا.
        """
        message = self._build_message(parameters)
        topic = self.topic_manager.get_topic("BALANCE_REQUESTS_TOPIC")
        self.logger.info(
            f"[BalanceConnector] Sending resource request to topic {topic} with request_id: {message['metadata']['request_id']}")
        try:
            # ارسال پیام به موضوع از طریق producer
            future = await self.producer.send(topic, message)
            await self.producer.flush()
            self.logger.info(f"[BalanceConnector] Resource request sent successfully.")
            return {"success": True, "request_id": message["metadata"]["request_id"], "topic": topic}
        except Exception as e:
            self.logger.error(f"[BalanceConnector] Error sending resource request: {str(e)}")
            return {"success": False, "error": str(e)}

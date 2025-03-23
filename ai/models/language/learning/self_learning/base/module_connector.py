"""
ModuleConnector Module
------------------------
این فایل مسئول یکپارچه‌سازی و برقراری ارتباط بین سیستم Self-Learning و ماژول‌های Balance، Data و Federation از طریق Kafka است.
پیام‌های ارسالی دارای ساختار استاندارد زیر می‌باشند:
{
  "metadata": {
      "request_id": <uuid>,
      "timestamp": <ISO datetime>,
      "source": "client",
      "destination": <target_module>,   # "balance", "data" یا "federation"
      "priority": <number>,             # مثال: 1 برای بالا، 2 برای متوسط
      "request_source": <"user" | "model" | "system">
  },
  "payload": {
      "operation": <OPERATION_NAME>,    # مانند "REGISTER_MODEL", "FETCH_DATA", "SHARE_KNOWLEDGE", "NOTIFY_EVENT"
      ...                               # سایر اطلاعات مرتبط به عملیات
  }
}
ارتباط از طریق Kafka انجام می‌شود.
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


# فرض می‌کنیم که producer و topic_manager توسط سایر بخش‌های سیستم فراهم شده‌اند.
# producer باید یک شیء KafkaProducer غیرهمزمان (مثلاً از aiokafka) باشد.
# topic_manager وظیفه مدیریت و ارائه نام موضوعات (topics) را بر عهده دارد.

class ModuleConnector:
    """
    کلاس ModuleConnector برای ایجاد یک رابط یکپارچه جهت ارتباط با ماژول‌های Balance، Data و Federation.
    این کلاس از Kafka برای ارسال پیام‌ها استفاده می‌کند و پیام‌ها را طبق ساختار استاندارد مورد انتظار می‌سازد.
    """

    def __init__(self, producer: Any, topic_manager: Any, response_topic: Optional[str] = None,
                 model_id: Optional[str] = None):
        self.logger = logging.getLogger("ModuleConnector")
        self.producer = producer  # Kafka producer (async)
        self.topic_manager = topic_manager  # شیء TopicManager جهت دریافت نام موضوعات
        self.response_topic = response_topic or "SELF_LEARNING_RESPONSE"
        self.model_id = model_id
        self.logger.info("[ModuleConnector] Initialized with Kafka producer and topic manager.")

    def _build_metadata(self, destination: str, priority: int, request_source: str = "user") -> Dict[str, Any]:
        return {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "source": "client",
            "destination": destination,
            "priority": priority,
            "request_source": request_source
        }

    def _build_message(self, destination: str, operation: str, payload: Dict[str, Any], priority: int = 1,
                       request_source: str = "user") -> Dict[str, Any]:
        return {
            "metadata": self._build_metadata(destination, priority, request_source),
            "payload": {
                "operation": operation,
                **payload
            }
        }

    async def request_balance_resources(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست دریافت منابع از ماژول Balance به صورت استاندارد.

        Args:
            request_data (Dict[str, Any]): اطلاعات مورد نیاز برای درخواست منابع.

        Returns:
            Dict[str, Any]: پاسخ دریافتی از ماژول Balance.
        """
        destination = "balance"
        operation = "REQUEST_RESOURCES"
        payload = {
            "parameters": request_data,
            "response_topic": self.response_topic
        }
        message = self._build_message(destination, operation, payload, priority=1, request_source="system")

        topic = self.topic_manager.get_topic("BALANCE_REQUESTS_TOPIC")
        self.logger.info(f"[ModuleConnector] Sending resource request to Balance on topic {topic}")
        try:
            future = await self.producer.send(topic, message)
            await self.producer.flush()
            self.logger.info(
                f"[ModuleConnector] Resource request sent with request_id: {message['metadata']['request_id']}")
            return {"success": True, "request_id": message["metadata"]["request_id"], "topic": topic}
        except Exception as e:
            self.logger.error(f"[ModuleConnector] Error in request_balance_resources: {str(e)}")
            return {"success": False, "error": str(e)}

    async def fetch_data(self, query: str, data_source: str = "WIKI", data_type: str = "TEXT",
                         params: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        ارسال درخواست جمع‌آوری داده به ماژول Data از طریق Balance.

        Args:
            query (str): عبارت جستجو (مثلاً عنوان یا URL).
            data_source (str): منبع داده، به عنوان مثال "WIKI", "WEB".
            data_type (str): نوع داده مانند "TEXT", "IMAGE".
            params (Optional[Dict[str, Any]]): پارامترهای اختیاری برای درخواست.

        Returns:
            Optional[str]: شناسه درخواست در صورت موفقیت، در غیر این صورت None.
        """
        destination = "balance"
        operation = "FETCH_DATA"
        # تنظیم پیش‌فرض برای ویکی
        if data_source.upper() == "WIKI" and params is None:
            params = {
                "title": query,
                "language": "fa",
                "max_sections": 5,
                "include_references": False
            }
        elif params is None:
            params = {"query": query}

        payload = {
            "model_id": self.model_id,
            "data_type": data_type,
            "data_source": data_source,
            "parameters": params,
            "response_topic": self.response_topic
        }
        message = self._build_message(destination, operation, payload, priority=2, request_source="user")

        topic = self.topic_manager.get_topic("DATA_REQUESTS_TOPIC")
        self.logger.info(f"[ModuleConnector] Sending data request to Data via Balance on topic {topic}")
        try:
            future = await self.producer.send(topic, message)
            await self.producer.flush()
            self.logger.info(
                f"[ModuleConnector] Data request sent with request_id: {message['metadata']['request_id']}")
            return message["metadata"]["request_id"]
        except Exception as e:
            self.logger.error(f"[ModuleConnector] Error in fetch_data: {str(e)}")
            return None

    async def share_knowledge(self, knowledge_payload: Dict[str, Any]) -> bool:
        """
        ارسال دانش به ماژول Federation جهت اشتراک‌گذاری.

        Args:
            knowledge_payload (Dict[str, Any]): اطلاعات دانش جهت به اشتراک‌گذاری.

        Returns:
            bool: نتیجه موفقیت‌آمیز بودن ارسال دانش.
        """
        destination = "federation"
        operation = "SHARE_KNOWLEDGE"
        payload = {
            "knowledge": knowledge_payload,
            "response_topic": self.response_topic
        }
        message = self._build_message(destination, operation, payload, priority=1, request_source="system")

        topic = self.topic_manager.get_topic("FEDERATION_REQUESTS_TOPIC")
        self.logger.info(f"[ModuleConnector] Sharing knowledge via Federation on topic {topic}")
        try:
            future = await self.producer.send(topic, message)
            await self.producer.flush()
            self.logger.info(f"[ModuleConnector] Knowledge shared with request_id: {message['metadata']['request_id']}")
            return True
        except Exception as e:
            self.logger.error(f"[ModuleConnector] Error in share_knowledge: {str(e)}")
            return False

    async def notify_module(self, module_name: str, event_data: Dict[str, Any]) -> bool:
        """
        ارسال اعلان به یک ماژول مشخص از طریق Kafka.

        Args:
            module_name (str): نام ماژول مقصد (مثلاً "balance" یا "data").
            event_data (Dict[str, Any]): داده‌های رویداد برای اطلاع‌رسانی.

        Returns:
            bool: نتیجه ارسال اعلان.
        """
        destination = module_name.lower()
        operation = "NOTIFY_EVENT"
        payload = {
            "event_data": event_data,
            "response_topic": self.response_topic
        }
        message = self._build_message(destination, operation, payload, priority=2, request_source="system")

        # انتخاب موضوع بر اساس ماژول مقصد
        if destination == "balance":
            topic = self.topic_manager.get_topic("BALANCE_EVENTS_TOPIC")
        elif destination == "data":
            topic = self.topic_manager.get_topic("DATA_EVENTS_TOPIC")
        else:
            topic = self.topic_manager.get_topic(f"{destination.upper()}_EVENTS_TOPIC")

        self.logger.info(f"[ModuleConnector] Notifying module '{module_name}' on topic {topic}")
        try:
            future = await self.producer.send(topic, message)
            await self.producer.flush()
            self.logger.info(
                f"[ModuleConnector] Notification sent with request_id: {message['metadata']['request_id']}")
            return True
        except Exception as e:
            self.logger.error(f"[ModuleConnector] Error in notify_module for {module_name}: {str(e)}")
            return False

"""
KnowledgeSharing Module
-------------------------
این فایل مسئول به اشتراک‌گذاری دانش بین مدل‌های زبان در سیستم Federation است.
این کلاس پیام‌های دانش را طبق فرمت استاندارد ایجاد کرده و از طریق سیستم پیام‌رسانی (مثلاً Kafka)
به سایر مدل‌ها ارسال می‌کند. این امر به افزایش همگام‌سازی و انتقال دانش میان مدل‌های زبانی کمک می‌کند.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import logging
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class KnowledgeSharing(BaseComponent):
    """
    KnowledgeSharing مسئول به اشتراک‌گذاری دانش بین مدل‌های زبان در سیستم Federation است.

    ویژگی‌ها:
      - ساخت پیام استاندارد شامل بخش‌های metadata و payload.
      - ارسال دانش به سایر مدل‌ها از طریق Kafka Producer.
      - ثبت و گزارش متریک‌های مرتبط با اشتراک دانش.
    """

    def __init__(self, producer: Any, topic_manager: Any, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه KnowledgeSharing.

        Args:
            producer (Any): شیء Kafka Producer (ناهمزمان) جهت ارسال پیام.
            topic_manager (Any): شیء مدیریت موضوعات (Topic Manager) جهت دریافت نام موضوع.
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل تنظیمات مانند:
                - "default_topic": موضوع پیش‌فرض برای ارسال دانش (پیش‌فرض: "FEDERATION_KNOWLEDGE_TOPIC")
                - "priority": سطح اولویت پیام (پیش‌فرض: 1)
        """
        super().__init__(component_type="knowledge_sharing", config=config)
        self.logger = logging.getLogger("KnowledgeSharing")
        self.producer = producer
        self.topic_manager = topic_manager
        self.default_topic = self.config.get("default_topic", "FEDERATION_KNOWLEDGE_TOPIC")
        self.priority = int(self.config.get("priority", 1))
        self.logger.info(
            f"[KnowledgeSharing] Initialized with default_topic={self.default_topic} and priority={self.priority}")

    def _build_message(self, knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """
        ساخت پیام استاندارد جهت به اشتراک‌گذاری دانش.

        Args:
            knowledge (Dict[str, Any]): دانش جهت به اشتراک‌گذاری.

        Returns:
            Dict[str, Any]: پیام استاندارد شامل metadata و payload.
        """
        message = {
            "metadata": {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "source": "federation_knowledge_sharing",
                "destination": "all_models",
                "priority": self.priority
            },
            "payload": {
                "operation": "SHARE_KNOWLEDGE",
                "knowledge": knowledge
            }
        }
        return message

    async def share_knowledge(self, knowledge: Dict[str, Any]) -> bool:
        """
        ارسال دانش به سایر مدل‌ها از طریق سیستم Federation.

        Args:
            knowledge (Dict[str, Any]): دانش جهت به اشتراک‌گذاری.

        Returns:
            bool: نتیجه ارسال پیام؛ True در صورت موفقیت، False در صورت خطا.
        """
        message = self._build_message(knowledge)
        topic = self.topic_manager.get_topic(self.default_topic)
        self.logger.info(
            f"[KnowledgeSharing] Sharing knowledge on topic '{topic}' with message_id {message['metadata']['message_id']}.")
        try:
            await self.producer.send(topic, message)
            await self.producer.flush()
            self.logger.info("[KnowledgeSharing] Knowledge shared successfully.")
            self.increment_metric("knowledge_shared")
            return True
        except Exception as e:
            self.logger.error(f"[KnowledgeSharing] Error sharing knowledge: {str(e)}")
            self.record_error_metric()
            return False


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)


    # Mock implementations for producer and topic_manager for testing purposes.
    class MockProducer:
        async def send(self, topic, message):
            print(f"Mock sending to topic '{topic}': {message}")

        async def flush(self):
            pass


    class MockTopicManager:
        def get_topic(self, topic_name):
            return topic_name


    async def main():
        producer = MockProducer()
        topic_manager = MockTopicManager()
        ks = KnowledgeSharing(producer=producer, topic_manager=topic_manager)
        sample_knowledge = {
            "topic": "Neural Networks",
            "content": "Advances in deep learning and convolutional networks are reshaping the AI landscape.",
            "source": "Research Paper",
            "timestamp": datetime.now().isoformat()
        }
        result = await ks.share_knowledge(sample_knowledge)
        print("Knowledge sharing result:", result)


    asyncio.run(main())

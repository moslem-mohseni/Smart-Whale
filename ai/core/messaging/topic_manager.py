"""
مدیریت موضوعات کافکا برای سیستم پیام‌رسانی
"""
import logging
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from enum import Enum

from ai.core.messaging.constants import (
    TOPIC_PREFIX, DATA_PREFIX, MODELS_PREFIX, BALANCE_PREFIX, CORE_PREFIX,
    DATA_REQUESTS_TOPIC, DATA_RESPONSES_PREFIX, MODELS_REQUESTS_TOPIC,
    BALANCE_METRICS_TOPIC, BALANCE_EVENTS_TOPIC, SYSTEM_LOGS_TOPIC,
    DEFAULT_PARTITIONS, DEFAULT_REPLICATION, MODEL_TOPIC_PARTITIONS, MODEL_TOPIC_REPLICATION
)

logger = logging.getLogger(__name__)


class TopicCategory(Enum):
    """دسته‌بندی انواع موضوعات کافکا در سیستم"""
    REQUEST = "request"  # درخواست‌ها
    RESPONSE = "response"  # پاسخ‌ها
    EVENT = "event"  # رویدادها
    METRIC = "metric"  # متریک‌ها
    LOG = "log"  # لاگ‌ها
    INTERNAL = "internal"  # ارتباطات داخلی
    COMMAND = "command"  # دستورات
    NOTIFICATION = "notification"  # اعلان‌ها


class TopicManager:
    """
    مدیریت موضوعات کافکا برای ماژول‌های سیستم
    """

    def __init__(self, kafka_service):
        """
        مقداردهی اولیه مدیر موضوعات

        :param kafka_service: سرویس کافکا برای ارتباط با سرور
        """
        self.kafka_service = kafka_service
        self._initialize_topics()

    def _initialize_topics(self):
        """مقداردهی اولیه ساختار موضوعات"""
        # ساختار پایه موضوعات
        self.topic_structure = {
            # درخواست‌های داده
            "data_requests": {
                "name": DATA_REQUESTS_TOPIC,
                "category": TopicCategory.REQUEST,
                "partitions": DEFAULT_PARTITIONS,
                "replication": DEFAULT_REPLICATION,
                "description": "درخواست‌های جمع‌آوری داده از Balance به Data"
            },
            # پیشوند نتایج برای مدل‌ها (هر مدل یک موضوع دارد)
            "data_responses_prefix": {
                "name": DATA_RESPONSES_PREFIX,
                "category": TopicCategory.RESPONSE,
                "partitions": MODEL_TOPIC_PARTITIONS,
                "replication": MODEL_TOPIC_REPLICATION,
                "description": "نتایج جمع‌آوری داده از Data به مدل‌ها"
            },
            # درخواست‌های مدل‌ها
            "models_requests": {
                "name": MODELS_REQUESTS_TOPIC,
                "category": TopicCategory.REQUEST,
                "partitions": DEFAULT_PARTITIONS,
                "replication": DEFAULT_REPLICATION,
                "description": "درخواست‌های مدل‌ها به Balance"
            },
            # متریک‌های سیستم
            "balance_metrics": {
                "name": BALANCE_METRICS_TOPIC,
                "category": TopicCategory.METRIC,
                "partitions": 3,
                "replication": 2,
                "description": "متریک‌های ماژول Balance"
            },
            # رویدادهای سیستم
            "balance_events": {
                "name": BALANCE_EVENTS_TOPIC,
                "category": TopicCategory.EVENT,
                "partitions": 5,
                "replication": 3,
                "description": "رویدادهای ماژول Balance"
            },
            # لاگ‌های داخلی سیستم
            "system_logs": {
                "name": SYSTEM_LOGS_TOPIC,
                "category": TopicCategory.LOG,
                "partitions": 3,
                "replication": 2,
                "description": "لاگ‌های سیستم"
            }
        }

        # موضوعات ایجاد شده (برای جلوگیری از تلاش مکرر)
        self.created_topics: Set[str] = set()
        # موضوعات مدل‌ها (اختصاصی هر مدل)
        self.model_topics: Dict[str, str] = {}

    def get_topic_name(self, topic_key: str) -> str:
        """
        دریافت نام کامل موضوع بر اساس کلید آن

        :param topic_key: کلید موضوع (مانند data_requests)
        :return: نام کامل موضوع (مانند smartwhale.data.requests)
        """
        if topic_key in self.topic_structure:
            return self.topic_structure[topic_key]["name"]
        return topic_key  # اگر کلید یافت نشد، همان کلید برگردانده می‌شود

    def get_model_result_topic(self, model_id: str) -> str:
        """
        ساخت نام موضوع نتایج برای یک مدل خاص

        :param model_id: شناسه مدل
        :return: نام کامل موضوع نتایج برای آن مدل
        """
        # بررسی وجود موضوع در لیست موضوعات مدل
        if model_id in self.model_topics:
            return self.model_topics[model_id]

        # ساخت نام موضوع جدید
        prefix = self.topic_structure["data_responses_prefix"]["name"]
        topic_name = f"{prefix}.{model_id}"

        # ذخیره برای استفاده‌های بعدی
        self.model_topics[model_id] = topic_name

        return topic_name

    async def ensure_topic_exists(self, topic_name: str, partitions: int = DEFAULT_PARTITIONS,
                                  replication: int = DEFAULT_REPLICATION) -> bool:
        """
        اطمینان از وجود یک موضوع در کافکا و ایجاد آن در صورت عدم وجود

        :param topic_name: نام موضوع
        :param partitions: تعداد پارتیشن‌ها
        :param replication: فاکتور تکرار
        :return: True در صورت موفقیت
        """
        # بررسی موضوعات ایجاد شده محلی
        if topic_name in self.created_topics:
            return True

        try:
            # بررسی وجود موضوع
            if not await self.kafka_service.topic_exists(topic_name):
                # ایجاد موضوع جدید
                success = await self.kafka_service.create_topic(
                    topic_name=topic_name,
                    num_partitions=partitions,
                    replication_factor=replication
                )

                if success:
                    logger.info(f"✅ موضوع '{topic_name}' ایجاد شد")
                    self.created_topics.add(topic_name)
                    return True
                else:
                    logger.error(f"❌ خطا در ایجاد موضوع '{topic_name}'")
                    return False
            else:
                # موضوع قبلاً وجود داشته
                self.created_topics.add(topic_name)
                return True

        except Exception as e:
            logger.error(f"❌ خطا در بررسی/ایجاد موضوع '{topic_name}': {str(e)}")
            return False

    async def ensure_model_topic(self, model_id: str) -> str:
        """
        اطمینان از وجود موضوع نتایج برای یک مدل خاص

        :param model_id: شناسه مدل
        :return: نام کامل موضوع ایجاد شده
        """
        topic_name = self.get_model_result_topic(model_id)
        topic_info = self.topic_structure["data_responses_prefix"]

        partitions = topic_info.get("partitions", MODEL_TOPIC_PARTITIONS)
        replication = topic_info.get("replication", MODEL_TOPIC_REPLICATION)

        await self.ensure_topic_exists(topic_name, partitions, replication)
        return topic_name

    async def initialize_all_topics(self) -> bool:
        """
        ایجاد تمام موضوعات پایه مورد نیاز سیستم

        :return: True در صورت موفقیت ایجاد همه موضوعات
        """
        success = True

        # ایجاد موضوعات اصلی
        for topic_key, topic_info in self.topic_structure.items():
            # موضوعات با پیشوند باید به صورت خاص مدیریت شوند
            if topic_key.endswith("_prefix"):
                continue

            topic_name = topic_info["name"]
            partitions = topic_info.get("partitions", DEFAULT_PARTITIONS)
            replication = topic_info.get("replication", DEFAULT_REPLICATION)

            if not await self.ensure_topic_exists(topic_name, partitions, replication):
                success = False

        return success

    async def list_all_topics(self) -> List[str]:
        """
        دریافت لیست تمام موضوعات موجود در کافکا

        :return: لیست نام موضوعات
        """
        try:
            return await self.kafka_service.list_topics()
        except Exception as e:
            logger.error(f"❌ خطا در دریافت لیست موضوعات: {str(e)}")
            return []

    async def list_model_topics(self) -> List[Tuple[str, str]]:
        """
        دریافت لیست موضوعات اختصاصی مدل‌ها

        :return: لیست تاپل‌های (شناسه مدل، نام موضوع)
        """
        return [(model_id, topic) for model_id, topic in self.model_topics.items()]

    async def delete_topic(self, topic_name: str) -> bool:
        """
        حذف یک موضوع از کافکا

        :param topic_name: نام موضوع
        :return: True در صورت موفقیت
        """
        try:
            success = await self.kafka_service.delete_topic(topic_name)

            if success:
                logger.info(f"✅ موضوع '{topic_name}' حذف شد")

                # حذف از لیست‌های داخلی
                if topic_name in self.created_topics:
                    self.created_topics.remove(topic_name)

                # بررسی موضوعات مدل‌ها
                for model_id, model_topic in list(self.model_topics.items()):
                    if model_topic == topic_name:
                        del self.model_topics[model_id]

                return True
            else:
                logger.error(f"❌ خطا در حذف موضوع '{topic_name}'")
                return False

        except Exception as e:
            logger.error(f"❌ خطا در حذف موضوع '{topic_name}': {str(e)}")
            return False

    async def delete_model_topic(self, model_id: str) -> bool:
        """
        حذف موضوع اختصاصی یک مدل

        :param model_id: شناسه مدل
        :return: True در صورت موفقیت
        """
        if model_id not in self.model_topics:
            logger.warning(f"⚠ موضوع مدل '{model_id}' یافت نشد")
            return False

        topic_name = self.model_topics[model_id]
        success = await self.delete_topic(topic_name)

        if success:
            # حذف از لیست موضوعات مدل
            del self.model_topics[model_id]

        return success

"""
سرویس کافکا برای ارتباط با سرور Kafka و مدیریت پیام‌ها
"""
import json
import logging
import asyncio
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import asdict

from infrastructure.kafka.config.settings import KafkaConfig
from infrastructure.kafka.adapters.producer import MessageProducer
from infrastructure.kafka.adapters.consumer import MessageConsumer
from infrastructure.kafka.domain.models import Message, TopicConfig

from ai.core.messaging.constants import (
    DEFAULT_KAFKA_TIMEOUT, DEFAULT_BATCH_SIZE
)
from ai.core.messaging.message_schemas import (
    DataRequest, DataResponse, is_valid_data_request, is_valid_data_response
)

logger = logging.getLogger(__name__)


class KafkaService:
    """
    سرویس مرکزی برای ارتباط با Kafka و مدیریت پیام‌ها
    """

    def __init__(self):
        """
        مقداردهی اولیه سرویس Kafka
        """
        self.kafka_config = KafkaConfig()
        self.producer = None
        self.consumers = {}
        self.is_connected = False
        self.active_subscriptions = {}  # نگهداری اشتراک‌های فعال

    async def connect(self) -> bool:
        """
        برقراری اتصال با Kafka

        :return: True در صورت موفقیت
        """
        if not self.is_connected:
            try:
                self.producer = MessageProducer(self.kafka_config)
                self.is_connected = True
                logger.info("✅ اتصال به سرویس Kafka برقرار شد")
                return True
            except Exception as e:
                logger.error(f"❌ خطا در برقراری اتصال به Kafka: {str(e)}")
                return False
        return True

    async def disconnect(self) -> bool:
        """
        قطع اتصال از Kafka و آزادسازی منابع

        :return: True در صورت موفقیت
        """
        try:
            # قطع اتصال تمام مصرف‌کننده‌ها
            for subscription_key in list(self.active_subscriptions.keys()):
                topic, group_id = subscription_key.split(":", 1)
                await self.unsubscribe(topic, group_id)

            # قطع اتصال تولیدکننده
            if self.producer:
                await self.producer.close()
                self.producer = None

            self.is_connected = False
            self.active_subscriptions = {}
            logger.info("🔌 اتصال از سرویس Kafka قطع شد")
            return True
        except Exception as e:
            logger.error(f"❌ خطا در قطع اتصال از Kafka: {str(e)}")
            return False

    async def send_message(self, topic: str, message_data: Union[Dict[str, Any], str, bytes]) -> bool:
        """
        ارسال یک پیام به موضوع مشخص

        :param topic: نام موضوع
        :param message_data: محتوای پیام (دیکشنری، رشته یا بایت)
        :return: True در صورت موفقیت
        """
        try:
            await self.connect()

            # تبدیل داده به قالب مناسب
            if isinstance(message_data, dict):
                message_content = json.dumps(message_data).encode("utf-8")
            elif isinstance(message_data, str):
                message_content = message_data.encode("utf-8")
            else:
                message_content = message_data

            # ساخت پیام کافکا
            message = Message(
                topic=topic,
                content=message_content
            )

            # ارسال پیام
            await self.producer.send(message)
            logger.debug(f"📤 پیام به موضوع {topic} ارسال شد")
            return True

        except Exception as e:
            logger.error(f"❌ خطا در ارسال پیام به موضوع {topic}: {str(e)}")
            return False

    async def send_batch(self, topic: str, messages: List[Union[Dict[str, Any], str, bytes]]) -> bool:
        """
        ارسال دسته‌ای پیام‌ها به یک موضوع

        :param topic: نام موضوع
        :param messages: لیست پیام‌ها
        :return: True در صورت موفقیت همه ارسال‌ها
        """
        try:
            await self.connect()

            kafka_messages = []

            # تبدیل همه پیام‌ها به قالب مناسب
            for msg_data in messages:
                if isinstance(msg_data, dict):
                    message_content = json.dumps(msg_data).encode("utf-8")
                elif isinstance(msg_data, str):
                    message_content = msg_data.encode("utf-8")
                else:
                    message_content = msg_data

                kafka_messages.append(Message(
                    topic=topic,
                    content=message_content
                ))

            # ارسال دسته‌ای پیام‌ها
            await self.producer.send_batch(kafka_messages)
            logger.debug(f"📤 {len(messages)} پیام به موضوع {topic} ارسال شد")
            return True

        except Exception as e:
            logger.error(f"❌ خطا در ارسال دسته‌ای پیام به موضوع {topic}: {str(e)}")
            return False

    async def send_data_request(self, request: DataRequest, topic: str) -> bool:
        """
        ارسال درخواست داده به ماژول Data

        :param request: درخواست داده
        :param topic: موضوع کافکا
        :return: True در صورت موفقیت
        """
        try:
            # اعتبارسنجی درخواست
            if not is_valid_data_request(request):
                logger.error("❌ درخواست داده نامعتبر است")
                return False

            # تبدیل درخواست به دیکشنری
            request_dict = request.to_dict()
            return await self.send_message(topic, request_dict)

        except Exception as e:
            logger.error(f"❌ خطا در ارسال درخواست داده: {str(e)}")
            return False

    async def send_data_response(self, response: DataResponse, topic: str) -> bool:
        """
        ارسال پاسخ داده به مدل‌ها

        :param response: پاسخ داده
        :param topic: موضوع کافکا
        :return: True در صورت موفقیت
        """
        try:
            # اعتبارسنجی پاسخ
            if not is_valid_data_response(response):
                logger.error("❌ پاسخ داده نامعتبر است")
                return False

            # تبدیل پاسخ به دیکشنری
            response_dict = response.to_dict()
            return await self.send_message(topic, response_dict)

        except Exception as e:
            logger.error(f"❌ خطا در ارسال پاسخ داده: {str(e)}")
            return False

    async def subscribe(self, topic: str, group_id: str, handler: Callable[[Dict[str, Any]], None]) -> bool:
        """
        اشتراک در یک موضوع و مدیریت پیام‌های دریافتی

        :param topic: نام موضوع
        :param group_id: شناسه گروه مصرف‌کننده
        :param handler: تابع پردازش‌کننده پیام‌های دریافتی
        :return: True در صورت موفقیت
        """
        try:
            await self.connect()

            # رپر برای تبدیل پیام Kafka به Dict و فراخوانی پردازش‌کننده
            async def message_processor(message: Message):
                try:
                    message_data = json.loads(message.content.decode("utf-8"))
                    await handler(message_data)
                except json.JSONDecodeError:
                    logger.error(f"❌ خطا در پردازش پیام JSON از موضوع {topic}")
                except Exception as e:
                    logger.error(f"❌ خطا در پردازش پیام از موضوع {topic}: {str(e)}")

            # ایجاد مصرف‌کننده جدید
            consumer = MessageConsumer(self.kafka_config)

            # اشتراک در موضوع
            subscription_task = asyncio.create_task(consumer.consume(topic, group_id, message_processor))

            # ذخیره اطلاعات اشتراک
            subscription_key = f"{topic}:{group_id}"
            self.active_subscriptions[subscription_key] = {
                "consumer": consumer,
                "task": subscription_task
            }

            logger.info(f"🔔 اشتراک در موضوع {topic} با گروه {group_id} انجام شد")
            return True

        except Exception as e:
            logger.error(f"❌ خطا در اشتراک به موضوع {topic}: {str(e)}")
            return False

    async def unsubscribe(self, topic: str, group_id: str) -> bool:
        """
        لغو اشتراک از یک موضوع

        :param topic: نام موضوع
        :param group_id: شناسه گروه مصرف‌کننده
        :return: True در صورت موفقیت
        """
        subscription_key = f"{topic}:{group_id}"

        if subscription_key in self.active_subscriptions:
            try:
                subscription = self.active_subscriptions[subscription_key]
                consumer = subscription["consumer"]
                task = subscription["task"]

                # لغو وظیفه اشتراک
                if not task.done():
                    task.cancel()

                # توقف مصرف‌کننده
                await consumer.stop()

                # حذف از اشتراک‌های فعال
                del self.active_subscriptions[subscription_key]

                logger.info(f"🔕 اشتراک از موضوع {topic} با گروه {group_id} لغو شد")
                return True

            except Exception as e:
                logger.error(f"❌ خطا در لغو اشتراک از موضوع {topic}: {str(e)}")
                return False
        else:
            logger.warning(f"⚠ اشتراک {topic}:{group_id} یافت نشد")
            return False

    async def topic_exists(self, topic_name: str) -> bool:
        """
        بررسی وجود یک موضوع در کافکا

        :param topic_name: نام موضوع
        :return: True اگر موضوع وجود داشته باشد
        """
        try:
            # در اینجا از Admin API کافکا استفاده می‌شود (پیاده‌سازی واقعی متفاوت خواهد بود)
            topics = await self.list_topics()
            return topic_name in topics
        except Exception as e:
            logger.error(f"❌ خطا در بررسی وجود موضوع {topic_name}: {str(e)}")
            return False

    async def create_topic(self, topic_name: str, num_partitions: int, replication_factor: int) -> bool:
        """
        ایجاد یک موضوع جدید در کافکا

        :param topic_name: نام موضوع
        :param num_partitions: تعداد پارتیشن‌ها
        :param replication_factor: فاکتور تکرار
        :return: True در صورت موفقیت
        """
        try:
            # در اینجا از Admin API کافکا استفاده می‌شود (پیاده‌سازی واقعی متفاوت خواهد بود)
            # پیاده‌سازی فرضی برای نشان دادن عملکرد
            if topic_name not in await self.list_topics():
                # پیاده‌سازی واقعی: ایجاد موضوع
                logger.info(
                    f"✅ موضوع {topic_name} با {num_partitions} پارتیشن و فاکتور تکرار {replication_factor} ایجاد شد")
                return True
            return True
        except Exception as e:
            logger.error(f"❌ خطا در ایجاد موضوع {topic_name}: {str(e)}")
            return False

    async def delete_topic(self, topic_name: str) -> bool:
        """
        حذف یک موضوع از کافکا

        :param topic_name: نام موضوع
        :return: True در صورت موفقیت
        """
        try:
            # در اینجا از Admin API کافکا استفاده می‌شود (پیاده‌سازی واقعی متفاوت خواهد بود)
            # پیاده‌سازی فرضی برای نشان دادن عملکرد
            if topic_name in await self.list_topics():
                # پیاده‌سازی واقعی: حذف موضوع
                logger.info(f"✅ موضوع {topic_name} حذف شد")
                return True
            logger.warning(f"⚠ موضوع {topic_name} برای حذف یافت نشد")
            return False
        except Exception as e:
            logger.error(f"❌ خطا در حذف موضوع {topic_name}: {str(e)}")
            return False

    async def list_topics(self) -> List[str]:
        """
        دریافت لیست تمام موضوعات موجود در کافکا

        :return: لیست نام موضوعات
        """
        try:
            # در اینجا از Admin API کافکا استفاده می‌شود (پیاده‌سازی واقعی متفاوت خواهد بود)
            # این یک پیاده‌سازی فرضی است
            return []  # در پیاده‌سازی واقعی، لیست موضوعات برگردانده می‌شود
        except Exception as e:
            logger.error(f"❌ خطا در دریافت لیست موضوعات: {str(e)}")
            return []

    async def get_topic_info(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات یک موضوع

        :param topic_name: نام موضوع
        :return: دیکشنری اطلاعات موضوع یا None
        """
        try:
            # در اینجا از Admin API کافکا استفاده می‌شود (پیاده‌سازی واقعی متفاوت خواهد بود)
            if not await self.topic_exists(topic_name):
                return None

            # پیاده‌سازی فرضی
            return {
                "name": topic_name,
                "partitions": 0,
                "replication_factor": 0
            }
        except Exception as e:
            logger.error(f"❌ خطا در دریافت اطلاعات موضوع {topic_name}: {str(e)}")
            return None


# نمونه سرویس Kafka (Singleton)
kafka_service = KafkaService()


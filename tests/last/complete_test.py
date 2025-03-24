#!/usr/bin/env python
"""
اسکریپت کامل برای تست فرآیند ثبت مدل، ارسال درخواست جمع‌آوری داده و دریافت پاسخ
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# تنظیم لاگینگ
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# تنظیمات کافکا
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
MODELS_REQUESTS_TOPIC = 'smartwhale.models.requests'


class SmartWhaleClient:
    """کلاینت برای تعامل با سیستم Smart Whale"""

    def __init__(self, model_id=None, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS):
        """مقداردهی اولیه"""
        self.bootstrap_servers = bootstrap_servers
        self.model_id = model_id or f"test_model_{uuid.uuid4().hex[:8]}"
        self.response_topic = f"smartwhale.models.responses.{self.model_id}"
        self.producer = None
        self.consumer = None
        self.initialize()

    def initialize(self):
        """راه‌اندازی اتصالات Kafka"""
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info(f"اتصال به Kafka برقرار شد: {self.bootstrap_servers}")
        except Exception as e:
            logger.error(f"خطا در اتصال به Kafka: {e}")
            raise

    def register_model(self):
        """ثبت مدل در سیستم"""
        request_id = str(uuid.uuid4())

        register_message = {
            "metadata": {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "source": "client",
                "destination": "balance",
                "priority": 1,
                "request_source": "user"
            },
            "payload": {
                "operation": "REGISTER_MODEL",
                "model_id": self.model_id,
                "parameters": {
                    "model_type": "test",
                    "version": "1.0",
                    "description": "مدل تست برای جمع‌آوری داده",
                    "capabilities": ["text_processing"]
                },
                "response_topic": self.response_topic
            }
        }

        try:
            future = self.producer.send(MODELS_REQUESTS_TOPIC, register_message)
            self.producer.flush()

            record_metadata = future.get(timeout=10)
            logger.info(f"درخواست ثبت مدل با ID {self.model_id} به موضوع {record_metadata.topic} ارسال شد")
            logger.info(f"موضوع پاسخ: {self.response_topic}")

            return True
        except KafkaError as e:
            logger.error(f"خطا در ثبت مدل: {e}")
            return False

    def request_data(self, query, data_source="WIKI", data_type="TEXT", params=None):
        """ارسال درخواست جمع‌آوری داده"""
        request_id = str(uuid.uuid4())

        # تنظیم پارامترهای پیش‌فرض برای ویکی‌پدیا
        if data_source == "WIKI" and not params:
            params = {
                "title": query,
                "language": "fa",
                "max_sections": 5,
                "include_references": False
            }

        # تنظیم پارامترهای عمومی
        if not params:
            params = {"query": query}

        request_message = {
            "metadata": {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "source": "client",
                "destination": "balance",
                "priority": 2,
                "request_source": "user"
            },
            "payload": {
                "operation": "FETCH_DATA",
                "model_id": self.model_id,
                "data_type": data_type,
                "data_source": data_source,
                "parameters": params,
                "response_topic": self.response_topic
            }
        }

        try:
            future = self.producer.send(MODELS_REQUESTS_TOPIC, request_message)
            self.producer.flush()

            record_metadata = future.get(timeout=10)
            logger.info(f"درخواست داده با ID {request_id} به موضوع {record_metadata.topic} ارسال شد")

            return request_id
        except KafkaError as e:
            logger.error(f"خطا در ارسال درخواست داده: {e}")
            return None

    def listen_for_response(self, request_id=None, timeout=120):
        """گوش دادن به موضوع پاسخ برای دریافت نتیجه"""
        try:
            # ایجاد consumer
            consumer = KafkaConsumer(
                self.response_topic,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='earliest',
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                group_id=f"client_{uuid.uuid4().hex[:8]}"
            )

            logger.info(f"در حال گوش دادن به پاسخ‌ها در موضوع {self.response_topic}...")

            # تنظیم زمان انتظار
            start_time = time.time()

            while time.time() - start_time < timeout:
                # بررسی پیام‌های جدید
                msg_pack = consumer.poll(timeout_ms=1000)

                if not msg_pack:
                    continue

                # بررسی تمام پیام‌های دریافتی
                for tp, messages in msg_pack.items():
                    for message in messages:
                        response_data = message.value

                        # نمایش همه پیام‌های دریافتی
                        logger.info("پیام دریافت شد:")
                        if request_id and response_data['metadata'].get('request_id') == request_id:
                            logger.info("***** پاسخ به درخواست ما یافت شد *****")

                        logger.info(json.dumps(response_data, indent=2, ensure_ascii=False))

                        # اگر این پاسخ درخواست ما باشد، آن را برمی‌گردانیم
                        if not request_id or response_data['metadata'].get('request_id') == request_id:
                            consumer.close()
                            return response_data

            logger.warning(f"زمان انتظار برای پاسخ به اتمام رسید (بعد از {timeout} ثانیه)")
            consumer.close()
            return None
        except Exception as e:
            logger.error(f"خطا در دریافت پاسخ: {e}")
            return None

    def close(self):
        """بستن اتصالات"""
        if self.producer:
            self.producer.close()
        logger.info("اتصال‌ها بسته شدند.")


def print_wiki_result(data):
    """نمایش نتیجه جستجو در ویکی‌پدیا"""
    print("\n" + "=" * 50)
    print(f"عنوان: {data.get('title', 'نامشخص')}")
    print(f"منبع: {data.get('source_url', 'نامشخص')}")

    if 'extract' in data:
        print("\nخلاصه مقاله:")
        print("-" * 50)
        print(data['extract'][:500] + "...")

    if 'sections' in data:
        print(f"\nبخش‌های مقاله ({len(data['sections'])} بخش):")
        print("-" * 50)
        for i, section in enumerate(data['sections'], 1):
            title = section.get('title', 'بدون عنوان')
            content = section.get('content', '')
            print(f"{i}. {title}")
            if content:
                print(f"   {content[:100]}...")

    print("=" * 50)


def main():
    """تابع اصلی برنامه"""
    parser = argparse.ArgumentParser(description="تست سیستم Smart Whale")
    parser.add_argument('--query', '-q', default="هوش مصنوعی", help='عبارت جستجو')
    parser.add_argument('--source', '-s', default="WIKI", help='منبع داده')
    parser.add_argument('--wait', '-w', type=int, default=120, help='زمان انتظار برای پاسخ (ثانیه)')

    args = parser.parse_args()

    client = SmartWhaleClient()

    try:
        # ثبت مدل
        logger.info("در حال ثبت مدل در سیستم...")
        if not client.register_model():
            logger.error("ثبت مدل با شکست مواجه شد.")
            return

        # ۲ ثانیه صبر می‌کنیم تا مدل ثبت شود
        time.sleep(2)

        # ارسال درخواست داده
        logger.info(f"در حال ارسال درخواست جستجو برای '{args.query}' از منبع {args.source}...")
        request_id = client.request_data(args.query, args.source)

        if not request_id:
            logger.error("ارسال درخواست با شکست مواجه شد.")
            return

        # دریافت پاسخ
        logger.info(f"در حال انتظار برای دریافت پاسخ (حداکثر {args.wait} ثانیه)...")
        response = client.listen_for_response(request_id, args.wait)

        if response:
            logger.info("پاسخ با موفقیت دریافت شد!")

            if response['payload'].get('status') == 'success':
                data = response['payload'].get('data', {})
                print_wiki_result(data)
            else:
                error = response['payload'].get('error_message', 'خطای نامشخص')
                logger.error(f"خطا در پردازش درخواست: {error}")
        else:
            logger.warning("پاسخی دریافت نشد.")

    finally:
        client.close()


if __name__ == "__main__":
    main()

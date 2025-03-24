#!/usr/bin/env python
"""
اسکریپت ارسال درخواست جمع‌آوری داده از ویکی‌پدیا به ماژول Balance
"""

import asyncio
import json
import uuid
import logging
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تنظیمات کافکا
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
MODELS_REQUESTS_TOPIC = 'smartwhale.models.requests'

# شناسه مدل و پاسخ
MODEL_ID = f"test_model_{uuid.uuid4().hex[:8]}"
RESPONSE_TOPIC = f"smartwhale.models.responses.{MODEL_ID}"


def send_request_to_balance():
    """ارسال درخواست به ماژول Balance از طریق Kafka"""
    try:
        # ایجاد producer کافکا
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )

        # ایجاد شناسه یکتا برای درخواست
        request_id = str(uuid.uuid4())

        # ساخت پیام درخواست
        request_message = {
            "metadata": {
                "request_id": request_id,
                "timestamp": "2023-11-15T10:00:00.000Z",
                "source": "client",
                "destination": "balance",
                "priority": 2,  # اولویت بالا
                "request_source": "user"
            },
            "payload": {
                "operation": "FETCH_DATA",
                "model_id": MODEL_ID,
                "data_type": "TEXT",
                "data_source": "WIKI",
                "parameters": {
                    "title": "هوش مصنوعی",
                    "language": "fa",
                    "max_sections": 5,
                    "include_references": False
                },
                "response_topic": RESPONSE_TOPIC
            }
        }

        # ارسال پیام به موضوع درخواست‌های مدل‌ها
        future = producer.send(MODELS_REQUESTS_TOPIC, request_message)
        producer.flush()

        # بررسی موفقیت ارسال
        try:
            record_metadata = future.get(timeout=10)
            logger.info(f"درخواست با ID {request_id} به موضوع {record_metadata.topic} ارسال شد")
            logger.info(f"- پارتیشن: {record_metadata.partition}")
            logger.info(f"- آفست: {record_metadata.offset}")
            return request_id
        except KafkaError as e:
            logger.error(f"خطا در ارسال درخواست: {e}")
            return None
    except Exception as e:
        logger.error(f"خطا در اتصال به کافکا: {e}")
        return None


def listen_for_response(request_id):
    """گوش دادن به موضوع پاسخ برای دریافت نتیجه"""
    try:
        # ایجاد consumer کافکا
        consumer = KafkaConsumer(
            RESPONSE_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='earliest',
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f"client_{uuid.uuid4().hex[:8]}"
        )

        logger.info(f"در انتظار دریافت پاسخ از موضوع {RESPONSE_TOPIC}...")

        # تنظیم timeout برای گوش دادن
        timeout = 60  # ۶۰ ثانیه منتظر می‌مانیم
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

                    # بررسی که آیا این پاسخ برای درخواست ما است
                    if response_data['metadata'].get('request_id') == request_id:
                        logger.info("پاسخ دریافت شد:")
                        logger.info(json.dumps(response_data, indent=2, ensure_ascii=False))
                        consumer.close()
                        return response_data

        logger.warning(f"زمان انتظار برای پاسخ به اتمام رسید (بعد از {timeout} ثانیه)")
        consumer.close()
        return None
    except Exception as e:
        logger.error(f"خطا در دریافت پاسخ: {e}")
        return None


if __name__ == "__main__":
    import time

    # ارسال درخواست
    request_id = send_request_to_balance()

    if request_id:
        # کمی صبر می‌کنیم تا سیستم درخواست را پردازش کند
        logger.info("منتظر پردازش درخواست...")
        time.sleep(2)

        # گوش دادن برای پاسخ
        response = listen_for_response(request_id)

        if response:
            # نمایش مختصر نتیجه
            if response['payload'].get('status') == 'success':
                data = response['payload'].get('data', {})
                print(f"\n===== نتیجه جستجو برای 'هوش مصنوعی' =====")
                print(f"عنوان: {data.get('title', 'نامشخص')}")
                print(f"منبع: {data.get('source_url', 'نامشخص')}")
                print(f"تعداد بخش‌ها: {len(data.get('sections', []))}")
                print(f"نمونه محتوا: {data.get('extract', '')[:200]}...")
            else:
                print(f"خطا در پردازش: {response['payload'].get('error_message', 'خطای نامشخص')}")
        else:
            print("پاسخی دریافت نشد.")
    else:
        print("ارسال درخواست ناموفق بود.")

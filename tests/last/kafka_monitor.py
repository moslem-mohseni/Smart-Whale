#!/usr/bin/env python
"""
اسکریپت مانیتورینگ موضوعات Kafka برای مشاهده جریان پیام‌ها در سیستم
"""

import argparse
import json
import logging
import threading
from kafka import KafkaConsumer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError

# تنظیم لاگینگ
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تنظیمات کافکا
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'

# موضوعات پیش‌فرض برای مانیتورینگ
DEFAULT_TOPICS = [
    'smartwhale.models.requests',
    'smartwhale.data.requests',
    'smartwhale.balance.events',
    'smartwhale.balance.metrics'
]


def list_topics():
    """نمایش تمام موضوعات موجود در Kafka"""
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        topics = admin_client.list_topics()
        logger.info("===== موضوعات موجود در Kafka =====")
        for i, topic in enumerate(topics, 1):
            logger.info(f"{i}. {topic}")
        return topics
    except Exception as e:
        logger.error(f"خطا در دریافت لیست موضوعات: {e}")
        return []


def ensure_topics_exist(topics):
    """اطمینان از وجود موضوعات مورد نیاز"""
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
        existing_topics = admin_client.list_topics()

        topics_to_create = [topic for topic in topics if topic not in existing_topics]

        if topics_to_create:
            logger.info(f"ایجاد موضوعات جدید: {topics_to_create}")
            new_topics = [
                NewTopic(name=topic, num_partitions=3, replication_factor=1)
                for topic in topics_to_create
            ]
            admin_client.create_topics(new_topics=new_topics)
            logger.info("موضوعات با موفقیت ایجاد شدند.")
    except TopicAlreadyExistsError:
        logger.warning("بعضی موضوعات از قبل وجود داشتند.")
    except Exception as e:
        logger.error(f"خطا در ایجاد موضوعات: {e}")


def monitor_topic(topic, pretty_print=True):
    """مانیتورینگ یک موضوع خاص در Kafka"""
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id=f"monitor_{topic.replace('.', '_')}",
            value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None
        )

        logger.info(f"شروع مانیتورینگ موضوع: {topic}")

        for message in consumer:
            try:
                if message.value is None:
                    continue

                # نمایش اطلاعات پیام
                logger.info(f"\n===== پیام جدید در {topic} =====")
                logger.info(f"زمان: {message.timestamp}")
                logger.info(f"پارتیشن: {message.partition}, آفست: {message.offset}")

                # نمایش محتوای پیام
                if pretty_print:
                    logger.info("محتوا:")
                    logger.info(json.dumps(message.value, indent=2, ensure_ascii=False))
                else:
                    logger.info(f"محتوا: {message.value}")

                logger.info("=" * 50)
            except Exception as e:
                logger.error(f"خطا در پردازش پیام: {e}")
                continue
    except Exception as e:
        logger.error(f"خطا در مانیتورینگ موضوع {topic}: {e}")


def monitor_multiple_topics(topics, pretty_print=True):
    """مانیتورینگ چندین موضوع به صورت همزمان"""
    # اطمینان از وجود موضوعات
    ensure_topics_exist(topics)

    # ایجاد یک thread جداگانه برای هر موضوع
    threads = []
    for topic in topics:
        thread = threading.Thread(target=monitor_topic, args=(topic, pretty_print))
        thread.daemon = True
        threads.append(thread)
        thread.start()

    # انتظار برای تمام thread ها
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        logger.info("مانیتورینگ متوقف شد.")


if __name__ == "__main__":
    # تنظیم پارامترهای ورودی
    parser = argparse.ArgumentParser(description="مانیتورینگ موضوعات Kafka در سیستم Smart Whale")
    parser.add_argument('--topics', '-t', nargs='+', help='لیست موضوعات برای مانیتورینگ (با فاصله جدا شوند)')
    parser.add_argument('--list', '-l', action='store_true', help='نمایش لیست تمام موضوعات')
    parser.add_argument('--no-pretty', action='store_true', help='غیرفعال کردن نمایش زیبای JSON')

    args = parser.parse_args()

    if args.list:
        list_topics()
    else:
        topics_to_monitor = args.topics if args.topics else DEFAULT_TOPICS
        monitor_multiple_topics(topics_to_monitor, not args.no_pretty)

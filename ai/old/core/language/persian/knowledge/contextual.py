from ..services.redis_service import RedisClient
from ..services.clickhouse_service import ClickHouseClient
from ..services.kafka_service import KafkaProducer, KafkaConsumer


class ContextualKnowledge:
    """
    مدیریت دانش زمینه‌ای و مکالمات قبلی
    """

    def __init__(self):
        # اتصال به سرویس‌های داده
        self.redis = RedisClient()
        self.clickhouse = ClickHouseClient()
        self.kafka_producer = KafkaProducer(topic="context_updates")
        self.kafka_consumer = KafkaConsumer(topic="context_updates")

    def store_context(self, user_id, context_data, storage="redis"):
        """
        ذخیره اطلاعات زمینه‌ای کاربر برای مکالمات بعدی
        :param user_id: شناسه کاربر
        :param context_data: داده‌های زمینه‌ای
        :param storage: نوع پایگاه داده ("redis", "clickhouse")
        """
        key = f"context:{user_id}"
        if storage == "redis":
            self.redis.set(key, context_data)
        elif storage == "clickhouse":
            self.clickhouse.insert("contextual_knowledge", {"user_id": user_id, "data": context_data})
        else:
            raise ValueError("نوع پایگاه‌داده نامعتبر است!")

    def get_context(self, user_id, storage="redis"):
        """
        واکشی اطلاعات زمینه‌ای ذخیره‌شده
        :param user_id: شناسه کاربر
        :param storage: نوع پایگاه داده ("redis", "clickhouse")
        :return: داده‌ی زمینه‌ای
        """
        key = f"context:{user_id}"
        if storage == "redis":
            return self.redis.get(key)
        elif storage == "clickhouse":
            return self.clickhouse.query(f"SELECT data FROM contextual_knowledge WHERE user_id='{user_id}'")
        else:
            raise ValueError("نوع پایگاه‌داده نامعتبر است!")

    def stream_context_updates(self):
        """
        استریم تغییرات و به‌روزرسانی‌های دانش زمینه‌ای از Kafka
        """
        for message in self.kafka_consumer.listen():
            print(f"📌 تغییر جدید در زمینه مکالمات دریافت شد: {message}")


# =========================== TEST ===========================
if __name__ == "__main__":
    context = ContextualKnowledge()

    # ذخیره دانش زمینه‌ای در Redis
    context.store_context("user_123", "کاربر در مورد آب‌وهوا صحبت کرده است.", storage="redis")

    # دریافت داده از Redis
    print(context.get_context("user_123", storage="redis"))

    # استریم تغییرات در Kafka
    context.stream_context_updates()

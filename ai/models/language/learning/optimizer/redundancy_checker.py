import hashlib
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class RedundancyChecker:
    """
    بررسی داده‌های تکراری و حذف پردازش‌های غیرضروری برای بهینه‌سازی مصرف منابع.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای بررسی داده‌های پردازشی
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره هش داده‌های پردازش‌شده در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره هش داده‌های پردازش‌شده.
        """
        query = """
        CREATE TABLE IF NOT EXISTS processed_data_hashes (
            data_hash String,
            model_name String,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (data_hash, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def generate_hash(self, data):
        """
        تولید هش از داده ورودی برای تشخیص تکراری بودن آن.
        """
        return hashlib.sha256(str(data).encode()).hexdigest()

    def is_duplicate(self, model_name, data):
        """
        بررسی اینکه آیا داده قبلاً پردازش شده است یا خیر.
        """
        data_hash = self.generate_hash(data)
        cache_key = f"redundancy:{data_hash}"

        # بررسی در کش Redis
        cached_result = self.redis_cache.get_cache(cache_key)
        if cached_result:
            return True

        # بررسی در ClickHouse
        query = f"""
        SELECT COUNT(*) as count FROM processed_data_hashes 
        WHERE data_hash = '{data_hash}' AND model_name = '{model_name}';
        """
        result = self.clickhouse_client.execute_query(query)

        if result and result[0]["count"] > 0:
            self.redis_cache.set_cache(cache_key, {"exists": True})
            return True

        return False

    def register_processed_data(self, model_name, data):
        """
        ثبت داده پردازش‌شده برای جلوگیری از پردازش‌های تکراری در آینده.
        """
        data_hash = self.generate_hash(data)
        query = f"""
        INSERT INTO processed_data_hashes (data_hash, model_name) 
        VALUES ('{data_hash}', '{model_name}');
        """
        self.clickhouse_client.execute_query(query)

        # ذخیره در کش Redis
        cache_key = f"redundancy:{data_hash}"
        self.redis_cache.set_cache(cache_key, {"exists": True})

        print(f"✅ داده پردازش‌شده با هش {data_hash} ثبت شد.")


# تست عملکرد
if __name__ == "__main__":
    checker = RedundancyChecker()

    model_name = "SampleModel"
    sample_data = {"text": "این یک نمونه داده است.", "metadata": {"lang": "fa"}}

    # بررسی اینکه داده قبلاً پردازش شده یا خیر
    if checker.is_duplicate(model_name, sample_data):
        print("⚠️ این داده قبلاً پردازش شده است و نیازی به پردازش مجدد نیست.")
    else:
        # ثبت داده پردازش‌شده
        checker.register_processed_data(model_name, sample_data)
        print("🚀 داده برای پردازش ثبت شد.")

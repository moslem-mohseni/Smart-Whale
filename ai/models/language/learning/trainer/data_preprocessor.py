import re
import hashlib
from ai.models.language.infrastructure.caching.cache_manager import CacheManager
from ai.models.language.infrastructure.redis_connector import RedisConnector
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB


class DataPreprocessor:
    """
    کلاس DataPreprocessor برای پردازش اولیه داده‌های متنی، حذف نویزها، نرمال‌سازی متن و جلوگیری از پردازش‌های تکراری.
    """

    def __init__(self):
        # استفاده از سرویس‌های infrastructure برای Redis و ClickHouse
        self.redis_client = RedisConnector()
        self.cache_manager = CacheManager()
        self.clickhouse_client = ClickHouseDB()

        # ایجاد جدول در ClickHouse برای ذخیره هش داده‌ها
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره هش‌های داده‌ها و جلوگیری از پردازش‌های غیرضروری.
        """
        query = """
        CREATE TABLE IF NOT EXISTS training_data_hashes (
            hash String,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY hash;
        """
        self.clickhouse_client.execute_query(query)

    def clean_text(self, text: str) -> str:
        """
        پردازش و پاک‌سازی داده‌های متنی برای حذف نویزها، کاراکترهای غیرضروری و نرمال‌سازی.
        """
        text = text.lower().strip()  # تبدیل متن به حروف کوچک و حذف فاصله‌های اضافه
        text = re.sub(r'\s+', ' ', text)  # حذف فاصله‌های غیرضروری
        text = re.sub(r'[^a-zA-Z0-9\u0600-\u06FF\s]', '', text)  # حذف کاراکترهای غیرضروری (فارسی و انگلیسی)
        return text

    def generate_text_hash(self, text: str) -> str:
        """
        تولید هش یکتا از متن برای جلوگیری از پردازش‌های تکراری.
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """
        بررسی اینکه آیا متن قبلاً پردازش شده است یا نه.
        ابتدا هش متن را در Redis جستجو کرده و سپس در ClickHouse بررسی می‌کنیم.
        """
        text_hash = self.generate_text_hash(text)

        # بررسی در Redis (کش سریع)
        if self.cache_manager.get_cached_result(text_hash):
            return True

        # بررسی در ClickHouse
        query = f"SELECT count() FROM training_data_hashes WHERE hash = '{text_hash}'"
        result = self.clickhouse_client.execute_query(query)
        if result[0][0] > 0:
            return True

        return False

    def store_hash(self, text: str):
        """
        ذخیره هش متن در Redis و ClickHouse برای جلوگیری از پردازش مجدد.
        """
        text_hash = self.generate_text_hash(text)

        # ذخیره در Redis (برای کش سریع)
        self.cache_manager.cache_result(text_hash, "1", ttl=86400)  # اعتبار 1 روزه

        # ذخیره در ClickHouse برای بررسی‌های آتی
        query = f"INSERT INTO training_data_hashes (hash) VALUES ('{text_hash}')"
        self.clickhouse_client.execute_query(query)

    def preprocess_text(self, text: str) -> str:
        """
        پردازش کامل متن: حذف نویزها، نرمال‌سازی، بررسی تکراری بودن و در نهایت بازگرداندن متن پردازش‌شده.
        """
        text = self.clean_text(text)  # نرمال‌سازی متن

        if self.is_duplicate(text):  # جلوگیری از پردازش تکراری
            return None

        self.store_hash(text)  # ذخیره هش در کش و پایگاه داده
        return text


# تست عملکرد
if __name__ == "__main__":
    processor = DataPreprocessor()
    sample_text = "این یک تست است!!!  "
    processed_text = processor.preprocess_text(sample_text)

    if processed_text:
        print("متن پردازش‌شده:", processed_text)
    else:
        print("🚀 این متن قبلاً پردازش شده است.")

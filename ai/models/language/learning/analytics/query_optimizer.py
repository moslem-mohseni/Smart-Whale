from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache

class QueryOptimizer:
    """
    بهینه‌سازی کوئری‌های ClickHouse برای کاهش بار پردازشی و افزایش سرعت پردازش داده‌ها.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای تحلیل و بهینه‌سازی کوئری‌ها
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

    def analyze_query_performance(self):
        """
        تحلیل عملکرد کوئری‌ها برای شناسایی کوئری‌های پرهزینه.
        """
        query = """
        SELECT query, query_duration_ms, memory_usage
        FROM system.query_log
        WHERE event_time > now() - INTERVAL 1 DAY
        ORDER BY query_duration_ms DESC
        LIMIT 10;
        """
        result = self.clickhouse_client.execute_query(query)

        return result if result else None

    def optimize_table_structure(self, table_name):
        """
        بررسی و ارائه پیشنهاد برای بهینه‌سازی ساختار جدول در ClickHouse.
        """
        query = f"""
        SELECT table, database, engine, total_rows, total_bytes
        FROM system.tables
        WHERE table = '{table_name}';
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            table_info = result[0]
            optimization_suggestions = []

            # پیشنهاد ایجاد شاخص برای بهینه‌سازی جستجو
            if table_info["engine"] in ["MergeTree", "ReplacingMergeTree"]:
                optimization_suggestions.append("ایجاد ORDER BY برای افزایش سرعت پردازش.")

            return {"table_info": table_info, "suggestions": optimization_suggestions}

        return None

    def cache_frequent_queries(self, query_name, query_result):
        """
        کش کردن نتایج کوئری‌های پرکاربرد برای جلوگیری از اجرای مکرر.
        """
        cache_key = f"query_cache:{query_name}"
        self.redis_cache.set_cache(cache_key, query_result)
        print(f"✅ نتیجه کوئری {query_name} در کش ذخیره شد.")

    def get_cached_query(self, query_name):
        """
        دریافت نتیجه کش‌شده یک کوئری برای کاهش پردازش مجدد.
        """
        cache_key = f"query_cache:{query_name}"
        return self.redis_cache.get_cache(cache_key)

# تست عملکرد
if __name__ == "__main__":
    optimizer = QueryOptimizer()

    # تحلیل عملکرد کوئری‌های ClickHouse
    slow_queries = optimizer.analyze_query_performance()
    print(f"🐢 کوئری‌های کند در سیستم: {slow_queries}")

    # بررسی و پیشنهاد بهینه‌سازی برای یک جدول
    table_optimization = optimizer.optimize_table_structure("model_performance")
    print(f"📊 پیشنهادات بهینه‌سازی جدول: {table_optimization}")

    # کش کردن یک کوئری پرکاربرد
    sample_query_result = [{"accuracy": 0.92, "loss": 0.08}]
    optimizer.cache_frequent_queries("latest_model_accuracy", sample_query_result)

    # دریافت نتیجه کش‌شده کوئری
    cached_query = optimizer.get_cached_query("latest_model_accuracy")
    print(f"⚡ نتیجه کش‌شده کوئری: {cached_query}")

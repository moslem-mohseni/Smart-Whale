from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class ClickHouseQueries:
    """
    اجرای کوئری‌های تحلیلی روی داده‌های یادگیری برای استخراج متریک‌های کلیدی.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای اجرای کوئری‌ها و کش کردن نتایج
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

    def get_learning_progress(self, model_name):
        """
        دریافت روند پیشرفت مدل از نظر دقت و میزان کاهش خطا.
        """
        cache_key = f"learning_progress:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result

        query = f"""
        SELECT timestamp, AVG(performance_score) as avg_performance
        FROM learning_rate_adjustments
        WHERE model_name = '{model_name}'
        GROUP BY timestamp
        ORDER BY timestamp ASC;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            self.redis_cache.set_cache(cache_key, result)
            return result

        return None

    def get_most_effective_parameters(self, model_name):
        """
        شناسایی بهترین تنظیمات پارامترهای یادگیری برای بهبود عملکرد مدل.
        """
        query = f"""
        SELECT param_config, AVG(performance_score) as avg_score
        FROM hyperparameter_tuning
        WHERE model_name = '{model_name}'
        GROUP BY param_config
        ORDER BY avg_score DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            return result[0]

        return None

    def get_memory_usage_trends(self, model_name):
        """
        تحلیل روند مصرف حافظه مدل در طول فرآیند یادگیری.
        """
        query = f"""
        SELECT timestamp, memory_before, memory_after
        FROM memory_optimization
        WHERE model_name = '{model_name}'
        ORDER BY timestamp ASC;
        """
        result = self.clickhouse_client.execute_query(query)

        return result if result else None


# تست عملکرد
if __name__ == "__main__":
    query_runner = ClickHouseQueries()

    model_name = "SampleModel"

    # دریافت روند یادگیری
    learning_progress = query_runner.get_learning_progress(model_name)
    print(f"📊 روند یادگیری: {learning_progress}")

    # دریافت بهترین پارامترهای بهینه‌سازی شده
    best_params = query_runner.get_most_effective_parameters(model_name)
    print(f"🎯 بهترین تنظیمات پارامترها: {best_params}")

    # تحلیل روند مصرف حافظه
    memory_trends = query_runner.get_memory_usage_trends(model_name)
    print(f"📉 روند مصرف حافظه: {memory_trends}")

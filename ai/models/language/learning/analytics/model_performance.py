from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class ModelPerformance:
    """
    تحلیل عملکرد مدل‌ها در طول زمان و بررسی روند پیشرفت آنها.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای تحلیل داده‌های عملکردی مدل‌ها
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

    def get_performance_trend(self, model_name):
        """
        دریافت روند تغییرات دقت و میزان خطا در نسخه‌های مختلف مدل.
        """
        cache_key = f"performance_trend:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result

        query = f"""
        SELECT timestamp, AVG(accuracy) as avg_accuracy, AVG(loss) as avg_loss
        FROM model_accuracy
        WHERE model_name = '{model_name}'
        GROUP BY timestamp
        ORDER BY timestamp ASC;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            self.redis_cache.set_cache(cache_key, result)
            return result

        return None

    def compare_model_versions(self, model_name):
        """
        مقایسه نسخه‌های مختلف مدل از نظر دقت، میزان خطا و عملکرد پردازشی.
        """
        query = f"""
        SELECT version, AVG(accuracy) as avg_accuracy, AVG(loss) as avg_loss, AVG(execution_time) as avg_execution_time
        FROM model_performance
        WHERE model_name = '{model_name}'
        GROUP BY version
        ORDER BY avg_accuracy DESC;
        """
        result = self.clickhouse_client.execute_query(query)

        return result if result else None

    def get_training_efficiency(self, model_name):
        """
        بررسی میزان کارایی مدل در فرآیند آموزش و مقایسه آن با نسخه‌های قبلی.
        """
        query = f"""
        SELECT AVG(training_time) as avg_training_time, AVG(memory_usage) as avg_memory_usage
        FROM training_metrics
        WHERE model_name = '{model_name}';
        """
        result = self.clickhouse_client.execute_query(query)

        return result[0] if result else None


# تست عملکرد
if __name__ == "__main__":
    performance_analyzer = ModelPerformance()

    model_name = "SampleModel"

    # دریافت روند عملکرد مدل
    performance_trend = performance_analyzer.get_performance_trend(model_name)
    print(f"📊 روند عملکرد مدل: {performance_trend}")

    # مقایسه نسخه‌های مختلف مدل
    version_comparison = performance_analyzer.compare_model_versions(model_name)
    print(f"📈 مقایسه نسخه‌های مدل: {version_comparison}")

    # بررسی میزان کارایی آموزش مدل
    efficiency = performance_analyzer.get_training_efficiency(model_name)
    print(f"⚡ کارایی آموزش مدل: {efficiency}")

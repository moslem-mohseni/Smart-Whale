from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class EvaluationReports:
    """
    تولید گزارش‌های تحلیلی از فرآیند یادگیری مدل‌ها برای ارزیابی عملکرد.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای پردازش گزارش‌ها
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

    def generate_performance_summary(self, model_name):
        """
        ایجاد خلاصه‌ای از عملکرد مدل شامل دقت و میزان بهبود یادگیری.
        """
        cache_key = f"performance_summary:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result

        query = f"""
        SELECT AVG(accuracy) as avg_accuracy, AVG(loss) as avg_loss, COUNT(*) as total_updates
        FROM model_accuracy
        WHERE model_name = '{model_name}';
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            self.redis_cache.set_cache(cache_key, result[0])
            return result[0]

        return None

    def generate_learning_trends(self, model_name):
        """
        ایجاد گزارش روندهای یادگیری و تحلیل تأثیر تنظیمات بر کارایی مدل.
        """
        query = f"""
        SELECT timestamp, accuracy, loss
        FROM model_accuracy
        WHERE model_name = '{model_name}'
        ORDER BY timestamp ASC
        LIMIT 100;
        """
        result = self.clickhouse_client.execute_query(query)

        return result if result else None

    def generate_optimization_summary(self, model_name):
        """
        ارائه خلاصه‌ای از تنظیمات بهینه‌شده و تأثیر آن‌ها بر عملکرد مدل.
        """
        query = f"""
        SELECT param_config, AVG(performance_score) as avg_score
        FROM hyperparameter_tuning
        WHERE model_name = '{model_name}'
        GROUP BY param_config
        ORDER BY avg_score DESC
        LIMIT 5;
        """
        result = self.clickhouse_client.execute_query(query)

        return result if result else None


# تست عملکرد
if __name__ == "__main__":
    report_generator = EvaluationReports()

    model_name = "SampleModel"

    # دریافت خلاصه عملکرد مدل
    performance_summary = report_generator.generate_performance_summary(model_name)
    print(f"📊 خلاصه عملکرد مدل: {performance_summary}")

    # دریافت روندهای یادگیری
    learning_trends = report_generator.generate_learning_trends(model_name)
    print(f"📈 روند یادگیری مدل: {learning_trends}")

    # دریافت خلاصه بهینه‌سازی مدل
    optimization_summary = report_generator.generate_optimization_summary(model_name)
    print(f"🔧 خلاصه بهینه‌سازی مدل: {optimization_summary}")

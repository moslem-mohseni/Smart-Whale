import numpy as np
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class ClickHouseAnalyzer:
    """
    تحلیل داده‌های آموزشی در ClickHouse برای یافتن بهترین تنظیمات پارامترهای مدل.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای اجرای کوئری‌ها و کش کردن نتایج
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

    def get_best_learning_rate(self, model_name):
        """
        تحلیل داده‌های آموزشی برای پیشنهاد بهترین نرخ یادگیری.
        """
        cache_key = f"best_lr:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result["best_learning_rate"]

        query = f"""
        SELECT new_learning_rate, AVG(performance_score) as avg_score
        FROM learning_rate_adjustments
        WHERE model_name = '{model_name}'
        GROUP BY new_learning_rate
        ORDER BY avg_score DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            best_learning_rate = result[0]["new_learning_rate"]
            self.redis_cache.set_cache(cache_key, {"best_learning_rate": best_learning_rate})
            return best_learning_rate

        return None

    def analyze_training_patterns(self, model_name):
        """
        بررسی روند آموزشی و تحلیل الگوهای بهینه برای مدل.
        """
        query = f"""
        SELECT timestamp, new_learning_rate, previous_loss, improvement_ratio
        FROM learning_rate_adjustments
        WHERE model_name = '{model_name}'
        ORDER BY timestamp DESC
        LIMIT 100;
        """
        results = self.clickhouse_client.execute_query(query)

        if results:
            timestamps = [row["timestamp"] for row in results]
            learning_rates = np.array([row["new_learning_rate"] for row in results])
            improvement_ratios = np.array([row["improvement_ratio"] for row in results])

            trend = np.polyfit(range(len(improvement_ratios)), improvement_ratios, 1)
            trend_slope = trend[0]

            return {
                "learning_rate_trend": trend_slope,
                "suggested_learning_rate": np.median(learning_rates) if trend_slope > 0 else np.mean(learning_rates),
            }
        return None

    def get_training_performance_summary(self, model_name):
        """
        خلاصه‌ای از عملکرد آموزشی مدل در طول زمان.
        """
        query = f"""
        SELECT COUNT(*) as total_updates, 
               AVG(previous_loss) as avg_loss, 
               AVG(improvement_ratio) as avg_improvement
        FROM learning_rate_adjustments
        WHERE model_name = '{model_name}';
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            return result[0]

        return None


# تست عملکرد
if __name__ == "__main__":
    analyzer = ClickHouseAnalyzer()

    model_name = "SampleModel"

    best_lr = analyzer.get_best_learning_rate(model_name)
    print(f"🎯 بهترین نرخ یادگیری پیشنهادی برای {model_name}: {best_lr}")

    trend_analysis = analyzer.analyze_training_patterns(model_name)
    print(f"📊 تحلیل روند یادگیری: {trend_analysis}")

    summary = analyzer.get_training_performance_summary(model_name)
    print(f"📈 خلاصه عملکرد آموزشی: {summary}")

import numpy as np
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class TrainingTrends:
    """
    تحلیل روندهای آموزشی برای تنظیم نرخ یادگیری و بهینه‌سازی فرآیند یادگیری مدل.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای تحلیل داده‌های آموزشی
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

    def get_accuracy_trend(self, model_name):
        """
        بررسی روند تغییرات دقت مدل در طول زمان.
        """
        cache_key = f"accuracy_trend:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result

        query = f"""
        SELECT timestamp, accuracy
        FROM model_accuracy
        WHERE model_name = '{model_name}'
        ORDER BY timestamp ASC;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            self.redis_cache.set_cache(cache_key, result)
            return result

        return None

    def get_loss_trend(self, model_name):
        """
        بررسی روند تغییرات میزان خطای مدل در طول زمان.
        """
        query = f"""
        SELECT timestamp, loss
        FROM model_accuracy
        WHERE model_name = '{model_name}'
        ORDER BY timestamp ASC;
        """
        result = self.clickhouse_client.execute_query(query)

        return result if result else None

    def suggest_learning_rate_adjustment(self, model_name):
        """
        ارائه پیشنهاد برای تنظیم نرخ یادگیری بر اساس روند تغییرات دقت و میزان خطا.
        """
        accuracy_trend = self.get_accuracy_trend(model_name)
        loss_trend = self.get_loss_trend(model_name)

        if not accuracy_trend or not loss_trend:
            return {"suggested_learning_rate": None, "reason": "عدم وجود داده‌های کافی"}

        accuracy_deltas = np.diff([row["accuracy"] for row in accuracy_trend])
        loss_deltas = np.diff([row["loss"] for row in loss_trend])

        if np.mean(accuracy_deltas) > 0 and np.mean(loss_deltas) < 0:
            return {"suggested_learning_rate": "افزایش", "reason": "روند بهبود مشاهده شده"}
        elif np.mean(accuracy_deltas) < 0 and np.mean(loss_deltas) > 0:
            return {"suggested_learning_rate": "کاهش", "reason": "روند افت عملکرد"}
        else:
            return {"suggested_learning_rate": "ثابت", "reason": "تغییرات جزئی"}


# تست عملکرد
if __name__ == "__main__":
    trends_analyzer = TrainingTrends()

    model_name = "SampleModel"

    # بررسی روند تغییرات دقت
    accuracy_trend = trends_analyzer.get_accuracy_trend(model_name)
    print(f"📈 روند دقت مدل: {accuracy_trend}")

    # بررسی روند تغییرات خطا
    loss_trend = trends_analyzer.get_loss_trend(model_name)
    print(f"📉 روند خطای مدل: {loss_trend}")

    # ارائه پیشنهاد برای تنظیم نرخ یادگیری
    suggested_lr = trends_analyzer.suggest_learning_rate_adjustment(model_name)
    print(f"🎯 پیشنهاد برای نرخ یادگیری: {suggested_lr}")

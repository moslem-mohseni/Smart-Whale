import numpy as np
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache


class LearningStrategy:
    """
    مدیریت استراتژی‌های یادگیری برای بهینه‌سازی فرآیند آموزشی مدل‌ها.
    """

    def __init__(self):
        # اتصال به ClickHouse و Redis برای مدیریت استراتژی‌های یادگیری
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره استراتژی‌های یادگیری در ClickHouse
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره استراتژی‌های یادگیری.
        """
        query = """
        CREATE TABLE IF NOT EXISTS learning_strategies (
            model_name String,
            version String,
            strategy_type String,
            learning_rate Float32,
            performance_score Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def select_best_strategy(self, model_name):
        """
        تحلیل داده‌های یادگیری و انتخاب بهترین استراتژی آموزشی.
        """
        query = f"""
        SELECT strategy_type, AVG(performance_score) as avg_score
        FROM learning_strategies
        WHERE model_name = '{model_name}'
        GROUP BY strategy_type
        ORDER BY avg_score DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            best_strategy = result[0]["strategy_type"]
            return best_strategy

        return "Supervised"  # مقدار پیش‌فرض در صورت نبود داده

    def update_strategy(self, model_name, strategy_type, learning_rate, performance_score):
        """
        ذخیره استراتژی جدید یادگیری در ClickHouse و به‌روزرسانی کش.
        """
        query = f"""
        INSERT INTO learning_strategies (model_name, version, strategy_type, learning_rate, performance_score)
        VALUES ('{model_name}', '1.0', '{strategy_type}', {learning_rate}, {performance_score});
        """
        self.clickhouse_client.execute_query(query)

        cache_key = f"learning_strategy:{model_name}"
        self.redis_cache.set_cache(cache_key, {"strategy": strategy_type, "learning_rate": learning_rate})

        print(f"✅ استراتژی جدید یادگیری برای مدل {model_name}: {strategy_type} با نرخ یادگیری {learning_rate} ثبت شد.")

    def get_current_strategy(self, model_name):
        """
        دریافت آخرین استراتژی ذخیره‌شده برای مدل از کش یا ClickHouse.
        """
        cache_key = f"learning_strategy:{model_name}"
        cached_result = self.redis_cache.get_cache(cache_key)

        if cached_result:
            return cached_result

        query = f"""
        SELECT strategy_type, learning_rate
        FROM learning_strategies
        WHERE model_name = '{model_name}'
        ORDER BY timestamp DESC
        LIMIT 1;
        """
        result = self.clickhouse_client.execute_query(query)

        if result:
            return result[0]

        return None


# تست عملکرد
if __name__ == "__main__":
    strategy_manager = LearningStrategy()

    model_name = "SampleModel"

    # به‌روزرسانی استراتژی جدید
    strategy_manager.update_strategy(model_name, strategy_type="Reinforcement", learning_rate=0.01,
                                     performance_score=0.85)

    # دریافت استراتژی فعلی مدل
    current_strategy = strategy_manager.get_current_strategy(model_name)
    print(f"📊 استراتژی فعلی مدل: {current_strategy}")

    # انتخاب بهترین استراتژی بر اساس داده‌های آموزشی
    best_strategy = strategy_manager.select_best_strategy(model_name)
    print(f"🎯 بهترین استراتژی پیشنهادی برای {model_name}: {best_strategy}")

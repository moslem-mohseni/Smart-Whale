import random
import torch
from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB
from ai.models.language.infrastructure.redis.redis_adapter import RedisCache

class HyperparameterTuner:
    """
    کلاس بهینه‌سازی پارامترهای مدل با استفاده از الگوریتم‌های جستجو.
    """

    def __init__(self, model, param_space, search_method="random", trials=10):
        self.model = model
        self.param_space = param_space
        self.search_method = search_method  # روش جستجو (random/grid/bayesian)
        self.trials = trials  # تعداد تکرار تست پارامترها

        # اتصال به ClickHouse و Redis برای ذخیره و کش کردن نتایج
        self.clickhouse_client = ClickHouseDB()
        self.redis_cache = RedisCache()

        # ایجاد جدول برای ذخیره نتایج بهینه‌سازی پارامترها
        self.create_clickhouse_table()

    def create_clickhouse_table(self):
        """
        ایجاد جدول در ClickHouse برای ذخیره نتایج تنظیمات هایپرپارامترها.
        """
        query = """
        CREATE TABLE IF NOT EXISTS hyperparameter_tuning (
            model_name String,
            version String,
            param_config String,
            performance_score Float32,
            timestamp DateTime DEFAULT now()
        ) ENGINE = MergeTree()
        ORDER BY (model_name, timestamp);
        """
        self.clickhouse_client.execute_query(query)

    def random_search(self):
        """
        اجرای جستجوی تصادفی روی فضای پارامترها.
        """
        best_config = None
        best_score = float("-inf")

        for _ in range(self.trials):
            config = {k: random.choice(v) for k, v in self.param_space.items()}
            score = self.evaluate_config(config)

            if score > best_score:
                best_score = score
                best_config = config

        return best_config, best_score

    def evaluate_config(self, config):
        """
        ارزیابی عملکرد یک تنظیم پارامتر خاص.
        """
        config_key = str(config)

        # بررسی کش برای جلوگیری از اجرای تست‌های تکراری
        cached_result = self.redis_cache.get_cache(f"hyperparam:{config_key}")
        if cached_result:
            return cached_result["score"]

        # اجرای تست مدل با پارامترهای مشخص‌شده
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"])
        loss_function = torch.nn.CrossEntropyLoss()
        dummy_input = torch.randn(1, 10)
        dummy_output = torch.tensor([1])

        optimizer.zero_grad()
        output = self.model(dummy_input)
        loss = loss_function(output, dummy_output)
        loss.backward()
        optimizer.step()

        score = -loss.item()  # امتیاز عملکرد بر اساس کاهش خطا

        # ذخیره نتیجه در ClickHouse و کش کردن مقدار آن در Redis
        self.log_result(config, score)

        return score

    def log_result(self, config, score):
        """
        ذخیره نتایج تنظیمات پارامتر در ClickHouse و Redis.
        """
        config_str = str(config)
        query = f"""
        INSERT INTO hyperparameter_tuning (model_name, version, param_config, performance_score)
        VALUES ('{self.model.__class__.__name__}', '1.0', '{config_str}', {score});
        """
        self.clickhouse_client.execute_query(query)

        # ذخیره نتیجه در Redis برای جلوگیری از پردازش تکراری
        self.redis_cache.set_cache(f"hyperparam:{config_str}", {"score": score})

        print(f"✅ نتیجه‌ی تنظیم پارامتر ذخیره شد: {config} -> Score: {score}")

# تست عملکرد
if __name__ == "__main__":
    import torch.nn as nn

    class SampleModel(nn.Module):
        def __init__(self):
            super(SampleModel, self).__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)

    # تعریف فضای پارامترها برای بهینه‌سازی
    param_space = {
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "optimizer": ["adam", "sgd"]
    }

    model = SampleModel()
    tuner = HyperparameterTuner(model, param_space, search_method="random", trials=5)

    # اجرای جستجوی تصادفی
    best_config, best_score = tuner.random_search()
    print(f"🎯 بهترین تنظیمات: {best_config} با امتیاز: {best_score}")

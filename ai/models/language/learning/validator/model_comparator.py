from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

class ModelComparator:
    """
    مقایسه نسخه‌های مختلف مدل‌ها برای انتخاب بهترین نسخه از لحاظ دقت، عملکرد و کارایی پردازشی.
    """

    def __init__(self):
        self.clickhouse_client = ClickHouseDB()

    def get_model_versions(self, model_name):
        """
        دریافت لیست نسخه‌های موجود از یک مدل در ClickHouse.
        """
        query = f"""
        SELECT DISTINCT version
        FROM model_accuracy
        WHERE model_name = '{model_name}'
        ORDER BY version DESC;
        """
        result = self.clickhouse_client.execute_query(query)
        return [row[0] for row in result]

    def compare_models(self, model_name):
        """
        مقایسه نسخه‌های مختلف یک مدل بر اساس دقت و عملکرد پردازشی.
        """
        query = f"""
        SELECT a.version, AVG(a.accuracy) as avg_accuracy, 
               AVG(p.execution_time) as avg_exec_time, 
               AVG(p.gpu_usage) as avg_gpu, 
               AVG(p.cpu_usage) as avg_cpu
        FROM model_accuracy a
        JOIN model_performance p
        ON a.model_name = p.model_name AND a.version = p.version
        WHERE a.model_name = '{model_name}'
        GROUP BY a.version
        ORDER BY avg_accuracy DESC, avg_exec_time ASC;
        """
        result = self.clickhouse_client.execute_query(query)

        model_comparisons = []
        for row in result:
            model_comparisons.append({
                "version": row[0],
                "avg_accuracy": row[1],
                "avg_exec_time": row[2],
                "avg_gpu_usage": row[3],
                "avg_cpu_usage": row[4]
            })

        return model_comparisons

    def get_best_model_version(self, model_name):
        """
        دریافت بهترین نسخه مدل بر اساس بالاترین دقت و کمترین هزینه پردازشی.
        """
        comparisons = self.compare_models(model_name)
        if comparisons:
            best_version = comparisons[0]["version"]
            print(f"✅ بهترین نسخه برای مدل {model_name}: {best_version}")
            return best_version
        else:
            print(f"🚨 هیچ اطلاعاتی برای مدل {model_name} یافت نشد.")
            return None

# تست عملکرد
if __name__ == "__main__":
    comparator = ModelComparator()

    model_name = "TestModel"

    # دریافت نسخه‌های مدل
    versions = comparator.get_model_versions(model_name)
    print(f"📌 نسخه‌های موجود برای مدل {model_name}: {versions}")

    # مقایسه نسخه‌های مدل
    comparisons = comparator.compare_models(model_name)
    print("📊 مقایسه مدل‌ها:", comparisons)

    # دریافت بهترین نسخه‌ی مدل
    best_version = comparator.get_best_model_version(model_name)

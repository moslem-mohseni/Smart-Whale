from typing import Dict, List


class ModelAggregator:
    """
    تجمیع‌کننده به‌روزرسانی‌های مدل‌های یادگیری فدراسیونی برای تولید یک مدل بهینه
    """

    def __init__(self):
        self.aggregated_weights: List[float] = []  # وزن‌های مدل تجمیع‌شده

    def aggregate_updates(self, model_updates: Dict[str, List[float]]) -> List[float]:
        """
        تجمیع به‌روزرسانی‌های مدل‌ها و ایجاد وزن‌های جدید
        :param model_updates: دیکشنری شامل نام مدل‌ها و لیست وزن‌های به‌روزرسانی‌شده آن‌ها
        :return: وزن‌های جدید مدل پس از تجمیع
        """
        if not model_updates:
            return []

        num_models = len(model_updates)
        weight_length = len(next(iter(model_updates.values())))
        aggregated_weights = [0.0] * weight_length

        for model, weights in model_updates.items():
            for i in range(weight_length):
                aggregated_weights[i] += weights[i] / num_models

        self.aggregated_weights = aggregated_weights
        return aggregated_weights

    def get_aggregated_weights(self) -> List[float]:
        """
        دریافت وزن‌های تجمیع‌شده آخرین مدل فدراسیونی
        :return: لیست وزن‌های نهایی مدل
        """
        return self.aggregated_weights


# نمونه استفاده از ModelAggregator برای تست
if __name__ == "__main__":
    aggregator = ModelAggregator()
    model_updates = {
        "model_a": [0.1, 0.2, 0.3],
        "model_b": [0.4, 0.5, 0.6],
        "model_c": [0.7, 0.8, 0.9]
    }

    aggregated = aggregator.aggregate_updates(model_updates)
    print(f"Aggregated Weights: {aggregated}")
    print(f"Retrieved Aggregated Weights: {aggregator.get_aggregated_weights()}")

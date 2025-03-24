from typing import Dict, List
import numpy as np


class ModelAggregator:
    """
    ماژول تجمیع و ترکیب وزن‌های مدل‌های فدراسیونی برای ایجاد یک مدل مرکزی بهینه.
    """

    def __init__(self, aggregation_method: str = "weighted_average"):
        """
        مقداردهی اولیه تجمیع مدل‌ها.
        :param aggregation_method: روش تجمیع (`weighted_average` یا `adaptive`).
        """
        if aggregation_method not in ["weighted_average", "adaptive"]:
            raise ValueError("روش تجمیع نامعتبر است. گزینه‌های معتبر: 'weighted_average' یا 'adaptive'.")

        self.aggregation_method = aggregation_method

    def aggregate_updates(self, model_updates: Dict[str, List[float]]) -> List[float]:
        """
        تجمیع به‌روزرسانی‌های ارسال‌شده از مدل‌های فدراسیونی.
        :param model_updates: دیکشنری شامل به‌روزرسانی وزن‌های مدل‌ها.
        :return: لیست وزن‌های تجمیع‌شده‌ی مدل.
        """
        if not model_updates:
            raise ValueError("هیچ به‌روزرسانی‌ای برای تجمیع وجود ندارد.")

        model_weights = list(model_updates.values())

        if self.aggregation_method == "weighted_average":
            return self._weighted_average_aggregation(model_weights)

        elif self.aggregation_method == "adaptive":
            return self._adaptive_aggregation(model_weights)

    def _weighted_average_aggregation(self, model_weights: List[List[float]]) -> List[float]:
        """
        تجمیع وزن‌ها با استفاده از میانگین وزنی.
        :param model_weights: لیستی از وزن‌های مدل‌ها.
        :return: وزن‌های ترکیبی مدل.
        """
        weights_array = np.array(model_weights)
        return np.mean(weights_array, axis=0).tolist()

    def _adaptive_aggregation(self, model_weights: List[List[float]]) -> List[float]:
        """
        تجمیع وزن‌ها با استفاده از روش ترکیبی تطبیقی (Adaptive Aggregation).
        :param model_weights: لیستی از وزن‌های مدل‌ها.
        :return: وزن‌های ترکیبی مدل.
        """
        weights_array = np.array(model_weights)

        # اعمال وزن تطبیقی بر اساس تنوع مدل‌ها (مثلاً وزن‌دهی به مدل‌هایی که تغییرات بیشتری دارند)
        variance = np.var(weights_array, axis=0)
        adaptive_weights = np.exp(-variance)  # وزن‌دهی معکوس با میزان تغییرات مدل‌ها

        weighted_sum = np.sum(weights_array * adaptive_weights, axis=0)
        normalization_factor = np.sum(adaptive_weights)

        return (weighted_sum / normalization_factor).tolist()

    def set_aggregation_method(self, method: str):
        """
        تنظیم روش تجمیع مدل‌ها.
        :param method: روش تجمیع (`weighted_average` یا `adaptive`).
        """
        if method not in ["weighted_average", "adaptive"]:
            raise ValueError("روش تجمیع نامعتبر است. گزینه‌های معتبر: 'weighted_average' یا 'adaptive'.")

        self.aggregation_method = method

from typing import List, Dict
import numpy as np
from .model_aggregator import ModelAggregator

class LearningOptimizer:
    """
    ماژول بهینه‌سازی فرآیند یادگیری فدراسیونی برای بهبود عملکرد مدل‌ها.
    """

    def __init__(self, initial_learning_rate: float = 0.01, min_learning_rate: float = 0.0001):
        """
        مقداردهی اولیه بهینه‌ساز یادگیری فدراسیونی.
        :param initial_learning_rate: مقدار اولیه نرخ یادگیری.
        :param min_learning_rate: حداقل مقدار نرخ یادگیری که سیستم می‌تواند تنظیم کند.
        """
        self.learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.model_aggregator = ModelAggregator()

    def adjust_learning_rate(self, loss_history: List[float]) -> float:
        """
        تنظیم نرخ یادگیری بر اساس روند کاهش خطا (Loss History).
        :param loss_history: لیست خطای مدل در مراحل قبلی یادگیری.
        :return: نرخ یادگیری جدید بهینه‌شده.
        """
        if len(loss_history) < 2:
            return self.learning_rate  # تغییر نرخ یادگیری در صورت عدم وجود داده‌ی کافی

        recent_loss = loss_history[-1]
        previous_loss = loss_history[-2]

        # کاهش نرخ یادگیری در صورت کاهش کند خطا
        if recent_loss >= previous_loss:
            self.learning_rate *= 0.9  # کاهش ۱۰٪ نرخ یادگیری
        else:
            self.learning_rate *= 1.02  # افزایش ۲٪ در صورت بهبود عملکرد

        # محدود کردن مقدار نرخ یادگیری
        self.learning_rate = max(self.min_learning_rate, self.learning_rate)
        return self.learning_rate

    def select_best_aggregation_method(self, performance_data: Dict[str, List[float]]) -> str:
        """
        انتخاب بهترین استراتژی تجمیع مدل‌ها بر اساس داده‌های عملکردی.
        :param performance_data: دیکشنری شامل متریک‌های عملکرد مدل‌ها.
        :return: روش تجمیع بهینه انتخاب‌شده.
        """
        model_variances = np.var(list(performance_data.values()), axis=0)

        if np.mean(model_variances) > 0.05:
            return "adaptive"
        return "weighted_average"

    def optimize_model_updates(self, model_updates: Dict[str, List[float]]) -> List[float]:
        """
        بهینه‌سازی داده‌های به‌روزرسانی مدل‌ها برای ایجاد وزن‌های جدید.
        :param model_updates: دیکشنری شامل به‌روزرسانی‌های مدل‌های فدراسیونی.
        :return: وزن‌های بهینه‌شده‌ی مدل نهایی.
        """
        best_method = self.select_best_aggregation_method(model_updates)
        self.model_aggregator.set_aggregation_method(best_method)
        return self.model_aggregator.aggregate_updates(model_updates)

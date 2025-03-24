from typing import List


class LearningOptimizer:
    """
    بهینه‌سازی فرآیند یادگیری فدراسیونی برای افزایش دقت و کاهش پیچیدگی محاسباتی
    """

    def __init__(self):
        self.learning_rates: List[float] = []  # نرخ‌های یادگیری ذخیره‌شده برای تجزیه و تحلیل

    def adjust_learning_rate(self, previous_losses: List[float]) -> float:
        """
        تنظیم نرخ یادگیری بر اساس روند تغییرات خطا
        :param previous_losses: لیستی از مقادیر خطای قبلی
        :return: مقدار بهینه‌شده نرخ یادگیری
        """
        if not previous_losses or len(previous_losses) < 2:
            return 0.01  # مقدار پیش‌فرض

        loss_trend = previous_losses[-1] - previous_losses[-2]
        learning_rate = 0.01

        if loss_trend > 0:
            learning_rate *= 0.9  # کاهش نرخ یادگیری در صورت افزایش خطا
        elif loss_trend < 0:
            learning_rate *= 1.1  # افزایش نرخ یادگیری در صورت کاهش خطا

        self.learning_rates.append(learning_rate)
        return learning_rate

    def get_learning_rates(self) -> List[float]:
        """
        دریافت لیست نرخ‌های یادگیری محاسبه‌شده در طول زمان
        :return: لیست نرخ‌های یادگیری
        """
        return self.learning_rates


# نمونه استفاده از LearningOptimizer برای تست
if __name__ == "__main__":
    optimizer = LearningOptimizer()
    loss_history = [0.5, 0.4, 0.42, 0.38, 0.35]
    adjusted_rate = optimizer.adjust_learning_rate(loss_history)
    print(f"Adjusted Learning Rate: {adjusted_rate}")
    print(f"Learning Rate History: {optimizer.get_learning_rates()}")

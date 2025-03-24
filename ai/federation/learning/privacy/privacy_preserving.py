from typing import List


class PrivacyPreserving:
    """
    مدیریت یادگیری فدراسیونی با حفظ حریم خصوصی داده‌ها
    """

    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level  # سطح نویز برای Differential Privacy

    def apply_differential_privacy(self, model_weights: List[float]) -> List[float]:
        """
        اعمال مکانیزم حفظ حریم خصوصی بر روی وزن‌های مدل با اضافه کردن نویز تصادفی
        :param model_weights: لیست وزن‌های مدل
        :return: لیست وزن‌های تغییر یافته با حفظ حریم خصوصی
        """
        import random
        return [w + random.uniform(-self.noise_level, self.noise_level) for w in model_weights]

    def set_noise_level(self, new_noise_level: float) -> None:
        """
        تنظیم سطح نویز برای کنترل سطح حفظ حریم خصوصی
        :param new_noise_level: مقدار جدید نویز
        """
        self.noise_level = new_noise_level


# نمونه استفاده از PrivacyPreserving برای تست
if __name__ == "__main__":
    privacy = PrivacyPreserving(noise_level=0.05)
    model_weights = [0.1, 0.2, 0.3, 0.4]
    noisy_weights = privacy.apply_differential_privacy(model_weights)
    print(f"Original Weights: {model_weights}")
    print(f"Noisy Weights: {noisy_weights}")


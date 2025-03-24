import numpy as np
from typing import Any, Dict

class PatternDetector:
    """
    ماژول شناسایی الگوهای برداری در داده‌های کوانتومی برای بهینه‌سازی و فشرده‌سازی.
    """

    @staticmethod
    def detect_redundant_patterns(quantum_vector: np.ndarray, threshold: float = 0.9) -> Dict[str, Any]:
        """
        شناسایی الگوهای پرتکرار در بردار کوانتومی.
        :param quantum_vector: بردار کوانتومی که باید تحلیل شود.
        :param threshold: آستانه‌ی شباهت برای حذف الگوهای تکراری.
        :return: دیکشنری شامل اطلاعات الگوهای شناسایی‌شده.
        """
        if not isinstance(quantum_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        # اجرای تبدیل فوریه برای تحلیل الگوهای برداری
        frequency_components = np.fft.fft(quantum_vector)
        magnitude_spectrum = np.abs(frequency_components)

        # شناسایی فرکانس‌های اصلی که بیشترین سهم را دارند
        dominant_frequencies = np.where(magnitude_spectrum > threshold * np.max(magnitude_spectrum))[0]

        return {
            "dominant_frequencies": dominant_frequencies.tolist(),
            "spectrum": magnitude_spectrum.tolist()
        }

    @staticmethod
    def remove_noise(quantum_vector: np.ndarray, filter_strength: float = 0.2) -> np.ndarray:
        """
        حذف نویز از بردار کوانتومی با استفاده از فیلتر پایین‌گذر (Low-Pass Filtering).
        :param quantum_vector: بردار کوانتومی که باید فیلتر شود.
        :param filter_strength: درصدی از فرکانس‌های پایین که حفظ می‌شوند.
        :return: بردار کوانتومی پس از حذف نویز.
        """
        if not isinstance(quantum_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        if filter_strength <= 0 or filter_strength > 1:
            raise ValueError("filter_strength باید بین 0 و 1 باشد.")

        # اجرای تبدیل فوریه برای تحلیل فرکانس‌ها
        frequency_components = np.fft.fft(quantum_vector)

        # اعمال فیلتر پایین‌گذر
        num_components_to_keep = int(len(frequency_components) * filter_strength)
        filtered_components = np.zeros_like(frequency_components)
        filtered_components[:num_components_to_keep] = frequency_components[:num_components_to_keep]

        # اجرای تبدیل معکوس فوریه برای بازسازی داده
        filtered_vector = np.fft.ifft(filtered_components).real

        return filtered_vector

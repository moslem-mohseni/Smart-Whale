import numpy as np
from typing import Any

class StateEncoder:
    """
    ماژول کدگذاری بردارهای کوانتومی برای بهینه‌سازی و فشرده‌سازی.
    """

    @staticmethod
    def encode_amplitudes(quantum_vector: np.ndarray) -> np.ndarray:
        """
        کدگذاری دامنه‌ی بردارهای کوانتومی برای کاهش نویز و افزایش دقت.
        :param quantum_vector: بردار کوانتومی که باید کدگذاری شود.
        :return: بردار کدگذاری‌شده با دامنه‌های بهینه.
        """
        if not isinstance(quantum_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        # اعمال نرمال‌سازی L2 برای حفظ دامنه‌های بردار کوانتومی
        norm = np.linalg.norm(quantum_vector)
        if norm == 0:
            return quantum_vector  # جلوگیری از تقسیم بر صفر

        encoded_vector = quantum_vector / norm

        return encoded_vector

    @staticmethod
    def optimize_representation(encoded_vector: np.ndarray, compression_factor: float = 0.8) -> np.ndarray:
        """
        بهینه‌سازی نمایش بردارهای کوانتومی با حذف اطلاعات کم‌اهمیت.
        :param encoded_vector: بردار کوانتومی که قبلاً کدگذاری شده است.
        :param compression_factor: فاکتور فشرده‌سازی بین 0 و 1.
        :return: بردار بهینه‌شده.
        """
        if not isinstance(encoded_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        if compression_factor <= 0 or compression_factor > 1:
            raise ValueError("compression_factor باید بین 0 و 1 باشد.")

        # حذف مقادیر کوچک‌تر از یک آستانه برای کاهش مصرف حافظه
        threshold = np.percentile(np.abs(encoded_vector), (1 - compression_factor) * 100)
        optimized_vector = np.where(np.abs(encoded_vector) >= threshold, encoded_vector, 0)

        return optimized_vector

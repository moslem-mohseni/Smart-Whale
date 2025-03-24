import numpy as np
from typing import Any

class VectorTransformer:
    """
    ماژول تبدیل بردارها به فضای کوانتومی با استفاده از تبدیل فوریه کوانتومی (QFT).
    """

    @staticmethod
    def transform_to_quantum(vector: np.ndarray) -> np.ndarray:
        """
        تبدیل بردار به فضای کوانتومی با استفاده از تبدیل فوریه کوانتومی (QFT).
        :param vector: بردار ورودی که باید به فضای کوانتومی تبدیل شود.
        :return: بردار تبدیل‌شده در فضای کوانتومی.
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        # اجرای تبدیل فوریه کوانتومی (QFT)
        quantum_vector = np.fft.fft(vector)

        return quantum_vector

    @staticmethod
    def inverse_transform(quantum_vector: np.ndarray) -> np.ndarray:
        """
        بازیابی بردار اولیه از فضای کوانتومی با استفاده از تبدیل معکوس QFT.
        :param quantum_vector: بردار تبدیل‌شده در فضای کوانتومی.
        :return: بردار اصلی بازیابی‌شده.
        """
        if not isinstance(quantum_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        # اجرای تبدیل معکوس فوریه کوانتومی (Inverse QFT)
        original_vector = np.fft.ifft(quantum_vector).real  # مقدار حقیقی را بازمی‌گردانیم

        return original_vector

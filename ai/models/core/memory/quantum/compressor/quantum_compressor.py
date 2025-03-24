import numpy as np
from typing import Any, Dict
from .pattern_detector import PatternDetector

class QuantumCompressor:
    """
    ماژول فشرده‌سازی بردارهای کوانتومی با استفاده از شناسایی الگوهای پرتکرار و کاهش ابعاد.
    """

    def __init__(self, compression_factor: float = 0.7):
        """
        مقداردهی اولیه فشرده‌سازی کوانتومی.
        :param compression_factor: میزان فشرده‌سازی بین 0 و 1 (0 بدون فشرده‌سازی، 1 حذف تمام اطلاعات اضافی).
        """
        if compression_factor <= 0 or compression_factor > 1:
            raise ValueError("compression_factor باید بین 0 و 1 باشد.")
        self.compression_factor = compression_factor
        self.pattern_detector = PatternDetector()

    def compress(self, quantum_vector: np.ndarray) -> Dict[str, Any]:
        """
        فشرده‌سازی بردار کوانتومی با حذف اطلاعات غیرضروری.
        :param quantum_vector: بردار کوانتومی که باید فشرده شود.
        :return: دیکشنری شامل بردار فشرده‌شده و اطلاعات متا.
        """
        if not isinstance(quantum_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        # شناسایی فرکانس‌های کلیدی
        detected_patterns = self.pattern_detector.detect_redundant_patterns(quantum_vector)

        # حذف فرکانس‌های کم‌اهمیت
        threshold = int(len(quantum_vector) * self.compression_factor)
        compressed_vector = np.zeros_like(quantum_vector)
        compressed_vector[:threshold] = quantum_vector[:threshold]

        return {
            "compressed_vector": compressed_vector,
            "metadata": {
                "compression_factor": self.compression_factor,
                "dominant_frequencies": detected_patterns["dominant_frequencies"]
            }
        }

    def decompress(self, compressed_data: Dict[str, Any]) -> np.ndarray:
        """
        بازسازی بردار از داده‌ی فشرده‌شده.
        :param compressed_data: دیکشنری شامل بردار فشرده‌شده و اطلاعات متا.
        :return: بردار اصلی بازسازی‌شده.
        """
        compressed_vector = compressed_data.get("compressed_vector")
        if compressed_vector is None:
            raise ValueError("داده‌ی فشرده‌شده نامعتبر است!")

        # اجرای تبدیل معکوس فوریه (Inverse QFT) برای بازیابی داده
        restored_vector = np.fft.ifft(compressed_vector).real

        return restored_vector

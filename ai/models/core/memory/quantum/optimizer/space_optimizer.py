import numpy as np
from scipy.sparse import csr_matrix
from typing import Any, Dict

class SpaceOptimizer:
    """
    ماژول بهینه‌سازی فضای ذخیره‌سازی بردارهای کوانتومی با استفاده از کاهش ابعاد و فشرده‌سازی.
    """

    def __init__(self, sparsity_threshold: float = 0.05):
        """
        مقداردهی اولیه برای بهینه‌سازی فضا.
        :param sparsity_threshold: آستانه‌ی تبدیل داده‌ها به `Sparse Matrix` (بین 0 و 1).
        """
        if sparsity_threshold <= 0 or sparsity_threshold > 1:
            raise ValueError("sparsity_threshold باید بین 0 و 1 باشد.")
        self.sparsity_threshold = sparsity_threshold

    def optimize_storage(self, quantum_vector: np.ndarray) -> Dict[str, Any]:
        """
        فشرده‌سازی بردارهای کوانتومی با استفاده از `Sparse Matrix`.
        :param quantum_vector: بردار کوانتومی که باید بهینه شود.
        :return: دیکشنری شامل بردار فشرده‌شده و متادیتا.
        """
        if not isinstance(quantum_vector, np.ndarray):
            raise ValueError("ورودی باید یک آرایه numpy باشد.")

        # شناسایی مقادیر صفر یا نزدیک به صفر
        sparsity_mask = np.abs(quantum_vector) < self.sparsity_threshold
        sparse_vector = csr_matrix(quantum_vector)

        return {
            "sparse_vector": sparse_vector,
            "metadata": {
                "original_size": quantum_vector.size,
                "compressed_size": sparse_vector.data.size,
                "compression_ratio": sparse_vector.data.size / quantum_vector.size
            }
        }

    def restore_vector(self, sparse_data: Dict[str, Any]) -> np.ndarray:
        """
        بازسازی بردار کوانتومی از داده‌های فشرده‌شده.
        :param sparse_data: دیکشنری شامل بردار فشرده‌شده.
        :return: بردار اصلی بازسازی‌شده.
        """
        sparse_vector = sparse_data.get("sparse_vector")
        if sparse_vector is None or not isinstance(sparse_vector, csr_matrix):
            raise ValueError("داده‌ی فشرده‌شده نامعتبر است!")

        return sparse_vector.toarray().flatten()

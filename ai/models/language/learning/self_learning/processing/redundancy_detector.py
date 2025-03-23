"""
RedundancyDetector Module
---------------------------
این فایل مسئول تشخیص داده‌های تکراری یا افزونگی در مجموعه داده‌های ورودی به سیستم خودآموزی است.
کلاس RedundancyDetector با استفاده از محاسبه‌ی هش (hash) نمایه‌های داده (برای داده‌های متنی یا ساختار یافته)
، داده‌های تکراری را شناسایی کرده و امکان فیلتر کردن آن‌ها را فراهم می‌کند.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده است.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set

from ..base.base_component import BaseComponent


class RedundancyDetector(BaseComponent):
    """
    RedundancyDetector مسئول شناسایی و فیلتر کردن داده‌های تکراری در مجموعه‌های داده است.

    امکانات:
      - محاسبه هش داده به منظور بررسی یکتایی.
      - نگهداری یک مجموعه از هش‌های داده‌های دیده‌شده.
      - فراهم آوردن متدهایی برای بررسی تکراری بودن داده، افزودن داده به مجموعه و فیلتر کردن مجموعه‌ای از داده‌ها.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه RedundancyDetector.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "hash_algorithm": الگوریتم هش مورد استفاده (پیش‌فرض: "sha256")
        """
        super().__init__(component_type="redundancy_detector", config=config)
        self.logger = logging.getLogger("RedundancyDetector")
        self.hash_algorithm = self.config.get("hash_algorithm", "sha256")
        self.seen_hashes: Set[str] = set()
        self.logger.info(f"[RedundancyDetector] Initialized using hash algorithm: {self.hash_algorithm}")

    def _compute_hash(self, data: Any) -> str:
        """
        محاسبه هش داده ورودی.

        Args:
            data (Any): داده ورودی؛ داده‌های متنی یا داده‌هایی که به رشته تبدیل شوند.

        Returns:
            str: هش محاسبه‌شده به صورت رشته هگزادسیمال.
        """
        # تبدیل داده به رشته و سپس به بایت
        data_str = str(data).strip()
        data_bytes = data_str.encode("utf-8")
        hash_func = hashlib.new(self.hash_algorithm)
        hash_func.update(data_bytes)
        computed_hash = hash_func.hexdigest()
        self.logger.debug(f"[RedundancyDetector] Computed hash for data: {computed_hash}")
        return computed_hash

    def is_duplicate(self, data: Any) -> bool:
        """
        بررسی تکراری بودن داده بر اساس محاسبه هش.

        Args:
            data (Any): داده ورودی.

        Returns:
            bool: True اگر داده تکراری باشد، در غیر این صورت False.
        """
        data_hash = self._compute_hash(data)
        if data_hash in self.seen_hashes:
            self.logger.debug("[RedundancyDetector] Data is duplicate.")
            return True
        self.logger.debug("[RedundancyDetector] Data is unique.")
        return False

    def add_entry(self, data: Any) -> None:
        """
        افزودن داده به مجموعه‌ی دیده‌شده پس از بررسی تکراری بودن.

        Args:
            data (Any): داده ورودی.
        """
        data_hash = self._compute_hash(data)
        self.seen_hashes.add(data_hash)
        self.logger.debug(f"[RedundancyDetector] Added data hash {data_hash} to seen set.")

    def filter_duplicates(self, data_list: List[Any]) -> List[Any]:
        """
        فیلتر کردن داده‌های تکراری از یک لیست داده‌ها.

        Args:
            data_list (List[Any]): لیستی از داده‌های ورودی.

        Returns:
            List[Any]: لیستی از داده‌های یکتا (بدون تکرار).
        """
        unique_data = []
        for data in data_list:
            if not self.is_duplicate(data):
                self.add_entry(data)
                unique_data.append(data)
            else:
                self.logger.debug("[RedundancyDetector] Duplicate data filtered out.")
        self.logger.info(
            f"[RedundancyDetector] Filtered {len(data_list) - len(unique_data)} duplicates; {len(unique_data)} unique entries remain.")
        return unique_data


# Example usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    rd = RedundancyDetector(config={"hash_algorithm": "sha256"})
    data_samples = [
        "This is a sample text.",
        "This is a sample text.",  # duplicate
        "Another unique piece of data.",
        "This is a sample text.  ",  # duplicate due to extra spaces removed by strip()
        {"id": 1, "value": "Test data"},
        {"id": 1, "value": "Test data"}  # duplicate dictionary (string representation same)
    ]
    unique_entries = rd.filter_duplicates(data_samples)
    print("Unique entries:")
    for entry in unique_entries:
        print(entry)

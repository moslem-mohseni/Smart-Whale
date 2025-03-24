import hashlib
import os


class HashService:
    """
    سرویس محاسبه هش فایل‌ها برای مدیریت Deduplication
    """

    @staticmethod
    def calculate_file_hash(file_path: str, algorithm: str = "sha256") -> str:
        """محاسبه هش فایل بر اساس الگوریتم مشخص شده"""
        hash_func = getattr(hashlib, algorithm, None)
        if hash_func is None:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hasher = hash_func()
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def calculate_bytes_hash(file_data: bytes, algorithm: str = "sha256") -> str:
        """محاسبه هش از داده‌های باینری"""
        hash_func = getattr(hashlib, algorithm, None)
        if hash_func is None:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

        hasher = hash_func()
        hasher.update(file_data)
        return hasher.hexdigest()


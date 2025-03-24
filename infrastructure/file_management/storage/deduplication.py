import os
from infrastructure.file_management.domain.hash_service import HashService
from infrastructure.file_management.cache.hash_cache import HashCache


class Deduplication:
    """
    مدیریت فایل‌های تکراری با بررسی هش
    """

    def __init__(self):
        self.hash_service = HashService()
        self.hash_cache = HashCache()

    async def check_duplicate(self, file_path: str) -> bool:
        """بررسی تکراری بودن فایل از طریق هش"""
        file_hash = self.hash_service.calculate_file_hash(file_path)
        cached_hash = await self.hash_cache.get_file_hash(os.path.basename(file_path))
        return cached_hash == file_hash if cached_hash else False

    async def store_file_hash(self, file_path: str):
        """ذخیره هش فایل برای استفاده در آینده"""
        file_hash = self.hash_service.calculate_file_hash(file_path)
        await self.hash_cache.store_file_hash(os.path.basename(file_path), file_hash)

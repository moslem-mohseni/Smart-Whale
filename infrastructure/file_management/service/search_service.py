import os
from infrastructure.file_management.cache.hash_cache import HashCache


class SearchService:
    """
    سرویس جستجوی فایل‌ها بر اساس متادیتا و هش فایل‌ها
    """

    def __init__(self):
        self.hash_cache = HashCache()

    async def search_by_file_name(self, file_name: str):
        """جستجو بر اساس نام فایل"""
        cached_hash = await self.hash_cache.get_file_hash(file_name)
        if cached_hash:
            return {"status": "success", "file_name": file_name, "hash": cached_hash}
        return {"status": "error", "message": "File not found"}

    async def search_by_hash(self, file_hash: str):
        """جستجو بر اساس هش فایل"""
        for key in await self.hash_cache.redis.keys("file_hash:*"):
            stored_hash = await self.hash_cache.get_file_hash(key.decode("utf-8").split(":")[-1])
            if stored_hash == file_hash:
                return {"status": "success", "file_name": key.decode("utf-8").split(":")[-1]}
        return {"status": "error", "message": "No file found with this hash"}

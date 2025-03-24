import os
import asyncio
from datetime import datetime, timedelta
from infrastructure.clickhouse.adapters.clickhouse_adapter import ClickHouseAdapter
from infrastructure.file_management.adapters.minio_adapter import MinIOAdapter
from infrastructure.redis.service.cache_service import CacheService

class CleanupManager:
    """
    مدیریت حذف داده‌های قدیمی برای بهینه‌سازی فضای ذخیره‌سازی.
    """

    def __init__(self, retention_days: int = 30, minio_bucket: str = "backups"):
        """
        مقداردهی اولیه.

        :param retention_days: تعداد روزهایی که داده‌ها باید نگه داشته شوند (پیش‌فرض: ۳۰ روز)
        :param minio_bucket: نام باکت MinIO برای مدیریت فایل‌های قدیمی
        """
        self.retention_days = retention_days
        self.expiry_date = datetime.utcnow() - timedelta(days=retention_days)
        self.clickhouse_adapter = ClickHouseAdapter()
        self.minio_adapter = MinIOAdapter()
        self.redis_cache = CacheService()
        self.minio_bucket = minio_bucket

    async def connect(self) -> None:
        """ اتصال به سرویس‌های موردنیاز. """
        await self.clickhouse_adapter.connect()
        await self.minio_adapter.connect()
        await self.redis_cache.connect()

    async def cleanup_clickhouse(self) -> None:
        """
        حذف داده‌های قدیمی از ClickHouse.
        """
        query = f"DELETE FROM analytics_data WHERE created_at < '{self.expiry_date.strftime('%Y-%m-%d')}'"
        await self.clickhouse_adapter.execute(query)
        print(f"✅ داده‌های قدیمی‌تر از {self.retention_days} روز در ClickHouse حذف شدند.")

    async def cleanup_redis(self) -> None:
        """
        حذف کلیدهای منقضی‌شده از Redis.
        """
        async for key in self.redis_cache.keys("*"):
            if await self.redis_cache.get(key) is None:
                await self.redis_cache.delete(key)
        print(f"✅ کلیدهای منقضی‌شده در Redis حذف شدند.")

    async def cleanup_minio(self) -> None:
        """
        حذف فایل‌های قدیمی از MinIO.
        """
        all_files = await self.minio_adapter.list_files(self.minio_bucket)
        for file_info in all_files:
            file_name, last_modified = file_info["name"], file_info["last_modified"]
            if last_modified < self.expiry_date:
                await self.minio_adapter.delete_file(self.minio_bucket, file_name)
                print(f"🗑 فایل قدیمی `{file_name}` از MinIO حذف شد.")

    async def run_cleanup(self) -> None:
        """
        اجرای فرآیند حذف داده‌های قدیمی.
        """
        await self.cleanup_clickhouse()
        await self.cleanup_redis()
        await self.cleanup_minio()

    async def close(self) -> None:
        """ قطع اتصال از سرویس‌های موردنیاز. """
        await self.clickhouse_adapter.disconnect()
        await self.minio_adapter.disconnect()
        await self.redis_cache.disconnect()

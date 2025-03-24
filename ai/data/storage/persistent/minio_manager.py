from infrastructure.file_management.adapters.minio_adapter import MinIOAdapter
from infrastructure.redis.service.cache_service import CacheService
from typing import Optional


class MinIOManager:
    """
    مدیریت ذخیره‌سازی فایل‌ها در MinIO.
    """

    def __init__(self):
        self.minio_adapter = MinIOAdapter()
        self.cache_service = CacheService()

    async def connect(self) -> None:
        """ اتصال به MinIO و Redis """
        await self.minio_adapter.connect()
        await self.cache_service.connect()

    async def upload_file(self, file_name: str, file_data: bytes, bucket: str = "default") -> str:
        """
        آپلود فایل در MinIO.

        :param file_name: نام فایل
        :param file_data: داده‌های فایل (باینری)
        :param bucket: نام باکت در MinIO (پیش‌فرض: default)
        :return: URL فایل آپلود شده
        """
        file_url = await self.minio_adapter.upload_file(bucket, file_name, file_data)

        # ذخیره متادیتا در Redis
        await self.cache_service.set(f"file:{file_name}", file_url, ttl=86400)  # TTL: 1 روز

        return file_url

    async def download_file(self, file_name: str, bucket: str = "default") -> Optional[bytes]:
        """
        دریافت فایل از MinIO.

        :param file_name: نام فایل
        :param bucket: نام باکت در MinIO (پیش‌فرض: default)
        :return: داده‌های فایل یا None در صورت عدم وجود
        """
        return await self.minio_adapter.download_file(bucket, file_name)

    async def delete_file(self, file_name: str, bucket: str = "default") -> bool:
        """
        حذف فایل از MinIO.

        :param file_name: نام فایل
        :param bucket: نام باکت در MinIO (پیش‌فرض: default)
        :return: True در صورت حذف موفق، False در غیر این صورت
        """
        success = await self.minio_adapter.delete_file(bucket, file_name)

        if success:
            # حذف متادیتا از Redis
            await self.cache_service.delete(f"file:{file_name}")

        return success

    async def close(self) -> None:
        """ قطع اتصال از MinIO و Redis """
        await self.minio_adapter.disconnect()
        await self.cache_service.disconnect()

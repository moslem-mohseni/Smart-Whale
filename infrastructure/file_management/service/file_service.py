from infrastructure.file_management.adapters.minio_adapter import MinioAdapter
from infrastructure.file_management.cache.hash_cache import HashCache
from infrastructure.file_management.domain.hash_service import HashService
from infrastructure.file_management.storage.deduplication import Deduplication
from infrastructure.file_management.storage.file_store import FileStore
from infrastructure.file_management.security.access_control import AccessControl
from infrastructure.file_management.security.file_validator import FileValidator


class FileService:
    """
    سرویس مدیریت فایل شامل آپلود، دانلود، حذف و کنترل دسترسی
    """

    def __init__(self):
        self.minio_adapter = MinioAdapter()
        self.hash_cache = HashCache()
        self.hash_service = HashService()
        self.deduplication = Deduplication()
        self.file_store = FileStore()
        self.access_control = AccessControl()
        self.file_validator = FileValidator()

    async def upload_file(self, file_path: str, file_name: str, user_id: str, permissions: list):
        """آپلود فایل با بررسی تکراری بودن و اعتبارسنجی"""
        if not self.file_validator.validate_file_type(file_path) or not self.file_validator.validate_file_size(
                file_path):
            return {"status": "error", "message": "Invalid file type or size"}

        if await self.deduplication.check_duplicate(file_path):
            return {"status": "error", "message": "Duplicate file detected"}

        stored_path = self.file_store.save_file(file_path, file_name)
        await self.deduplication.store_file_hash(stored_path)
        upload_result = self.minio_adapter.upload_file(open(stored_path, 'rb'), file_name, "application/octet-stream")
        token = self.access_control.generate_token(user_id, permissions)
        return {"status": "success", "file_name": file_name, "access_token": token, "upload_result": upload_result}

    async def download_file(self, file_name: str, token: str):
        """دانلود فایل با بررسی دسترسی"""
        auth_data = self.access_control.verify_token(token)
        if "error" in auth_data:
            return {"status": "error", "message": auth_data["error"]}

        file_data = self.minio_adapter.download_file(file_name)
        return {"status": "success", "file_name": file_name, "file_data": file_data}

    async def delete_file(self, file_name: str, token: str):
        """حذف فایل با بررسی دسترسی"""
        auth_data = self.access_control.verify_token(token)
        if "error" in auth_data:
            return {"status": "error", "message": auth_data["error"]}

        self.file_store.delete_file(file_name)
        self.minio_adapter.delete_file(file_name)
        await self.hash_cache.delete_file_hash(file_name)
        return {"status": "success", "message": "File deleted successfully"}

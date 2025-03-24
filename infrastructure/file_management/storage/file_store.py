import os
import shutil


class FileStore:
    """
    مدیریت ذخیره‌سازی فایل‌ها در مسیر مشخص شده
    """

    def __init__(self):
        self.storage_path = os.getenv("FILE_STORAGE_PATH", "./storage")
        os.makedirs(self.storage_path, exist_ok=True)

    def save_file(self, file_path: str, destination_name: str) -> str:
        """ذخیره فایل در مسیر مشخص شده"""
        destination_path = os.path.join(self.storage_path, destination_name)
        shutil.move(file_path, destination_path)
        return destination_path

    def delete_file(self, file_name: str) -> bool:
        """حذف فایل از مسیر ذخیره‌سازی"""
        file_path = os.path.join(self.storage_path, file_name)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False

    def file_exists(self, file_name: str) -> bool:
        """بررسی وجود فایل در مسیر ذخیره‌سازی"""
        return os.path.exists(os.path.join(self.storage_path, file_name))

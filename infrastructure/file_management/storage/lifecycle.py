import os
import time


class FileLifecycle:
    """
    مدیریت چرخه حیات فایل‌ها شامل حذف و آرشیو فایل‌های قدیمی
    """

    def __init__(self):
        self.storage_path = os.getenv("FILE_STORAGE_PATH", "./storage")
        self.retention_days = int(os.getenv("FILE_RETENTION_DAYS", 30))

    def cleanup_old_files(self):
        """حذف فایل‌های قدیمی‌تر از مدت زمان مشخص"""
        current_time = time.time()
        for file_name in os.listdir(self.storage_path):
            file_path = os.path.join(self.storage_path, file_name)
            if os.path.isfile(file_path):
                file_age_days = (current_time - os.path.getmtime(file_path)) / 86400
                if file_age_days > self.retention_days:
                    os.remove(file_path)

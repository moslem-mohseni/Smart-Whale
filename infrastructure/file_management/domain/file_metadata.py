import time


class FileMetadata:
    """
    مدیریت متادیتای فایل‌ها شامل نام، اندازه، نوع و زمان آپلود
    """

    def __init__(self, file_name: str, file_size: int, file_type: str):
        self.file_name = file_name
        self.file_size = file_size
        self.file_type = file_type
        self.upload_time = time.time()

    def to_dict(self):
        """تبدیل متادیتا به دیکشنری"""
        return {
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "upload_time": self.upload_time
        }

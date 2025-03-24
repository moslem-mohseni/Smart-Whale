import time


class File:
    """
    مدل داده‌ای فایل شامل اطلاعات کلیدی فایل‌های ذخیره‌شده
    """

    def __init__(self, file_id: str, file_name: str, file_size: int, file_type: str):
        self.file_id = file_id
        self.file_name = file_name
        self.file_size = file_size
        self.file_type = file_type
        self.created_at = time.time()

    def to_dict(self):
        """تبدیل مدل به دیکشنری"""
        return {
            "file_id": self.file_id,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "created_at": self.created_at
        }

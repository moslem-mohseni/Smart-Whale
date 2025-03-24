import os
import magic


class FileValidator:
    """
    اعتبارسنجی فایل‌ها شامل بررسی نوع، اندازه و امنیت
    """

    def __init__(self):
        self.allowed_file_types = os.getenv("ALLOWED_FILE_TYPES", "jpg,png,pdf,txt").split(",")
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", 104857600))  # 100MB

    def validate_file_type(self, file_path: str) -> bool:
        """بررسی نوع فایل مجاز بر اساس MIME type"""
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return any(file_type.endswith(ext) for ext in self.allowed_file_types)

    def validate_file_size(self, file_path: str) -> bool:
        """بررسی محدودیت حجم فایل"""
        return os.path.getsize(file_path) <= self.max_file_size

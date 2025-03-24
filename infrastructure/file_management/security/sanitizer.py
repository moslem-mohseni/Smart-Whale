import re


class FileSanitizer:
    """
    پاکسازی نام و مسیر فایل‌ها از کاراکترهای خطرناک
    """

    @staticmethod
    def sanitize_file_name(file_name: str) -> str:
        """حذف کاراکترهای خطرناک از نام فایل"""
        return re.sub(r'[^a-zA-Z0-9._-]', '_', file_name)

    @staticmethod
    def sanitize_path(path: str) -> str:
        """حذف کاراکترهای غیرمجاز از مسیر فایل"""
        return re.sub(r'[^a-zA-Z0-9/_-]', '_', path)

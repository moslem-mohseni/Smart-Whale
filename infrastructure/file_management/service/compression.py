import os
import gzip
import shutil


class FileCompression:
    """
    مدیریت فشرده‌سازی و استخراج فایل‌ها
    """

    def __init__(self):
        self.storage_path = os.getenv("FILE_STORAGE_PATH", "./storage")

    def compress_file(self, file_name: str) -> str:
        """فشرده‌سازی فایل با Gzip"""
        file_path = os.path.join(self.storage_path, file_name)
        compressed_path = f"{file_path}.gz"
        with open(file_path, 'rb') as f_in, gzip.open(compressed_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return compressed_path

    def decompress_file(self, compressed_file_name: str) -> str:
        """استخراج فایل فشرده‌شده"""
        compressed_path = os.path.join(self.storage_path, compressed_file_name)
        if not compressed_file_name.endswith(".gz"):
            raise ValueError("Invalid compressed file format")
        original_path = compressed_path[:-3]
        with gzip.open(compressed_path, 'rb') as f_in, open(original_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return original_path

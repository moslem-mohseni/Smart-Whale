import gzip
import bz2
import lzma
import shutil
from typing import Optional, Union
from pathlib import Path


class CompressionManager:
    """
    مدیریت فشرده‌سازی و کاهش حجم داده‌ها با پشتیبانی از چندین الگوریتم.
    """

    def __init__(self, algorithm: str = "gzip", compression_level: int = 5):
        """
        مقداردهی اولیه ماژول فشرده‌سازی.

        :param algorithm: نوع الگوریتم فشرده‌سازی (gzip, bz2, lzma)
        :param compression_level: سطح فشرده‌سازی (۱ تا ۹، فقط برای gzip و bz2)
        """
        if algorithm not in ["gzip", "bz2", "lzma"]:
            raise ValueError("الگوریتم نامعتبر! فقط 'gzip'، 'bz2' و 'lzma' پشتیبانی می‌شوند.")

        self.algorithm = algorithm
        self.compression_level = compression_level

    def compress_data(self, data: bytes) -> Optional[bytes]:
        """
        فشرده‌سازی داده‌های ورودی.

        :param data: داده‌های باینری برای فشرده‌سازی
        :return: داده فشرده‌شده یا None در صورت خطا
        """
        try:
            if self.algorithm == "gzip":
                return gzip.compress(data, compresslevel=self.compression_level)
            elif self.algorithm == "bz2":
                return bz2.compress(data, compresslevel=self.compression_level)
            elif self.algorithm == "lzma":
                return lzma.compress(data)
        except Exception as e:
            print(f"⚠️ خطا در فشرده‌سازی داده: {e}")
            return None

    def decompress_data(self, compressed_data: bytes) -> Optional[bytes]:
        """
        باز کردن داده‌های فشرده‌شده.

        :param compressed_data: داده‌های فشرده‌شده
        :return: داده باز شده یا None در صورت خطا
        """
        try:
            if self.algorithm == "gzip":
                return gzip.decompress(compressed_data)
            elif self.algorithm == "bz2":
                return bz2.decompress(compressed_data)
            elif self.algorithm == "lzma":
                return lzma.decompress(compressed_data)
        except Exception as e:
            print(f"⚠️ خطا در باز کردن داده فشرده‌شده: {e}")
            return None

    def compress_file(self, file_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        فشرده‌سازی یک فایل و ذخیره در مسیر مشخص.

        :param file_path: مسیر فایل ورودی
        :param output_path: مسیر فایل خروجی فشرده‌شده
        :return: True در صورت موفقیت، False در صورت خطا
        """
        try:
            file_path = Path(file_path)
            output_path = Path(output_path)

            if self.algorithm == "gzip":
                with open(file_path, 'rb') as f_in, gzip.open(output_path, 'wb',
                                                              compresslevel=self.compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            elif self.algorithm == "bz2":
                with open(file_path, 'rb') as f_in, bz2.open(output_path, 'wb',
                                                             compresslevel=self.compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            elif self.algorithm == "lzma":
                with open(file_path, 'rb') as f_in, lzma.open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return True
        except Exception as e:
            print(f"⚠️ خطا در فشرده‌سازی فایل: {e}")
            return False

    def decompress_file(self, compressed_file_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        باز کردن یک فایل فشرده و ذخیره در مسیر مشخص.

        :param compressed_file_path: مسیر فایل فشرده‌شده
        :param output_path: مسیر فایل خروجی پس از استخراج
        :return: True در صورت موفقیت، False در صورت خطا
        """
        try:
            compressed_file_path = Path(compressed_file_path)
            output_path = Path(output_path)

            if self.algorithm == "gzip":
                with gzip.open(compressed_file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            elif self.algorithm == "bz2":
                with bz2.open(compressed_file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            elif self.algorithm == "lzma":
                with lzma.open(compressed_file_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            return True
        except Exception as e:
            print(f"⚠️ خطا در باز کردن فایل فشرده‌شده: {e}")
            return False

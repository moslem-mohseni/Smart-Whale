import zlib
from typing import Any


class DataCompressor:
    """
    فشرده‌سازی داده‌های دانش برای کاهش حجم انتقال و افزایش کارایی ارتباطات
    """

    @staticmethod
    def compress_data(data: Any) -> bytes:
        """
        فشرده‌سازی داده‌های ورودی برای کاهش حجم انتقال
        :param data: داده‌های ورودی به‌صورت رشته یا بایت
        :return: داده‌های فشرده‌شده به‌صورت بایت
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        return zlib.compress(data)

    @staticmethod
    def decompress_data(compressed_data: bytes) -> str:
        """
        استخراج داده‌های فشرده‌شده
        :param compressed_data: داده‌های فشرده‌شده به‌صورت بایت
        :return: داده‌های اصلی پس از استخراج
        """
        return zlib.decompress(compressed_data).decode('utf-8')


# نمونه استفاده از DataCompressor برای تست
if __name__ == "__main__":
    compressor = DataCompressor()
    original_data = "This is a test string for compression."
    compressed = compressor.compress_data(original_data)
    decompressed = compressor.decompress_data(compressed)

    print(f"Original Data: {original_data}")
    print(f"Compressed Data: {compressed}")
    print(f"Decompressed Data: {decompressed}")

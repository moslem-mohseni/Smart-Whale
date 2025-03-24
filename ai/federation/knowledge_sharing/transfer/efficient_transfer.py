from typing import Any
from .data_compressor import DataCompressor


class EfficientTransfer:
    """
    مدیریت انتقال بهینه داده‌های دانش با استفاده از فشرده‌سازی و رمزگذاری
    """

    def __init__(self):
        self.compressor = DataCompressor()

    def prepare_data_for_transfer(self, data: Any) -> bytes:
        """
        آماده‌سازی داده‌ها برای انتقال از طریق فشرده‌سازی
        :param data: داده‌های ورودی
        :return: داده‌های فشرده‌شده آماده برای ارسال
        """
        return self.compressor.compress_data(data)

    def receive_transferred_data(self, compressed_data: bytes) -> Any:
        """
        دریافت داده‌های فشرده‌شده و بازگردانی آن‌ها به حالت اصلی
        :param compressed_data: داده‌های فشرده‌شده دریافت‌شده
        :return: داده‌های اصلی پس از استخراج
        """
        return self.compressor.decompress_data(compressed_data)


# نمونه استفاده از EfficientTransfer برای تست
if __name__ == "__main__":
    transfer = EfficientTransfer()
    original_data = "This is a sample data for efficient transfer."
    compressed = transfer.prepare_data_for_transfer(original_data)
    decompressed = transfer.receive_transferred_data(compressed)

    print(f"Original Data: {original_data}")
    print(f"Compressed Data: {compressed}")
    print(f"Decompressed Data: {decompressed}")

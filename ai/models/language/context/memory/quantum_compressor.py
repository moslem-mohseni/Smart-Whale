import zlib
import base64
from typing import List

class QuantumCompressor:
    """
    این کلاس مسئول فشرده‌سازی داده‌های مکالمه‌ای برای کاهش مصرف حافظه در `L2 Cache` و `L3 Cache` است.
    """

    def __init__(self, compression_level: int = 6):
        """
        مقداردهی اولیه `QuantumCompressor`.
        :param compression_level: سطح فشرده‌سازی (`0` تا `9`، پیش‌فرض: `6`)
        """
        self.compression_level = compression_level

    def compress_data(self, data: str) -> str:
        """
        فشرده‌سازی متن ورودی برای کاهش حجم ذخیره‌سازی.
        :param data: رشته‌ی متنی برای فشرده‌سازی
        :return: داده‌ی فشرده‌شده به صورت `Base64`
        """
        compressed = zlib.compress(data.encode('utf-8'), self.compression_level)
        return base64.b64encode(compressed).decode('utf-8')

    def decompress_data(self, compressed_data: str) -> str:
        """
        باز کردن فشرده‌سازی داده برای بازیابی اطلاعات.
        :param compressed_data: داده‌ی فشرده‌شده به صورت `Base64`
        :return: داده‌ی اصلی به صورت متن
        """
        decompressed = zlib.decompress(base64.b64decode(compressed_data))
        return decompressed.decode('utf-8')

    def compress_messages(self, messages: List[str]) -> str:
        """
        فشرده‌سازی لیست پیام‌های مکالمه‌ای.
        :param messages: لیستی از پیام‌های مکالمه
        :return: داده‌ی فشرده‌شده به صورت `Base64`
        """
        combined_text = "\n".join(messages)
        return self.compress_data(combined_text)

    def decompress_messages(self, compressed_messages: str) -> List[str]:
        """
        باز کردن فشرده‌سازی پیام‌های مکالمه‌ای.
        :param compressed_messages: داده‌ی فشرده‌شده
        :return: لیستی از پیام‌های اصلی
        """
        decompressed_text = self.decompress_data(compressed_messages)
        return decompressed_text.split("\n")


# تست اولیه ماژول
if __name__ == "__main__":
    compressor = QuantumCompressor()

    messages = [
        "سلام، امروز چه خبر؟",
        "من به دنبال یادگیری مدل‌های زبان طبیعی هستم.",
        "چه تفاوتی بین BERT و GPT وجود دارد؟",
        "چطور می‌توان از `Transfer Learning` استفاده کرد؟"
    ]

    compressed_data = compressor.compress_messages(messages)
    print("\n🔹 Compressed Data:")
    print(compressed_data)

    decompressed_messages = compressor.decompress_messages(compressed_data)
    print("\n🔹 Decompressed Messages:")
    print(decompressed_messages)

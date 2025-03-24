import zlib
import pickle
from typing import Any

class Compression:
    """
    مدیریت فشرده‌سازی و استخراج داده‌ها در Redis
    """
    @staticmethod
    def compress(data: Any) -> bytes:
        """فشرده‌سازی داده قبل از ذخیره در Redis"""
        serialized = pickle.dumps(data)
        return zlib.compress(serialized)

    @staticmethod
    def decompress(data: bytes) -> Any:
        """باز کردن فشرده‌سازی داده دریافت‌شده از Redis"""
        decompressed = zlib.decompress(data)
        return pickle.loads(decompressed)

# مثال استفاده:
# compressed_data = Compression.compress({"key": "value"})
# original_data = Compression.decompress(compressed_data)

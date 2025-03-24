from typing import List, Dict, Any
import hashlib


class RequestMerger:
    """
    این کلاس مسئول ادغام درخواست‌های مشابه برای کاهش تعداد پردازش‌های غیرضروری است.
    """

    def __init__(self):
        self.merged_requests = {}

    def _generate_hash(self, request: Dict[str, Any]) -> str:
        """
        تولید هش یکتا برای درخواست بر اساس محتوای آن.
        """
        request_str = str(sorted(request.items()))
        return hashlib.md5(request_str.encode()).hexdigest()

    def merge_requests(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        بررسی و ادغام درخواست‌های مشابه برای کاهش پردازش‌های تکراری.
        """
        merged_data = {}

        for request in batch_data:
            request_hash = self._generate_hash(request)

            if request_hash in merged_data:
                # افزایش شمارش درخواست‌های مشابه به‌جای پردازش مجدد
                merged_data[request_hash]['count'] += 1
            else:
                # ذخیره درخواست جدید در لیست پردازش
                merged_data[request_hash] = {**request, 'count': 1}

        return list(merged_data.values())

"""
RequestBuilder Module
-----------------------
این فایل مسئول ساخت درخواست‌های داده جهت آموزش مدل در سیستم خودآموزی است.
کلاس RequestBuilder یک درخواست استاندارد با ساختار پیام شامل بخش‌های metadata و payload می‌سازد.
پیام نهایی طبق استانداردهای ارتباطی با ماژول‌های Balance و Data (مثلاً جهت ارسال از طریق Kafka) تولید می‌شود.

این نسخه نهایی و عملیاتی با بهترین مکانیسم‌ها و کارایی بالا پیاده‌سازی شده است.
"""

import uuid
from datetime import datetime
import logging
from typing import Any, Dict, Optional


class RequestBuilder:
    """
    RequestBuilder مسئول ساخت درخواست‌های جمع‌آوری داده جهت آموزش مدل است.

    ویژگی‌ها:
      - تولید پیام استاندارد با ساختار "metadata" و "payload".
      - استفاده از UUID برای شناسه یکتا و زمان‌بندی به صورت ISO.
      - پذیرش پارامترهای مختلف مانند query، data_type، source_type، و سایر پارامترهای اختیاری.
      - قابلیت تعیین model_id و response_topic برای هماهنگی دقیق با سرویس‌های Balance و Data.
    """

    def __init__(self, model_id: Optional[str] = None, response_topic: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه RequestBuilder.

        Args:
            model_id (Optional[str]): شناسه مدل مرتبط با درخواست (در صورت نیاز).
            response_topic (Optional[str]): موضوع پاسخ جهت دریافت نتایج از سرویس Data.
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری جهت تنظیمات پیش‌فرض (مانند سطح اولویت).
        """
        self.logger = logging.getLogger("RequestBuilder")
        self.model_id = model_id
        self.response_topic = response_topic or "DATA_RESPONSE_TOPIC"
        self.config = config or {}
        # سطح پیش‌فرض اولویت در صورت عدم ارائه؛ عددی صحیح (مثلاً 2 برای متوسط)
        self.default_priority = int(self.config.get("default_priority", 2))
        self.logger.info(
            f"[RequestBuilder] Initialized with model_id={self.model_id} and response_topic={self.response_topic}")

    def build_request(self,
                      query: str,
                      data_type: str = "TEXT",
                      source_type: str = "WIKI",
                      params: Optional[Dict[str, Any]] = None,
                      priority: Optional[int] = None
                      ) -> Dict[str, Any]:
        """
        ساخت یک درخواست جمع‌آوری داده به صورت استاندارد.

        Args:
            query (str): عبارت یا ورودی جستجو (مثلاً عنوان مقاله یا URL).
            data_type (str): نوع داده مورد نظر (مثلاً "TEXT", "IMAGE").
            source_type (str): منبع داده (مثلاً "WIKI", "WEB", "TWITTER").
            params (Optional[Dict[str, Any]]): پارامترهای اختیاری برای درخواست.
            priority (Optional[int]): سطح اولویت درخواست؛ در صورت عدم ارائه از مقدار پیش‌فرض استفاده می‌شود.

        Returns:
            Dict[str, Any]: درخواست نهایی به صورت دیکشنری با ساختار استاندارد.
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        used_priority = priority if priority is not None else self.default_priority

        # در صورت عدم ارائه پارامترهای اختیاری، یک دیکشنری خالی در نظر می‌گیریم.
        params = params or {}

        # در صورت منبع "WIKI" و عدم وجود پارامترهای خاص، تنظیمات پیش‌فرض برای ویکی را اعمال می‌کنیم.
        if source_type.upper() == "WIKI" and not params:
            params = {
                "title": query,
                "language": "fa",
                "max_sections": 5,
                "include_references": False
            }
        elif not params:
            params = {"query": query}

        message = {
            "metadata": {
                "request_id": request_id,
                "timestamp": timestamp,
                "source": "self_learning_acquisition",
                "destination": "balance",  # مقصد نهایی درخواست به ماژول Balance جهت هدایت به Data
                "priority": used_priority,
                "request_source": "system"  # یا "user" بسته به منبع درخواست
            },
            "payload": {
                "operation": "FETCH_DATA",
                "model_id": self.model_id,
                "data_type": data_type,
                "data_source": source_type,
                "parameters": params,
                "response_topic": self.response_topic
            }
        }
        self.logger.debug(f"[RequestBuilder] Built request: {message}")
        return message


# Example usage (for testing purposes only – final version is meant for production use)
if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.DEBUG)

    rb = RequestBuilder(model_id="model_123")
    req = rb.build_request(
        query="Artificial Intelligence",
        data_type="TEXT",
        source_type="WIKI"
    )
    print(json.dumps(req, indent=2, ensure_ascii=False))

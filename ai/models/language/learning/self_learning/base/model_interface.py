"""
ModelInterface Module
---------------------
این فایل رابط ارتباطی بین سیستم خودآموزی (Self-Learning) و مدل زبانی عملیاتی را تعریف می‌کند.
این پیاده‌سازی نهایی از طریق API HTTP به سرویس مدل متصل شده و از مکانیزم‌های پیشرفته مانند تایم‌اوت،
لاگینگ دقیق و کنترل استثنا استفاده می‌کند.
"""

import aiohttp
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict


class ModelInterface(ABC):
    """
    رابط پایه برای ارتباط با مدل زبانی.
    این کلاس باید در پیاده‌سازی‌های عملیاتی برای زبان‌ها یا مدل‌های مختلف توسعه یابد.
    """

    @abstractmethod
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست به مدل زبانی و دریافت پاسخ.
        """
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی مدل زبانی (مانند وضعیت بار، سلامت و غیره).
        """
        pass

    @abstractmethod
    async def update_model(self, update_data: Dict[str, Any]) -> bool:
        """
        به‌روزرسانی مدل زبانی بر اساس داده‌های جدید یا تنظیمات بهبود یافته.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        بستن منابع مرتبط (مثلاً اتصال HTTP).
        """
        pass


class OperationalModelInterface(ModelInterface):
    """
    پیاده‌سازی عملیاتی ModelInterface برای ارتباط با سرویس مدل زبانی از طریق API HTTP.

    ویژگی‌ها:
      - استفاده از aiohttp برای انجام درخواست‌های غیرهمزمان با تایم‌اوت تنظیم‌شده.
      - لاگینگ دقیق در سطح DEBUG برای بررسی عملکرد.
      - کنترل استثنا برای افزایش پایداری و قابلیت اطمینان.

    نکته: آدرس سرویس مدل (model_service_url) باید به صورت کامل (با پروتکل http/https) تعیین شود.
    """

    def __init__(self, model_service_url: str, timeout: int = 10):
        self.model_service_url = model_service_url.rstrip("/")
        self.timeout = timeout
        self.logger = logging.getLogger("OperationalModelInterface")
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        self.logger.info(f"[OperationalModelInterface] Initialized with endpoint {self.model_service_url}")

    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست به سرویس مدل زبانی.

        Args:
            request (Dict[str, Any]): داده‌های ورودی برای مدل (مثلاً سوال یا ورودی متنی).

        Returns:
            Dict[str, Any]: پاسخ دریافتی از سرویس مدل.
        """
        url = f"{self.model_service_url}/process"
        self.logger.debug(f"[send_request] Sending POST request to {url} with payload: {request}")
        try:
            async with self.session.post(url, json=request) as response:
                response.raise_for_status()
                data = await response.json()
                self.logger.debug(f"[send_request] Received response: {data}")
                return data
        except Exception as e:
            self.logger.error(f"[send_request] Error sending request: {str(e)}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی سرویس مدل زبانی.

        Returns:
            Dict[str, Any]: دیکشنری شامل وضعیت سرویس (مانند وضعیت بار، سلامت، نسخه و ...).
        """
        url = f"{self.model_service_url}/status"
        self.logger.debug(f"[get_status] Sending GET request to {url}")
        try:
            async with self.session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                self.logger.debug(f"[get_status] Status received: {data}")
                return data
        except Exception as e:
            self.logger.error(f"[get_status] Error getting status: {str(e)}")
            raise

    async def update_model(self, update_data: Dict[str, Any]) -> bool:
        """
        به‌روزرسانی مدل زبانی با داده‌های جدید.

        Args:
            update_data (Dict[str, Any]): داده‌های به‌روزرسانی (مثلاً پارامترهای تنظیم‌شده).

        Returns:
            bool: True در صورت موفقیت‌آمیز بودن به‌روزرسانی.
        """
        url = f"{self.model_service_url}/update"
        self.logger.debug(f"[update_model] Sending POST request to {url} with data: {update_data}")
        try:
            async with self.session.post(url, json=update_data) as response:
                response.raise_for_status()
                data = await response.json()
                self.logger.debug(f"[update_model] Update response: {data}")
                return data.get("success", False)
        except Exception as e:
            self.logger.error(f"[update_model] Error updating model: {str(e)}")
            return False

    async def close(self) -> None:
        """
        بستن جلسه HTTP و آزادسازی منابع.
        """
        self.logger.info("[OperationalModelInterface] Closing HTTP session.")
        await self.session.close()

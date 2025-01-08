# tests/unit/__init__.py
"""
Unit Testing Module
-----------------
این ماژول مسئول تست‌های واحد است که کوچکترین اجزای سیستم را به صورت ایزوله تست می‌کنند.

تست‌های واحد در این بخش شامل موارد زیر می‌شوند:
- تست توابع منفرد
- تست متدهای کلاس‌ها
- تست منطق کسب و کار
- تست رفتار کامپوننت‌های جداگانه

هر تست واحد باید:
1. سریع باشد
2. مستقل از سایر تست‌ها باشد
3. قابل تکرار باشد
4. خودکار باشد
"""

import pytest
from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TestCase:
    """ساختار داده برای نگهداری اطلاعات یک مورد تست"""
    name: str
    inputs: Dict[str, Any]
    expected: Any
    created_at: datetime = datetime.now()
    timeout: Optional[int] = None


class UnitTestConfiguration:
    """تنظیمات پایه برای تست‌های واحد"""

    def __init__(self):
        self.test_data_path = self._setup_test_data_path()
        self.mocks_path = self._setup_mocks_path()
        self.timeout = 2  # حداکثر زمان اجرای هر تست به ثانیه

    def _setup_test_data_path(self) -> Path:
        """آماده‌سازی مسیر داده‌های تست"""
        path = Path(__file__).parent / 'data'
        path.mkdir(exist_ok=True)
        return path

    def _setup_mocks_path(self) -> Path:
        """آماده‌سازی مسیر mock ها"""
        path = Path(__file__).parent / 'mocks'
        path.mkdir(exist_ok=True)
        return path

    def create_test_case(self, name: str, inputs: Dict[str, Any], expected: Any,
                        timeout: Optional[int] = None) -> TestCase:
        """ایجاد یک مورد تست استاندارد"""
        return TestCase(
            name=name,
            inputs=inputs,
            expected=expected,
            timeout=timeout or self.timeout
        )

    def load_test_data(self, filename: str) -> Dict[str, Any]:
        """بارگذاری داده‌های تست از فایل"""
        file_path = self.test_data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Test data file not found: {filename}")
        # پیاده‌سازی بارگذاری داده‌ها
        pass


# ثابت‌های مورد نیاز برای تست‌های واحد
TIMEOUT_FAST = 1    # برای تست‌های ساده و سریع
TIMEOUT_NORMAL = 2  # برای تست‌های معمولی
TIMEOUT_SLOW = 5    # برای تست‌های پیچیده‌تر

__all__ = ['UnitTestConfiguration', 'TestCase', 'TIMEOUT_FAST', 'TIMEOUT_NORMAL', 'TIMEOUT_SLOW']
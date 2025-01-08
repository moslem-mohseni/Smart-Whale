"""
Language Processing Tests Package
-----------------------------
این پکیج شامل تست‌های واحد برای تمام قابلیت‌های پردازش زبان در سیستم است.
از توابع و کلاس‌های کمکی مشترک برای تست زبان‌های مختلف استفاده می‌کند.
"""

import pytest
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# تنظیم لاگر برای تست‌های زبان
logger = logging.getLogger(__name__)

# مسیر پایه برای داده‌های تست
TEST_DATA_ROOT = Path(__file__).parent.parent.parent / 'test_data' / 'language'
TEST_DATA_ROOT.mkdir(parents=True, exist_ok=True)


class LanguageTestHelper:
    """
    کلاس کمکی برای تست‌های پردازش زبان

    این کلاس توابع و ثابت‌های مشترک مورد نیاز برای تست‌های
    پردازش زبان‌های مختلف را فراهم می‌کند.
    """

    @staticmethod
    def get_test_config(language_code: str) -> Dict[str, Any]:
        """
        دریافت تنظیمات پایه برای تست یک زبان خاص

        Args:
            language_code: کد زبان (مثل 'fa' برای فارسی)

        Returns:
            دیکشنری تنظیمات پایه برای تست
        """
        return {
            'storage_path': TEST_DATA_ROOT / language_code,
            'min_confidence': 0.6,
            'learning_rate': 0.1,
            'test_mode': True
        }

    @staticmethod
    def verify_text_processing_result(text: str, result: Any) -> None:
        """
        اعتبارسنجی نتایج پردازش متن

        این تابع بررسی‌های پایه روی نتیجه پردازش متن انجام می‌دهد.

        Args:
            text: متن ورودی
            result: نتیجه پردازش

        Raises:
            AssertionError: اگر نتیجه معتبر نباشد
        """
        assert result is not None, "نتیجه پردازش نباید None باشد"
        assert hasattr(result, 'tokens'), "نتیجه باید شامل توکن‌ها باشد"
        assert len(result.tokens) > 0, "لیست توکن‌ها نباید خالی باشد"
        assert hasattr(result, 'confidence'), "نتیجه باید شامل میزان اطمینان باشد"
        assert 0 <= result.confidence <= 1, "میزان اطمینان باید بین 0 و 1 باشد"


# تنظیمات پیش‌فرض pytest برای تست‌های زبان
def pytest_configure(config):
    """تنظیم پیکربندی pytest"""
    config.addinivalue_line(
        "markers",
        "language: mark test as a language processing test"
    )


# فیکسچر مشترک برای تمام تست‌های زبان
@pytest.fixture
def language_test_helper():
    """فیکسچر برای دسترسی به توابع کمکی تست زبان"""
    return LanguageTestHelper()


@pytest.fixture
def test_data_path():
    """فیکسچر برای دسترسی به مسیر داده‌های تست"""
    return TEST_DATA_ROOT
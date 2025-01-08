# tests/__init__.py
"""
Testing Framework
---------------
این ماژول چارچوب اصلی تست‌های نرم‌افزاری را فراهم می‌کند و شامل سه نوع اصلی تست است:
1. تست‌های واحد (Unit Tests)
2. تست‌های یکپارچگی (Integration Tests)
3. تست‌های end-to-end (E2E Tests)

این چارچوب اطمینان حاصل می‌کند که:
- تمام قابلیت‌های سیستم به درستی تست شده‌اند
- تست‌ها قابل تکرار و قابل اعتماد هستند
- نتایج تست به درستی گزارش می‌شوند
- پوشش کد (Code Coverage) مناسب است
"""

import logging
from typing import Dict, List, Union, Optional
from pathlib import Path
from datetime import datetime
import pytest


class TestManager:
    """
    مدیریت کلی تست‌های نرم‌افزار
    این کلاس مسئول هماهنگی بین انواع مختلف تست و گزارش‌گیری است.
    """

    def __init__(self):
        self.logger = self._setup_logger()
        self.results_dir = self._setup_results_directory()
        self.test_suites: Dict[str, List] = {
            'unit': [],
            'integration': [],
            'e2e': []
        }

    def _setup_logger(self) -> logging.Logger:
        """راه‌اندازی و پیکربندی logger"""
        logger = logging.getLogger('test_manager')
        logger.setLevel(logging.INFO)
        return logger

    def _setup_results_directory(self) -> Path:
        """ایجاد و آماده‌سازی پوشه نتایج"""
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        return results_dir

    def configure_logging(self) -> None:
        """پیکربندی سیستم ثبت لاگ"""
        log_file = self.results_dir / f'test_run_{datetime.now():%Y%m%d_%H%M%S}.log'
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

    async def run_test_suite(self, suite_type: str) -> Dict[str, Union[datetime, List]]:
        """اجرای یک مجموعه تست خاص"""
        if suite_type not in self.test_suites:
            raise ValueError(f"Unknown test suite type: {suite_type}")

        results = {
            'start_time': datetime.now(),
            'tests': []
        }

        try:
            # اجرای pytest برای مجموعه تست مورد نظر
            pytest.main([f'tests/{suite_type}'])
            self.logger.info(f"Completed {suite_type} test suite")
        except Exception as e:
            self.logger.error(f"Error running {suite_type} tests: {str(e)}")
            results['error'] = str(e)

        results['end_time'] = datetime.now()
        return results

    async def run_all_tests(self) -> Dict:
        """اجرای تمام تست‌ها و جمع‌آوری نتایج"""
        results = {
            'start_time': datetime.now(),
            'suites': {}
        }

        for suite in self.test_suites.keys():
            results['suites'][suite] = await self.run_test_suite(suite)

        results['end_time'] = datetime.now()
        return results

    def generate_report(self, results: Dict) -> str:
        """تولید گزارش جامع از نتایج تست"""
        # پیاده‌سازی در نسخه‌های بعدی تکمیل خواهد شد
        pass


# نسخه و ثابت‌های عمومی
__version__ = '0.1.0'
VERSION_INFO = {
    'major': 0,
    'minor': 1,
    'patch': 0,
    'release': 'alpha'
}

__all__ = ['TestManager']
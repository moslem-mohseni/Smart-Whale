# scripts/run_tests.py
"""
پکیج: scripts.run_tests
توضیحات: اسکریپت اجرای تست‌های پروژه
نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

import pytest
import sys
from pathlib import Path


def main():
    """اجرای تست‌های پروژه"""
    # اضافه کردن مسیر اصلی پروژه به PYTHONPATH
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    # اجرای تست‌ها
    pytest.main(['tests', '-v'])


if __name__ == '__main__':
    main()
# persian/language_processors/proverbs/proverb_services.py
"""
ماژول proverb_services.py

این ماژول توابع خدمات تکمیلی مربوط به پشتیبان‌گیری و بازیابی داده‌ها، مدیریت تاریخچه جستجو و سایر خدمات جانبی را فراهم می‌کند.
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ProverbServices:
    def __init__(self, backup_dir: str = "backup", search_history_file: str = "search_history.json"):
        """
        سازنده ProverbServices.

        Args:
            backup_dir (str): پوشه‌ای که فایل‌های پشتیبان در آن ذخیره می‌شوند.
            search_history_file (str): نام فایل تاریخچه جستجو.
        """
        self.backup_dir = backup_dir
        self.search_history_file = search_history_file

        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            logger.info(f"پوشه پشتیبان '{self.backup_dir}' ایجاد شد.")

    def backup_data(self, data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        پشتیبان‌گیری از داده‌های ضرب‌المثل.

        Args:
            data (Dict[str, Any]): داده‌هایی که باید پشتیبان‌گیری شوند.
            filename (Optional[str]): نام فایل پشتیبان (اختیاری).

        Returns:
            str: مسیر فایل پشتیبان.
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"proverbs_backup_{timestamp}.json"
        file_path = os.path.join(self.backup_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"پشتیبان‌گیری با موفقیت انجام شد: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"خطا در پشتیبان‌گیری: {e}")
            raise

    def restore_data(self, file_path: str) -> Dict[str, Any]:
        """
        بازیابی داده‌های ضرب‌المثل از فایل پشتیبان.

        Args:
            file_path (str): مسیر فایل پشتیبان.

        Returns:
            Dict[str, Any]: داده‌های بازیابی‌شده به صورت دیکشنری.
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"فایل پشتیبان در مسیر {file_path} یافت نشد.")
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"داده‌ها با موفقیت از {file_path} بازیابی شدند.")
            return data
        except Exception as e:
            logger.error(f"خطا در بازیابی داده‌ها: {e}")
            raise

    def record_search_history(self, query: str, results: Dict[str, Any]) -> None:
        """
        ثبت تاریخچه جستجو.

        Args:
            query (str): عبارت جستجو.
            results (Dict[str, Any]): نتایج جستجو.
        """
        history_entry = {
            "timestamp": time.time(),
            "query": query,
            "results": results
        }
        try:
            if os.path.exists(self.search_history_file):
                with open(self.search_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(history_entry)
            with open(self.search_history_file, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info("تاریخچه جستجو ثبت شد.")
        except Exception as e:
            logger.error(f"خطا در ثبت تاریخچه جستجو: {e}")

    def get_search_history(self) -> Dict[str, Any]:
        """
        دریافت تاریخچه جستجو.

        Returns:
            Dict[str, Any]: تاریخچه جستجو به صورت دیکشنری.
        """
        try:
            if os.path.exists(self.search_history_file):
                with open(self.search_history_file, "r", encoding="utf-8") as f:
                    history = json.load(f)
                logger.info("تاریخچه جستجو بازیابی شد.")
                return history
            else:
                logger.info("تاریخچه جستجو وجود ندارد.")
                return {}
        except Exception as e:
            logger.error(f"خطا در دریافت تاریخچه جستجو: {e}")
            return {}

if __name__ == "__main__":
    services = ProverbServices()
    sample_data = {"proverbs": {"p_1": {"proverb": "هر که بامش بیش برفش بیشتر"}}}
    backup_file = services.backup_data(sample_data)
    restored = services.restore_data(backup_file)
    print("Restored data:", json.dumps(restored, ensure_ascii=False, indent=2))
    services.record_search_history("بام", {"results": ["p_1"]})
    history = services.get_search_history()
    print("Search history:", json.dumps(history, ensure_ascii=False, indent=2))

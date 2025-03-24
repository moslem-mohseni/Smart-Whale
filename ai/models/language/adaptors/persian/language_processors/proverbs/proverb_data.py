# persian/language_processors/proverbs/proverb_data.py
"""
ماژول proverb_data.py

این ماژول شامل کلاس ProverbDataAccess است که وظیفه دسترسی به داده‌های ضرب‌المثل، نسخه‌ها و کلمات کلیدی را از پایگاه داده (ClickHouseDB) بر عهده دارد.
متدهای اصلی این کلاس شامل:
  - load_proverbs: بارگذاری ضرب‌المثل‌ها
  - store_proverb: ذخیره یا به‌روزرسانی ضرب‌المثل
  - load_variants: بارگذاری نسخه‌های ضرب‌المثل
  - store_variant: ذخیره یا به‌روزرسانی نسخه ضرب‌المثل
  - load_keywords: بارگذاری کلمات کلیدی ضرب‌المثل
  - store_keyword: ذخیره یا به‌روزرسانی کلمه کلیدی
  - همچنین توابع کمکی مانند array_to_sql برای تبدیل آرایه‌ها به فرمت SQL مناسب فراهم شده است.
"""

import json
import logging
import re
import time
from typing import Dict, List, Any

from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ProverbDataAccess:
    def __init__(self):
        try:
            self.database = ClickHouseDB()
            logger.info("ProverbDataAccess: اتصال به پایگاه داده برقرار شد.")
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در اتصال به پایگاه داده: {e}")
            raise

    def array_to_sql(self, array: List[str]) -> str:
        """تبدیل آرایه به فرمت مناسب SQL برای ClickHouse"""
        if not array:
            return "[]"
        escaped_items = [item.replace("'", "''") for item in array]
        items_str = "', '".join(escaped_items)
        return f"['{items_str}']"

    def load_proverbs(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری ضرب‌المثل‌ها از پایگاه داده.
        Returns:
            دیکشنری ضرب‌المثل‌ها با کلید proverb_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM proverbs")
            if result and len(result) > 0:
                proverbs = {row['proverb_id']: row for row in result}
                logger.info(f"ProverbDataAccess: {len(proverbs)} ضرب‌المثل بازیابی شدند.")
                return proverbs
            else:
                logger.info("ProverbDataAccess: هیچ ضرب‌المثلی یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در بارگذاری ضرب‌المثل‌ها: {e}")
            return {}

    def store_proverb(self, proverb_data: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی ضرب‌المثل در پایگاه داده.
        Args:
            proverb_data: دیکشنری اطلاعات ضرب‌المثل.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT proverb_id FROM proverbs WHERE proverb_id='{proverb_data['proverb_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("proverbs", proverb_data)
                logger.debug(f"ProverbDataAccess: ضرب‌المثل جدید ذخیره شد: {proverb_data['proverb_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE proverbs
                    SET meaning = '{proverb_data.get('meaning', '').replace("'", "''")}',
                        category = '{proverb_data.get('category', '')}',
                        formality = '{proverb_data.get('formality', '')}',
                        examples = {self.array_to_sql(proverb_data.get('examples', []))},
                        confidence = {proverb_data.get('confidence', 0)},
                        usage_count = usage_count + 1
                    WHERE proverb_id = '{proverb_data['proverb_id']}'
                """)
                logger.debug(f"ProverbDataAccess: ضرب‌المثل موجود به‌روزرسانی شد: {proverb_data['proverb_id']}")
            return True
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در ذخیره ضرب‌المثل: {e}")
            return False

    def load_variants(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری نسخه‌های ضرب‌المثل از پایگاه داده.
        Returns:
            دیکشنری نسخه‌ها با کلید variant_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM proverb_variants")
            if result and len(result) > 0:
                variants = {row['variant_id']: row for row in result}
                logger.info(f"ProverbDataAccess: {len(variants)} نسخه بازیابی شدند.")
                return variants
            else:
                logger.info("ProverbDataAccess: هیچ نسخه‌ای یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در بارگذاری نسخه‌ها: {e}")
            return {}

    def store_variant(self, variant_data: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی نسخه ضرب‌المثل در پایگاه داده.
        Args:
            variant_data: دیکشنری اطلاعات نسخه.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT variant_id FROM proverb_variants WHERE variant_id='{variant_data['variant_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("proverb_variants", variant_data)
                logger.debug(f"ProverbDataAccess: نسخه جدید ذخیره شد: {variant_data['variant_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE proverb_variants
                    SET formality = '{variant_data.get('formality', '')}',
                        confidence = {variant_data.get('confidence', 0)},
                        usage_count = usage_count + 1
                    WHERE variant_id = '{variant_data['variant_id']}'
                """)
                logger.debug(f"ProverbDataAccess: نسخه موجود به‌روزرسانی شد: {variant_data['variant_id']}")
            return True
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در ذخیره نسخه: {e}")
            return False

    def load_keywords(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری کلمات کلیدی ضرب‌المثل از پایگاه داده.
        Returns:
            دیکشنری کلمات کلیدی با کلید keyword.
        """
        try:
            result = self.database.execute_query("SELECT * FROM proverb_keywords")
            if result and len(result) > 0:
                keywords = {row['keyword']: row for row in result}
                logger.info(f"ProverbDataAccess: {len(keywords)} کلمه کلیدی بازیابی شدند.")
                return keywords
            else:
                logger.info("ProverbDataAccess: هیچ کلمه کلیدی یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در بارگذاری کلمات کلیدی: {e}")
            return {}

    def store_keyword(self, keyword_data: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی یک کلمه کلیدی ضرب‌المثل.
        Args:
            keyword_data: دیکشنری اطلاعات کلمه کلیدی شامل keyword، proverb_ids، weight، usage_count.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            result = self.database.execute_query(
                f"SELECT keyword FROM proverb_keywords WHERE keyword='{keyword_data['keyword']}' LIMIT 1"
            )
            if not result or len(result) == 0:
                self.database.insert_data("proverb_keywords", keyword_data)
                logger.debug(f"ProverbDataAccess: کلمه کلیدی جدید ذخیره شد: {keyword_data['keyword']}")
            else:
                self.database.execute_query(f"""
                    UPDATE proverb_keywords
                    SET proverb_ids = {self.array_to_sql(keyword_data.get('proverb_ids', []))},
                        usage_count = usage_count + 1
                    WHERE keyword = '{keyword_data['keyword']}'
                """)
                logger.debug(f"ProverbDataAccess: کلمه کلیدی موجود به‌روزرسانی شد: {keyword_data['keyword']}")
            return True
        except Exception as e:
            logger.error(f"ProverbDataAccess: خطا در ذخیره کلمه کلیدی: {e}")
            return False

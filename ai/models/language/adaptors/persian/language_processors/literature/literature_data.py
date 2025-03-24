# persian/language_processors/literature/literature_data.py
"""
ماژول literature_data.py

این ماژول شامل کلاس LiteratureDataAccess است که وظیفه دسترسی به داده‌های ادبی (دوره‌های ادبی، سبک‌های ادبی،
آرایه‌های ادبی، وزن‌های شعری، متون ادبی و نویسندگان مشهور) را از پایگاه داده (ClickHouseDB) بر عهده دارد.
متدهای اصلی:
  - load_literary_periods: بارگذاری دوره‌های ادبی
  - store_literary_period: ذخیره یا به‌روزرسانی دوره ادبی
  - load_literary_styles: بارگذاری سبک‌های ادبی
  - store_literary_style: ذخیره یا به‌روزرسانی سبک ادبی
  - load_literary_devices: بارگذاری آرایه‌های ادبی
  - store_literary_device: ذخیره یا به‌روزرسانی آرایه ادبی
  - load_poetry_meters: بارگذاری وزن‌های شعری
  - store_poetry_meter: ذخیره یا به‌روزرسانی وزن شعری
  - load_literary_corpus: بارگذاری متون ادبی
  - store_literary_corpus_item: ذخیره یا به‌روزرسانی متن ادبی
  - load_famous_authors: بارگذاری نویسندگان مشهور
  - store_famous_author: ذخیره یا به‌روزرسانی نویسنده مشهور
"""

import json
import logging
import re
import time
from typing import Dict, List, Any

from ai.models.language.infrastructure.clickhouse.clickhouse_adapter import ClickHouseDB

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteratureDataAccess:
    def __init__(self):
        try:
            self.database = ClickHouseDB()
            logger.info("LiteratureDataAccess: اتصال به پایگاه داده برقرار شد.")
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در اتصال به پایگاه داده: {e}")
            raise

    def array_to_sql(self, array: List[str]) -> str:
        """
        تبدیل آرایه به فرمت مناسب SQL برای ClickHouse
        """
        if not array:
            return "[]"
        escaped_items = [item.replace("'", "''") for item in array]
        items_str = "', '".join(escaped_items)
        return f"['{items_str}']"

    def load_literary_periods(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری دوره‌های ادبی از پایگاه داده.
        Returns:
            دیکشنری دوره‌های ادبی با کلید period_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM literary_periods")
            if result and len(result) > 0:
                periods = {row['period_id']: row for row in result}
                logger.info(f"LiteratureDataAccess: {len(periods)} دوره ادبی بازیابی شدند.")
                return periods
            else:
                logger.info("LiteratureDataAccess: هیچ دوره ادبی یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در بارگذاری دوره‌های ادبی: {e}")
            return {}

    def store_literary_period(self, period: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی دوره ادبی در پایگاه داده.
        Args:
            period: دیکشنری اطلاعات دوره ادبی.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT period_id FROM literary_periods WHERE period_id='{period['period_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("literary_periods", period)
                logger.debug(f"LiteratureDataAccess: دوره ادبی جدید ذخیره شد: {period['period_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE literary_periods
                    SET period_name = '{period.get('period_name', '').replace("'", "''")}',
                        period_code = '{period.get('period_code', '')}',
                        start_year = {period.get('start_year', 0)},
                        end_year = {period.get('end_year', 0)},
                        description = '{period.get('description', '').replace("'", "''")}',
                        key_characteristics = {self.array_to_sql(period.get('key_characteristics', []))},
                        notable_authors = {self.array_to_sql(period.get('notable_authors', []))},
                        popularity = {period.get('popularity', 0)}
                    WHERE period_id = '{period['period_id']}'
                """)
                logger.debug(f"LiteratureDataAccess: دوره ادبی موجود به‌روزرسانی شد: {period['period_id']}")
            return True
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در ذخیره دوره ادبی: {e}")
            return False

    def load_literary_styles(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری سبک‌های ادبی از پایگاه داده.
        Returns:
            دیکشنری سبک‌های ادبی با کلید style_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM literary_styles")
            if result and len(result) > 0:
                styles = {row['style_id']: row for row in result}
                logger.info(f"LiteratureDataAccess: {len(styles)} سبک ادبی بازیابی شدند.")
                return styles
            else:
                logger.info("LiteratureDataAccess: هیچ سبک ادبی یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در بارگذاری سبک‌های ادبی: {e}")
            return {}

    def store_literary_style(self, style: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی سبک ادبی در پایگاه داده.
        Args:
            style: دیکشنری اطلاعات سبک ادبی.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT style_id FROM literary_styles WHERE style_id='{style['style_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("literary_styles", style)
                logger.debug(f"LiteratureDataAccess: سبک ادبی جدید ذخیره شد: {style['style_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE literary_styles
                    SET style_name = '{style.get('style_name', '').replace("'", "''")}',
                        style_code = '{style.get('style_code', '')}',
                        period_id = '{style.get('period_id', '')}',
                        description = '{style.get('description', '').replace("'", "''")}',
                        key_characteristics = {self.array_to_sql(style.get('key_characteristics', []))},
                        notable_authors = {self.array_to_sql(style.get('notable_authors', []))},
                        popularity = {style.get('popularity', 0)}
                    WHERE style_id = '{style['style_id']}'
                """)
                logger.debug(f"LiteratureDataAccess: سبک ادبی موجود به‌روزرسانی شد: {style['style_id']}")
            return True
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در ذخیره سبک ادبی: {e}")
            return False

    def load_literary_devices(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری آرایه‌های ادبی از پایگاه داده.
        Returns:
            دیکشنری آرایه‌های ادبی با کلید device_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM literary_devices")
            if result and len(result) > 0:
                devices = {row['device_id']: row for row in result}
                logger.info(f"LiteratureDataAccess: {len(devices)} آرایه ادبی بازیابی شدند.")
                return devices
            else:
                logger.info("LiteratureDataAccess: هیچ آرایه ادبی یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در بارگذاری آرایه‌های ادبی: {e}")
            return {}

    def store_literary_device(self, device: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی آرایه ادبی در پایگاه داده.
        Args:
            device: دیکشنری اطلاعات آرایه ادبی.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT device_id FROM literary_devices WHERE device_id='{device['device_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("literary_devices", device)
                logger.debug(f"LiteratureDataAccess: آرایه ادبی جدید ذخیره شد: {device['device_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE literary_devices
                    SET device_name = '{device.get('device_name', '').replace("'", "''")}',
                        device_code = '{device.get('device_code', '')}',
                        device_type = '{device.get('device_type', '')}',
                        description = '{device.get('description', '').replace("'", "''")}',
                        detection_pattern = '{device.get('detection_pattern', '').replace("'", "''")}',
                        examples = {self.array_to_sql(device.get('examples', []))},
                        confidence = {device.get('confidence', 0)},
                        usage_count = usage_count + 1
                    WHERE device_id = '{device['device_id']}'
                """)
                logger.debug(f"LiteratureDataAccess: آرایه ادبی موجود به‌روزرسانی شد: {device['device_id']}")
            return True
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در ذخیره آرایه ادبی: {e}")
            return False

    def load_poetry_meters(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری وزن‌های شعری از پایگاه داده.
        Returns:
            دیکشنری وزن‌های شعری با کلید meter_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM poetry_meters")
            if result and len(result) > 0:
                meters = {row['meter_id']: row for row in result}
                logger.info(f"LiteratureDataAccess: {len(meters)} وزن شعری بازیابی شدند.")
                return meters
            else:
                logger.info("LiteratureDataAccess: هیچ وزن شعری یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در بارگذاری وزن‌های شعری: {e}")
            return {}

    def store_poetry_meter(self, meter: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی وزن شعری در پایگاه داده.
        Args:
            meter: دیکشنری اطلاعات وزن شعری.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT meter_id FROM poetry_meters WHERE meter_id='{meter['meter_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("poetry_meters", meter)
                logger.debug(f"LiteratureDataAccess: وزن شعری جدید ذخیره شد: {meter['meter_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE poetry_meters
                    SET meter_name = '{meter.get('meter_name', '').replace("'", "''")}',
                        meter_pattern = '{meter.get('meter_pattern', '').replace("'", "''")}',
                        transliteration = '{meter.get('transliteration', '').replace("'", "''")}',
                        description = '{meter.get('description', '').replace("'", "''")}',
                        examples = {self.array_to_sql(meter.get('examples', []))},
                        confidence = {meter.get('confidence', 0)},
                        usage_count = usage_count + 1
                    WHERE meter_id = '{meter['meter_id']}'
                """)
                logger.debug(f"LiteratureDataAccess: وزن شعری موجود به‌روزرسانی شد: {meter['meter_id']}")
            return True
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در ذخیره وزن شعری: {e}")
            return False

    def load_literary_corpus(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری متون ادبی از پایگاه داده.
        Returns:
            دیکشنری متون ادبی با کلید text_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM literary_corpus ORDER BY discovery_time DESC LIMIT 1000")
            if result and len(result) > 0:
                corpus = {row['text_id']: row for row in result}
                logger.info(f"LiteratureDataAccess: {len(corpus)} متون ادبی بازیابی شدند.")
                return corpus
            else:
                logger.info("LiteratureDataAccess: هیچ متنی در corpus یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در بارگذاری متون ادبی: {e}")
            return {}

    def store_literary_corpus_item(self, corpus_item: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی یک متن ادبی در پایگاه داده.
        Args:
            corpus_item: دیکشنری اطلاعات متن ادبی.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            text_id = corpus_item.get("text_id", "")
            existing = self.database.execute_query(
                f"SELECT text_id FROM literary_corpus WHERE text_id='{text_id}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("literary_corpus", corpus_item)
                logger.debug(f"LiteratureDataAccess: متن ادبی جدید ذخیره شد: {text_id}")
            else:
                self.database.execute_query(f"""
                    UPDATE literary_corpus
                    SET text = '{corpus_item.get('text','').replace("'", "''")}',
                        title = '{corpus_item.get('title','').replace("'", "''")}',
                        author = '{corpus_item.get('author','').replace("'", "''")}',
                        period_id = '{corpus_item.get('period_id','')}',
                        style_id = '{corpus_item.get('style_id','')}',
                        year = {corpus_item.get('year', 0)},
                        text_type = '{corpus_item.get('text_type','')}',
                        devices = {self.array_to_sql(corpus_item.get('devices', []))},
                        meter_id = '{corpus_item.get('meter_id','')}',
                        rhyme_pattern = '{corpus_item.get('rhyme_pattern','').replace("'", "''")}'
                    WHERE text_id = '{text_id}'
                """)
                logger.debug(f"LiteratureDataAccess: متن ادبی موجود به‌روزرسانی شد: {text_id}")
            return True
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در ذخیره متن ادبی: {e}")
            return False

    def load_famous_authors(self) -> Dict[str, Dict[str, Any]]:
        """
        بارگذاری نویسندگان مشهور از پایگاه داده.
        Returns:
            دیکشنری نویسندگان مشهور با کلید author_id.
        """
        try:
            result = self.database.execute_query("SELECT * FROM famous_authors")
            if result and len(result) > 0:
                authors = {row['author_id']: row for row in result}
                logger.info(f"LiteratureDataAccess: {len(authors)} نویسنده مشهور بازیابی شدند.")
                return authors
            else:
                logger.info("LiteratureDataAccess: هیچ نویسنده مشهوری یافت نشد.")
                return {}
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در بارگذاری نویسندگان مشهور: {e}")
            return {}

    def store_famous_author(self, author: Dict[str, Any]) -> bool:
        """
        ذخیره یا به‌روزرسانی نویسنده مشهور در پایگاه داده.
        Args:
            author: دیکشنری اطلاعات نویسنده.
        Returns:
            True در صورت موفقیت، False در غیر این صورت.
        """
        try:
            existing = self.database.execute_query(
                f"SELECT author_id FROM famous_authors WHERE author_id='{author['author_id']}' LIMIT 1"
            )
            if not existing or len(existing) == 0:
                self.database.insert_data("famous_authors", author)
                logger.debug(f"LiteratureDataAccess: نویسنده جدید ذخیره شد: {author['author_id']}")
            else:
                self.database.execute_query(f"""
                    UPDATE famous_authors
                    SET author_name = '{author.get('author_name','').replace("'", "''")}',
                        birth_year = {author.get('birth_year', 0)},
                        death_year = {author.get('death_year', 0)},
                        periods = {self.array_to_sql(author.get('periods', []))},
                        styles = {self.array_to_sql(author.get('styles', []))},
                        notable_works = {self.array_to_sql(author.get('notable_works', []))},
                        literary_forms = {self.array_to_sql(author.get('literary_forms', []))},
                        biography = '{author.get('biography','').replace("'", "''")}'
                    WHERE author_id = '{author['author_id']}'
                """)
                logger.debug(f"LiteratureDataAccess: نویسنده موجود به‌روزرسانی شد: {author['author_id']}")
            return True
        except Exception as e:
            logger.error(f"LiteratureDataAccess: خطا در ذخیره نویسنده مشهور: {e}")
            return False


if __name__ == "__main__":
    lda = LiteratureDataAccess()
    periods = lda.load_literary_periods()
    print("Literary Periods:", json.dumps(periods, ensure_ascii=False, indent=2))

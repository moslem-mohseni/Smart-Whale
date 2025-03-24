import asyncio
import json
import logging
from typing import Dict, Any, Optional
import wikipediaapi as wikiapi
from ai.data.collectors.base.collector import BaseCollector
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.redis.config.settings import RedisConfig

logging.basicConfig(level=logging.INFO)


class WikiCollector(BaseCollector):
    """
    جمع‌آورنده اطلاعات از ویکی‌پدیا با پشتیبانی از کش‌گذاری در Redis
    """

    def __init__(self, language="fa", max_length=5000, cache_enabled=True):
        """
        مقداردهی اولیه جمع‌آورنده ویکی‌پدیا

        :param language: زبان مورد نظر (پیش‌فرض: فارسی)
        :param max_length: حداکثر طول متن استخراج‌شده
        :param cache_enabled: فعال‌سازی کش‌گذاری
        """
        super().__init__(source_name=f"Wikipedia-{language}")
        self.user_agent = "SmartWhaleBot/1.0"
        self.language = language
        self.max_length = max_length
        self.cache_enabled = cache_enabled
        self.wiki = wikiapi.Wikipedia(language=language, user_agent=self.user_agent)

        # راه‌اندازی سرویس کش Redis
        if cache_enabled:
            self.cache_service = None  # در متد connect مقداردهی می‌شود
            self.cache_ttl = 3600 * 24  # مدت اعتبار کش (یک روز)

        self.title = None  # عنوان مقاله مورد جستجو

    async def connect(self):
        """
        اتصال به سرویس‌های مورد نیاز
        """
        if self.cache_enabled and self.cache_service is None:
            redis_config = RedisConfig()
            self.cache_service = CacheService(redis_config)
            await self.cache_service.connect()
            logging.info("✅ اتصال به سرویس کش Redis برقرار شد.")

    async def disconnect(self):
        """
        قطع اتصال از سرویس‌ها
        """
        if self.cache_service:
            await self.cache_service.disconnect()
            logging.info("🔌 اتصال از سرویس کش Redis قطع شد.")

    async def collect_data(self) -> Optional[Dict[str, Any]]:
        """
        جمع‌آوری داده از ویکی‌پدیا

        :return: دیکشنری حاوی اطلاعات ویکی‌پدیا یا None در صورت خطا
        """
        if not self.title:
            logging.error("❌ عنوان مقاله برای جستجو مشخص نشده است.")
            return None

        try:
            # اتصال به سرویس‌های مورد نیاز
            await self.connect()

            # جمع‌آوری داده
            text_content = await self.get_page_text(self.title)
            if not text_content:
                logging.warning(f"⚠ محتوایی برای '{self.title}' پیدا نشد.")
                return None

            return {
                "source": "wikipedia",
                "title": self.title,
                "language": self.language,
                "content": text_content,
                "length": len(text_content),
                "timestamp": self._get_current_timestamp()
            }
        except Exception as e:
            logging.error(f"❌ خطا در جمع‌آوری داده از ویکی‌پدیا: {e}")
            return None
        finally:
            # قطع اتصال از سرویس‌ها در پایان
            await self.disconnect()

    async def get_page_text(self, title: str) -> Optional[str]:
        """
        دریافت متن صفحه از ویکی‌پدیا با پشتیبانی از کش‌گذاری

        :param title: عنوان صفحه ویکی‌پدیا
        :return: متن استخراج‌شده یا None در صورت خطا
        """
        cache_key = f"wiki:{self.language}:{title}"

        try:
            # بررسی کش
            if self.cache_enabled and self.cache_service:
                cached_data = await self.cache_service.get(cache_key)
                if cached_data:
                    logging.info(f"✅ محتوای '{title}' از کش Redis بارگذاری شد.")
                    return cached_data

            # دریافت از ویکی‌پدیا
            raw_text = self._fetch_wikipedia_page(title)
            if not raw_text:
                return None

            text = raw_text[:self.max_length]

            # ذخیره در کش
            if self.cache_enabled and self.cache_service:
                await self.cache_service.set(cache_key, text, ttl=self.cache_ttl)
                logging.info(f"✅ محتوای '{title}' در کش Redis ذخیره شد.")

            return text
        except Exception as e:
            logging.error(f"❌ خطا در دریافت محتوای '{title}': {e}")
            return None

    def _fetch_wikipedia_page(self, title: str) -> Optional[str]:
        """
        دریافت متن خام از ویکی‌پدیا

        :param title: عنوان صفحه
        :return: متن خام یا None در صورت خطا
        """
        page = self.wiki.page(title)
        if not page.exists():
            logging.warning(f"⚠ صفحه '{title}' در ویکی‌پدیا یافت نشد.")
            return None
        return page.text

    def _get_current_timestamp(self) -> str:
        """
        تولید زمان فعلی برای ثبت در داده‌ها

        :return: رشته زمان
        """
        from datetime import datetime
        return datetime.now().isoformat()

    def set_title(self, title: str):
        """
        تنظیم عنوان برای جستجو

        :param title: عنوان مقاله ویکی‌پدیا
        """
        self.title = title
        return self


# تست مستقیم کلاس
if __name__ == "__main__":
    import sys


    async def run_test():
        # دریافت عنوان از آرگومان خط فرمان یا استفاده از مقدار پیش‌فرض
        title = sys.argv[1] if len(sys.argv) > 1 else "هوش مصنوعی"

        collector = WikiCollector(language="fa", max_length=3000, cache_enabled=True)
        collector.set_title(title)

        print(f"🔍 جستجوی مقاله '{title}' در ویکی‌پدیا...")
        result = await collector.collect_data()

        if result:
            print("\n✅ نتیجه جمع‌آوری:")
            print(f"عنوان: {result['title']}")
            print(f"زبان: {result['language']}")
            print(f"طول محتوا: {result['length']} کاراکتر")
            print(f"زمان: {result['timestamp']}")
            print("\nبخشی از محتوا:")
            print(result['content'][:500] + "...\n")
        else:
            print("❌ جمع‌آوری اطلاعات ناموفق بود.")


    # اجرای تست
    asyncio.run(run_test())
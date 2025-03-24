# infrastructure/clickhouse/service/analytics_cache.py
import logging
import hashlib
import json
from typing import Optional, Any, Dict, Union
from ..config.config import config
from ..exceptions import OperationalError
from ...redis.adapters.redis_adapter import RedisAdapter
from ...redis.config.settings import RedisConfig

logger = logging.getLogger(__name__)


class AnalyticsCache:
    """
    مدیریت کش برای نتایج تحلیلی ClickHouse

    این کلاس از Redis برای کش کردن نتایج کوئری‌های تحلیلی استفاده می‌کند
    و امکان ذخیره، بازیابی و حذف نتایج را فراهم می‌کند.
    """

    def __init__(self, redis_config: Optional[RedisConfig] = None):
        """
        مقداردهی اولیه کش تحلیلی

        Args:
            redis_config (RedisConfig, optional): تنظیمات Redis
                اگر ارائه نشود، یک نمونه پیش‌فرض ایجاد می‌شود.
        """
        self.redis_config = redis_config or RedisConfig()
        self._adapter = RedisAdapter(self.redis_config)
        self._connected = False

        # دریافت TTL پیش‌فرض از تنظیمات مرکزی
        monitoring_config = config.get_monitoring_config()
        self.default_ttl = int(monitoring_config.get("cache_ttl", 3600))

        logger.info(f"Analytics Cache initialized with default TTL: {self.default_ttl}s")

    async def connect(self) -> None:
        """
        برقراری اتصال به Redis

        Raises:
            OperationalError: در صورت بروز خطا در اتصال به Redis
        """
        if self._connected:
            return

        try:
            await self._adapter.connect()
            self._connected = True
            logger.debug("Successfully connected to Redis cache")
        except Exception as e:
            error_msg = f"Failed to connect to Redis cache: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE601")

    async def disconnect(self) -> None:
        """
        قطع اتصال از Redis
        """
        if not self._connected:
            return

        try:
            await self._adapter.disconnect()
            self._connected = False
            logger.debug("Disconnected from Redis cache")
        except Exception as e:
            logger.warning(f"Error disconnecting from Redis cache: {str(e)}")

    async def _ensure_connected(self) -> None:
        """
        اطمینان از برقراری اتصال به Redis

        Raises:
            OperationalError: اگر اتصال به Redis برقرار نباشد و تلاش برای اتصال ناموفق باشد
        """
        if not self._connected:
            await self.connect()

    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        تولید کلید یکتا برای ذخیره‌سازی کوئری در کش

        از ترکیب کوئری و پارامترهای آن یک هش SHA-256 ایجاد می‌کند
        که به عنوان کلید کش استفاده می‌شود.

        Args:
            query (str): متن کوئری
            params (Dict[str, Any], optional): پارامترهای کوئری

        Returns:
            str: کلید کش
        """
        # ایجاد یک رشته ترکیبی از کوئری و پارامترها
        key_data = query
        if params:
            try:
                # مرتب‌سازی کلیدها برای ثبات کلید کش
                key_data += json.dumps(params, sort_keys=True)
            except (TypeError, ValueError):
                logger.warning("Failed to serialize params for cache key")

        return f"analytics_cache:{hashlib.sha256(key_data.encode()).hexdigest()}"

    async def get_cached_result(self, query: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        دریافت نتیجه کش شده برای یک کوئری تحلیلی

        Args:
            query (str): متن کوئری
            params (Dict[str, Any], optional): پارامترهای کوئری

        Returns:
            Any: نتیجه کش شده یا None در صورت عدم وجود

        Raises:
            OperationalError: در صورت بروز خطا در دسترسی به کش
        """
        try:
            await self._ensure_connected()

            cache_key = self._generate_cache_key(query, params)
            cached_data = await self._adapter.get(cache_key)

            if not cached_data:
                logger.debug(f"Cache miss for query: {query[:50]}...")
                return None

            # تبدیل JSON به دیکشنری
            try:
                result = json.loads(cached_data)
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return result
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON data in cache for key: {cache_key}")
                # حذف خودکار داده نامعتبر
                await self._adapter.delete(cache_key)
                return None

        except OperationalError:
            # خطاهای عملیاتی را منتقل می‌کنیم
            raise
        except Exception as e:
            error_msg = f"Error retrieving data from cache: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE603")

    async def set_cached_result(self, query: str, result: Any,
                                ttl: Optional[int] = None,
                                params: Optional[Dict[str, Any]] = None) -> None:
        """
        ذخیره نتیجه کوئری تحلیلی در کش

        Args:
            query (str): متن کوئری
            result (Any): نتیجه کوئری
            ttl (int, optional): زمان انقضا (ثانیه)
            params (Dict[str, Any], optional): پارامترهای کوئری

        Raises:
            OperationalError: در صورت بروز خطا در ذخیره‌سازی در کش
        """
        try:
            await self._ensure_connected()

            cache_key = self._generate_cache_key(query, params)
            ttl_value = ttl if ttl is not None else self.default_ttl

            # تبدیل نتیجه به JSON
            try:
                serialized_result = json.dumps(result)
            except (TypeError, ValueError) as e:
                logger.warning(f"Failed to serialize result to JSON: {str(e)}")
                return

            await self._adapter.set(cache_key, serialized_result, ttl_value)
            logger.debug(f"Cached result for query with TTL: {ttl_value}s")

        except OperationalError:
            # خطاهای عملیاتی را منتقل می‌کنیم
            raise
        except Exception as e:
            error_msg = f"Error caching result: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE604")

    async def invalidate_cache(self, query: Optional[str] = None,
                               params: Optional[Dict[str, Any]] = None) -> None:
        """
        حذف کش یک کوئری خاص یا پاکسازی کل کش

        Args:
            query (str, optional): متن کوئری. اگر None باشد، کل کش پاک می‌شود.
            params (Dict[str, Any], optional): پارامترهای کوئری

        Raises:
            OperationalError: در صورت بروز خطا در حذف کش
        """
        try:
            await self._ensure_connected()

            if query:
                # حذف کش یک کوئری خاص
                cache_key = self._generate_cache_key(query, params)
                await self._adapter.delete(cache_key)
                logger.debug(f"Invalidated cache key for query: {query[:50]}...")
            else:
                # پاکسازی کل کش
                await self._adapter.flush()
                logger.info("Analytics cache cleared successfully")

        except OperationalError:
            # خطاهای عملیاتی را منتقل می‌کنیم
            raise
        except Exception as e:
            error_msg = f"Error invalidating cache: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE605")

    async def get_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار کش

        Returns:
            Dict[str, Any]: آمار مربوط به کش

        Raises:
            OperationalError: در صورت بروز خطا در دریافت آمار
        """
        try:
            await self._ensure_connected()

            # تعداد کلیدها
            key_count = 0
            if hasattr(self._adapter, 'dbsize'):
                key_count = await self._adapter.dbsize()

            stats = {
                "key_count": key_count,
                "default_ttl": self.default_ttl,
                "connected": self._connected
            }

            return stats

        except Exception as e:
            error_msg = f"Error getting cache stats: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE607")

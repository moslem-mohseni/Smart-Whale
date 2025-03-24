# infrastructure/clickhouse/service/analytics_service.py
import logging
from typing import Optional, List, Dict, Any, Union
from ..config.config import config
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..optimization.query_optimizer import QueryOptimizer
from ..domain.models import AnalyticsQuery, AnalyticsResult
from ..exceptions import QueryError, OperationalError
from .analytics_cache import AnalyticsCache

logger = logging.getLogger(__name__)


class AnalyticsService:
    """
    سرویس تحلیل داده‌ها در ClickHouse با پشتیبانی از کش و بهینه‌سازی کوئری‌ها

    این سرویس مسئول اجرای کوئری‌های تحلیلی در ClickHouse است و برای بهبود عملکرد،
    از کش و بهینه‌سازی کوئری استفاده می‌کند.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter,
                 query_optimizer: Optional[QueryOptimizer] = None,
                 analytics_cache: Optional[AnalyticsCache] = None):
        """
        مقداردهی اولیه سرویس تحلیل داده‌ها

        Args:
            clickhouse_adapter (ClickHouseAdapter): آداپتور اتصال به ClickHouse
            query_optimizer (QueryOptimizer, optional): بهینه‌ساز کوئری
            analytics_cache (AnalyticsCache, optional): کش تحلیلی پیشرفته
        """
        # آداپتور ClickHouse اجباری است
        self.clickhouse_adapter = clickhouse_adapter

        # استفاده از query_optimizer ارائه شده یا ایجاد نمونه جدید
        self.query_optimizer = query_optimizer or QueryOptimizer(self.clickhouse_adapter)

        # تنظیم سیستم کش
        self.analytics_cache = analytics_cache

        # لاگ مناسب برای تنظیمات کش
        if self.analytics_cache:
            logger.info("Analytics Service initialized with AnalyticsCache")
        else:
            logger.info("Analytics Service initialized without caching")

    async def execute_analytics_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """
        اجرای یک پرس‌وجوی تحلیلی با بررسی کش و بهینه‌سازی

        این متد ابتدا کش را بررسی می‌کند، سپس کوئری را بهینه‌سازی کرده و
        در نهایت آن را در ClickHouse اجرا می‌کند.

        Args:
            query (AnalyticsQuery): کوئری تحلیلی با متن و پارامترها

        Returns:
            AnalyticsResult: نتیجه کوئری با داده‌ها یا خطا

        Raises:
            QueryError: در صورت بروز خطا در اجرای کوئری
        """
        # بررسی کش برای نتیجه از قبل ذخیره‌شده
        cached_result = None

        try:
            # تلاش برای بازیابی از کش
            if self.analytics_cache:
                cached_result = await self.analytics_cache.get_cached_result(
                    query.query_text, query.params)

            if cached_result:
                logger.info(f"Returning cached result for query ID: {id(query)}")
                return AnalyticsResult(query=query, data=cached_result)

        except Exception as e:
            # خطای کش را لاگ می‌کنیم ولی ادامه می‌دهیم
            logger.warning(f"Cache retrieval error, proceeding without cache: {str(e)}")

        try:
            # بهینه‌سازی کوئری قبل از اجرا
            if hasattr(self.query_optimizer, 'optimize_query_with_column_expansion'):
                optimized_query = await self.query_optimizer.optimize_query_with_column_expansion(query.query_text)
            else:
                optimized_query = self.query_optimizer.optimize_query(query.query_text)

            # اجرای کوئری در ClickHouse با پارامترها
            result = await self.clickhouse_adapter.execute(optimized_query, query.params)

            # ذخیره نتیجه در کش
            try:
                if self.analytics_cache:
                    await self.analytics_cache.set_cached_result(
                        query.query_text, result, params=query.params)
            except Exception as cache_error:
                # خطای کش را لاگ می‌کنیم ولی نتیجه را برمی‌گردانیم
                logger.warning(f"Failed to cache result: {str(cache_error)}")

            return AnalyticsResult(query=query, data=result)

        except QueryError:
            # خطای کوئری را مستقیماً منتقل می‌کنیم
            raise

        except Exception as e:
            error_msg = f"Analytics query execution failed: {str(e)}"
            logger.error(error_msg)
            raise QueryError(
                message=error_msg,
                code="CHE610",
                query=query.query_text[:100],
                details={"params": query.params if query.params else {}}
            )

    async def execute_batch_queries(self, queries: List[AnalyticsQuery]) -> List[AnalyticsResult]:
        """
        اجرای دسته‌ای چندین کوئری تحلیلی

        این متد چندین کوئری را یکی پس از دیگری اجرا می‌کند و نتایج را
        در یک لیست برمی‌گرداند. حتی اگر یک کوئری شکست بخورد، بقیه کوئری‌ها اجرا می‌شوند.

        Args:
            queries (List[AnalyticsQuery]): لیست کوئری‌های تحلیلی

        Returns:
            List[AnalyticsResult]: لیست نتایج کوئری‌ها
        """
        results = []

        for query in queries:
            try:
                result = await self.execute_analytics_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing query in batch: {str(e)}")
                # ادامه اجرای بقیه کوئری‌ها با وجود خطا
                results.append(AnalyticsResult(
                    query=query,
                    data=[],
                    error=str(e)
                ))

        return results

    async def invalidate_cache(self, query: Optional[AnalyticsQuery] = None) -> None:
        """
        حذف کش برای یک کوئری خاص یا کل کش

        Args:
            query (AnalyticsQuery, optional): کوئری برای حذف از کش. اگر None باشد، کل کش پاک می‌شود.

        Raises:
            OperationalError: در صورت بروز خطا در حذف کش
        """
        if not self.analytics_cache:
            logger.info("No cache system available to invalidate")
            return

        try:
            if query:
                # حذف کش برای یک کوئری خاص
                await self.analytics_cache.invalidate_cache(query.query_text, query.params)
                logger.info(f"Cache invalidated for query ID: {id(query)}")
            else:
                # حذف کل کش
                await self.analytics_cache.invalidate_cache()
                logger.info("All cache entries invalidated")

        except Exception as e:
            error_msg = f"Failed to invalidate cache: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE611")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار کش

        Returns:
            Dict[str, Any]: آمار کش

        Raises:
            OperationalError: در صورت بروز خطا در دریافت آمار
        """
        try:
            if self.analytics_cache:
                return await self.analytics_cache.get_stats()
            else:
                return {
                    "cache_available": False,
                    "message": "No cache system available"
                }
        except Exception as e:
            error_msg = f"Failed to get cache stats: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(message=error_msg, code="CHE612")

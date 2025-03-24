# infrastructure/clickhouse/optimization/query_optimizer.py
import logging
import re
from typing import Optional, Dict, Any, List, Set
from ..config.config import config
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..exceptions import (
    QueryError, QuerySyntaxError, OperationalError
)

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """
    بهینه‌سازی اجرای کوئری‌ها برای افزایش کارایی و کاهش زمان پردازش

    این کلاس کوئری‌های SQL را قبل از اجرا بهینه‌سازی می‌کند تا عملکرد بهتری داشته باشند.
    """
    # الگوهای رگولار اکسپرشن برای شناسایی بخش‌های مختلف کوئری
    PATTERNS = {
        'select_star': re.compile(r'SELECT\s+\*\s+FROM', re.IGNORECASE),
        'table_name': re.compile(r'FROM\s+([a-zA-Z0-9_[.]]+)', re.IGNORECASE),
        'where_clause': re.compile(r'WHERE', re.IGNORECASE),
        'order_by': re.compile(r'ORDER\s+BY', re.IGNORECASE),
        'group_by': re.compile(r'GROUP\s+BY', re.IGNORECASE),
        'limit': re.compile(r'LIMIT\s+\d+', re.IGNORECASE),
        'join': re.compile(r'(LEFT|RIGHT|INNER|OUTER|FULL|CROSS)?\s*JOIN', re.IGNORECASE)
    }

    def __init__(self, clickhouse_adapter: Optional[ClickHouseAdapter] = None):
        """
        مقداردهی اولیه سیستم بهینه‌سازی کوئری‌ها

        Args:
            clickhouse_adapter (ClickHouseAdapter, optional): آداپتور اتصال به ClickHouse
        """
        self.clickhouse_adapter = clickhouse_adapter
        self.optimize_level = config.get_monitoring_config().get("query_optimize_level", 1)

        # جدول کش برای نگهداری ساختار جداول
        self.table_schema_cache: Dict[str, List[str]] = {}

        logger.info(f"Query Optimizer initialized with optimization level: {self.optimize_level}")

    async def _get_table_columns(self, table_name: str) -> List[str]:
        """
        دریافت لیست ستون‌های یک جدول

        Args:
            table_name (str): نام جدول

        Returns:
            List[str]: لیست نام ستون‌ها
        """
        if not self.clickhouse_adapter:
            logger.warning("ClickHouse adapter not provided. Cannot get table columns.")
            return []

        if table_name in self.table_schema_cache:
            return self.table_schema_cache[table_name]

        try:
            query = "DESCRIBE TABLE {table}"
            result = await self.clickhouse_adapter.execute(query, {"table": table_name})

            columns = [row['name'] for row in result]
            self.table_schema_cache[table_name] = columns
            return columns

        except Exception as e:
            logger.warning(f"Failed to get columns for table {table_name}: {str(e)}")
            return []

    def optimize_query(self, query: str) -> str:
        """
        بررسی و بهینه‌سازی کوئری قبل از اجرا

        Args:
            query (str): متن کوئری SQL

        Returns:
            str: کوئری بهینه‌شده

        Raises:
            QuerySyntaxError: در صورت خطای نحوی در کوئری
        """
        if not query or not isinstance(query, str):
            raise QuerySyntaxError(
                message="Invalid query: must be a non-empty string",
                code="CHE521"
            )

        optimized_query = query.strip()
        optimization_applied = False

        try:
            # بررسی وجود SELECT * و جایگزینی با ستون‌های مشخص (فقط توصیه)
            if self.PATTERNS['select_star'].search(optimized_query):
                logger.warning("Query uses 'SELECT *' which may not be optimal for performance.")
                # در سطح بهینه‌سازی 0 فقط هشدار می‌دهیم
                # توجه: جایگزینی واقعی به صورت async انجام می‌شود و اینجا فقط هشدار می‌دهیم

            # بررسی وجود WHERE برای بهبود کارایی
            if not self.PATTERNS['where_clause'].search(optimized_query):
                logger.warning("Query does not contain WHERE condition. This may affect performance.")

            # بررسی وجود LIMIT برای کنترل حجم نتایج
            if not self.PATTERNS['limit'].search(optimized_query) and not self.PATTERNS['group_by'].search(
                    optimized_query):
                logger.warning("Query does not contain LIMIT clause. Consider adding to control result set size.")

            # بررسی وجود ORDER BY بدون LIMIT (می‌تواند پرهزینه باشد)
            if self.PATTERNS['order_by'].search(optimized_query) and not self.PATTERNS['limit'].search(optimized_query):
                logger.warning(
                    "Query contains ORDER BY without LIMIT. This may cause performance issues for large result sets.")

            # ثبت نتیجه در لاگ
            if optimization_applied:
                logger.info("Query optimized successfully")
            else:
                logger.debug("No optimizations applied to query")

            return optimized_query

        except Exception as e:
            error_msg = f"Error optimizing query: {str(e)}"
            logger.error(error_msg)

            raise QueryError(
                message=error_msg,
                code="CHE522",
                details={"query": query[:100]}
            )

    async def optimize_query_with_column_expansion(self, query: str) -> str:
        """
        بهینه‌سازی پیشرفته کوئری با استخراج و جایگزینی ستون‌های واقعی به جای SELECT *

        Args:
            query (str): متن کوئری SQL

        Returns:
            str: کوئری بهینه‌شده
        """
        if not self.clickhouse_adapter:
            logger.warning("ClickHouse adapter not provided. Cannot expand column list.")
            return query

        optimized_query = query.strip()

        # جایگزینی SELECT * با ستون‌های واقعی
        if self.PATTERNS['select_star'].search(optimized_query):
            table_match = self.PATTERNS['table_name'].search(optimized_query)
            if table_match:
                table_name = table_match.group(1)
                columns = await self._get_table_columns(table_name)
                if columns:
                    column_list = ", ".join(columns)
                    optimized_query = self.PATTERNS['select_star'].sub(f"SELECT {column_list} FROM", optimized_query)
                    logger.info(f"Replaced 'SELECT *' with explicit column list for table {table_name}")

        return optimized_query

    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        تحلیل کوئری و ارائه پیشنهادات بهینه‌سازی

        Args:
            query (str): متن کوئری SQL

        Returns:
            Dict[str, Any]: نتایج تحلیل کوئری

        Raises:
            QueryError: در صورت بروز خطا در تحلیل کوئری
        """
        if not self.clickhouse_adapter:
            raise OperationalError(
                message="ClickHouse adapter not provided. Cannot analyze query.",
                code="CHE523"
            )

        try:
            # تحلیل کوئری با استفاده از توابع سیستمی ClickHouse
            explain_query = "EXPLAIN SYNTAX {query}"
            explain_result = await self.clickhouse_adapter.execute(explain_query, {"query": query})

            # متغیر برای نگهداری نتایج تحلیل اجرایی
            pipeline_result = None

            # تحلیل اجرایی در صورت امکان
            try:
                explain_pipeline_query = "EXPLAIN PIPELINE {query}"
                pipeline_result = await self.clickhouse_adapter.execute(explain_pipeline_query, {"query": query})
            except Exception:
                pass

            # گردآوری نتایج
            analysis: Dict[str, Any] = {
                "syntax_analysis": explain_result,
                "pipeline_analysis": pipeline_result,
                "recommendations": []
            }

            # اضافه کردن توصیه‌ها بر اساس الگوهای کوئری
            if self.PATTERNS['select_star'].search(query):
                analysis["recommendations"].append(
                    "Consider replacing 'SELECT *' with only the columns you need to reduce data transfer"
                )

            if not self.PATTERNS['where_clause'].search(query):
                analysis["recommendations"].append(
                    "Add WHERE clause to filter data early and improve query performance"
                )

            if not self.PATTERNS['limit'].search(query) and not self.PATTERNS['group_by'].search(query):
                analysis["recommendations"].append(
                    "Add LIMIT clause to control result set size"
                )

            return analysis

        except Exception as e:
            error_msg = f"Error analyzing query: {str(e)}"
            logger.error(error_msg)

            raise QueryError(
                message=error_msg,
                code="CHE524",
                details={"query": query[:100]}
            )

    async def execute_optimized_query(self, query: str, params: Optional[Dict[str, Any]] = None):
        """
        اجرای کوئری بهینه‌شده

        Args:
            query (str): متن کوئری SQL
            params (Dict[str, Any], optional): پارامترهای کوئری

        Returns:
            نتیجه اجرای کوئری

        Raises:
            OperationalError: در صورت عدم وجود آداپتور ClickHouse
            QueryError: در صورت بروز خطا در اجرای کوئری
        """
        if not self.clickhouse_adapter:
            raise OperationalError(
                message="ClickHouse adapter not provided. Cannot execute query.",
                code="CHE525"
            )

        # بهینه‌سازی اولیه بدون استخراج ستون‌ها
        basic_optimized_query = self.optimize_query(query)

        # اگر سطح بهینه‌سازی بالاتر از 0 باشد، استخراج ستون‌ها را انجام می‌دهیم
        if self.optimize_level > 0 and self.PATTERNS['select_star'].search(basic_optimized_query):
            optimized_query = await self.optimize_query_with_column_expansion(basic_optimized_query)
        else:
            optimized_query = basic_optimized_query

        try:
            result = await self.clickhouse_adapter.execute(optimized_query, params)
            return result

        except Exception as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg)

            raise QueryError(
                message=error_msg,
                code="CHE526",
                details={"query": query[:100], "params": params}
            )
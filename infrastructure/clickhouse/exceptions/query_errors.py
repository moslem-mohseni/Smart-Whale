# infrastructure/clickhouse/exceptions/query_errors.py
"""
خطاهای مرتبط با اجرای کوئری در ClickHouse
"""

from .base import ClickHouseBaseError
from typing import Optional, Dict, Any


class QueryError(ClickHouseBaseError):
    """
    خطای پایه برای مشکلات اجرای کوئری در ClickHouse
    """

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, query: Optional[str] = None):
        """
        مقداردهی اولیه خطای کوئری

        Args:
            message (str): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            query (str, optional): کوئری مورد نظر
        """
        # تنظیم کد پیش‌فرض برای خطای کوئری
        if code is None:
            code = "CHE200"

        # اضافه کردن اطلاعات کوئری به جزئیات
        if query:
            if details is None:
                details = {}
            # محدود کردن طول کوئری در جزئیات برای جلوگیری از لاگ‌های بزرگ
            if len(query) > 500:
                details["query"] = query[:500] + "..."
            else:
                details["query"] = query

        super().__init__(message, code, details)


class QuerySyntaxError(QueryError):
    """
    خطای نحوی در کوئری ClickHouse
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, query: Optional[str] = None,
                 error_position: Optional[int] = None):
        """
        مقداردهی اولیه خطای نحوی کوئری

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            query (str, optional): کوئری مورد نظر
            error_position (int, optional): موقعیت خطا در کوئری
        """
        if message is None:
            message = "SQL syntax error in query"

        if code is None:
            code = "CHE201"

        # اضافه کردن موقعیت خطا به جزئیات
        if error_position is not None:
            if details is None:
                details = {}
            details["error_position"] = error_position

        super().__init__(message, code, details, query)


class QueryExecutionTimeoutError(QueryError):
    """
    خطای زمان انتظار برای اجرای کوئری در ClickHouse
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, query: Optional[str] = None,
                 timeout: Optional[float] = None):
        """
        مقداردهی اولیه خطای زمان انتظار اجرای کوئری

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            query (str, optional): کوئری مورد نظر
            timeout (float, optional): مدت زمان انتظار (ثانیه)
        """
        if message is None:
            message = "Query execution timed out"

        if code is None:
            code = "CHE202"

        # اضافه کردن زمان انتظار به جزئیات
        if timeout is not None:
            if details is None:
                details = {}
            details["timeout"] = timeout

        super().__init__(message, code, details, query)


class QueryCancellationError(QueryError):
    """
    خطای لغو اجرای کوئری در ClickHouse
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, query: Optional[str] = None,
                 reason: Optional[str] = None):
        """
        مقداردهی اولیه خطای لغو کوئری

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            query (str, optional): کوئری مورد نظر
            reason (str, optional): دلیل لغو کوئری
        """
        if message is None:
            message = "Query was cancelled"

        if code is None:
            code = "CHE203"

        # اضافه کردن دلیل لغو به جزئیات
        if reason:
            if details is None:
                details = {}
            details["cancellation_reason"] = reason

        super().__init__(message, code, details, query)


class DataTypeError(QueryError):
    """
    خطای نوع داده در کوئری ClickHouse
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, query: Optional[str] = None,
                 column: Optional[str] = None, expected_type: Optional[str] = None,
                 actual_type: Optional[str] = None):
        """
        مقداردهی اولیه خطای نوع داده

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            query (str, optional): کوئری مورد نظر
            column (str, optional): نام ستون
            expected_type (str, optional): نوع داده مورد انتظار
            actual_type (str, optional): نوع داده واقعی
        """
        if message is None:
            if column and expected_type and actual_type:
                message = f"Data type mismatch for column '{column}'. Expected {expected_type}, got {actual_type}"
            else:
                message = "Data type mismatch in query"

        if code is None:
            code = "CHE204"

        # اضافه کردن اطلاعات نوع داده به جزئیات
        if column or expected_type or actual_type:
            if details is None:
                details = {}
            if column:
                details["column"] = column
            if expected_type:
                details["expected_type"] = expected_type
            if actual_type:
                details["actual_type"] = actual_type

        super().__init__(message, code, details, query)

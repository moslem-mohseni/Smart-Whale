# infrastructure/clickhouse/integration/rest_api.py
"""
API REST برای دسترسی به داده‌های ClickHouse
"""

import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Response, Request, Depends, status
from pydantic import BaseModel, Field
from ..service.analytics_service import AnalyticsService
from ..domain.models import AnalyticsQuery
from ..exceptions import QueryError, OperationalError, ClickHouseBaseError
from ..config.config import config

logger = logging.getLogger(__name__)


# مدل‌های Pydantic برای درخواست و پاسخ
class AnalyticsRequest(BaseModel):
    """مدل درخواست برای اجرای کوئری تحلیلی"""
    query: str = Field(..., description="متن کوئری SQL برای اجرا در ClickHouse")
    params: Optional[Dict[str, Any]] = Field(None, description="پارامترهای کوئری")


class ErrorResponse(BaseModel):
    """مدل پاسخ خطا"""
    error_type: str = Field(..., description="نوع خطا")
    code: str = Field(..., description="کد خطا")
    message: str = Field(..., description="پیام خطا")
    details: Optional[Dict[str, Any]] = Field(None, description="جزئیات اضافی خطا")


class AnalyticsResponse(BaseModel):
    """مدل پاسخ برای نتایج کوئری تحلیلی"""
    data: Optional[Any] = Field(None, description="داده‌های نتیجه")
    error: Optional[ErrorResponse] = Field(None, description="اطلاعات خطا در صورت بروز مشکل")


# ایجاد نمونه FastAPI
app = FastAPI(title="ClickHouse Analytics API",
              description="API برای اجرای کوئری‌های تحلیلی در ClickHouse",
              version="1.0.0")


class RestAPI:
    """
    API REST برای مدیریت کوئری‌های تحلیلی ClickHouse

    این کلاس وظیفه ارائه یک API REST برای اجرای کوئری‌های تحلیلی در ClickHouse
    و ارائه نتایج به فرمت JSON را بر عهده دارد.
    """

    def __init__(self, analytics_service: AnalyticsService):
        """
        مقداردهی اولیه API REST با سرویس تحلیل داده‌ها

        Args:
            analytics_service (AnalyticsService): سرویس تحلیل داده‌ها
        """
        self.analytics_service = analytics_service

        # اضافه کردن مسیرها و روت‌ها به FastAPI
        self._setup_routes()

        logger.info("REST API initialized")

    def _setup_routes(self):
        """تنظیم مسیرها و روت‌های API"""

        @app.get("/health")
        async def health_check():
            """بررسی سلامت سرویس"""
            return {"status": "healthy", "service": "clickhouse-analytics"}

        @app.post("/analytics", response_model=AnalyticsResponse, status_code=status.HTTP_200_OK)
        async def execute_query(request: AnalyticsRequest, response: Response):
            """
            اجرای کوئری تحلیلی در ClickHouse

            Args:
                request (AnalyticsRequest): درخواست حاوی کوئری و پارامترها

            Returns:
                AnalyticsResponse: نتیجه اجرای کوئری یا اطلاعات خطا
            """
            try:
                # ایجاد شیء کوئری تحلیلی
                query = AnalyticsQuery(
                    query_text=request.query,
                    params=request.params
                )

                # اجرای کوئری
                result = await self.analytics_service.execute_analytics_query(query)

                # بررسی خطا در نتیجه
                if hasattr(result, 'error') and result.error:
                    response.status_code = status.HTTP_400_BAD_REQUEST
                    return AnalyticsResponse(
                        error=ErrorResponse(
                            error_type="QueryError",
                            code="API001",
                            message=result.error
                        )
                    )

                return AnalyticsResponse(data=result.data)

            except QueryError as e:
                # خطای کوئری
                response.status_code = status.HTTP_400_BAD_REQUEST
                return AnalyticsResponse(
                    error=ErrorResponse(
                        error_type=e.__class__.__name__,
                        code=e.code,
                        message=e.message,
                        details=e.details
                    )
                )

            except OperationalError as e:
                # خطای عملیاتی
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                return AnalyticsResponse(
                    error=ErrorResponse(
                        error_type=e.__class__.__name__,
                        code=e.code,
                        message=e.message,
                        details=e.details
                    )
                )

            except Exception as e:
                # سایر خطاها
                logger.error(f"Unexpected error in query execution: {str(e)}")
                response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                return AnalyticsResponse(
                    error=ErrorResponse(
                        error_type="ServerError",
                        code="API999",
                        message=f"Unexpected server error: {str(e)}"
                    )
                )

        @app.get("/analytics/cache/stats")
        async def get_cache_stats():
            """
            دریافت آمار کش

            Returns:
                Dict[str, Any]: آمار کش
            """
            try:
                stats = await self.analytics_service.get_cache_stats()
                return {"cache_stats": stats}
            except Exception as e:
                logger.error(f"Failed to get cache stats: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to get cache stats: {str(e)}"
                )

        @app.post("/analytics/cache/invalidate")
        async def invalidate_cache(query: Optional[str] = None):
            """
            حذف کش یک کوئری خاص یا کل کش

            Args:
                query (str, optional): کوئری برای حذف از کش. اگر None باشد، کل کش پاک می‌شود.

            Returns:
                Dict[str, str]: پیام موفقیت
            """
            try:
                if query:
                    query_obj = AnalyticsQuery(query_text=query)
                    await self.analytics_service.invalidate_cache(query_obj)
                    return {"message": "Cache for specific query invalidated successfully"}
                else:
                    await self.analytics_service.invalidate_cache()
                    return {"message": "All cache entries invalidated successfully"}
            except Exception as e:
                logger.error(f"Failed to invalidate cache: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to invalidate cache: {str(e)}"
                )

    def get_app(self) -> FastAPI:
        """
        دریافت نمونه FastAPI برای استفاده در ASGI server

        Returns:
            FastAPI: نمونه FastAPI
        """
        return app

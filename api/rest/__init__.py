# api/rest/__init__.py
"""
REST API Module
-------------
This module implements RESTful API endpoints for the system.

The REST API follows standard HTTP methods and status codes:
- GET: Retrieve resources
- POST: Create new resources
- PUT: Update existing resources
- DELETE: Remove resources

Key features:
- JWT authentication
- Rate limiting
- Request validation
- Response formatting
- Error handling

All endpoints follow a consistent structure and documentation format
to ensure ease of use and maintainability.
"""

from enum import Enum
from typing import Optional, Dict, Any


class APIVersion(Enum):
    """API versions supported by the system"""
    V1 = "v1"
    V2 = "v2"


class EndpointCategory(Enum):
    """Categories of API endpoints"""
    USER = "user"
    AUTH = "auth"
    TRADING = "trading"
    ANALYSIS = "analysis"


# استاندارد پاسخ‌های API
def create_response(
        data: Optional[Dict[str, Any]] = None,
        message: str = "",
        success: bool = True,
        status_code: int = 200
) -> Dict[str, Any]:
    """
    Creates a standardized API response format.

    این ساختار استاندارد به کلاینت‌ها کمک می‌کنه تا همیشه
    پاسخ‌های یکسان و قابل پیش‌بینی دریافت کنن.
    """
    return {
        "success": success,
        "message": message,
        "data": data,
        "status": status_code,
        "timestamp": datetime.now().isoformat()
    }
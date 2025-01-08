# api/external/__init__.py
"""
External API Integration Module
---------------------------
Manages connections and interactions with external services and APIs.

This module handles:
- Exchange API integrations
- Third-party data providers
- External authentication services
- Payment gateways
- Market data feeds

Features:
- Connection pooling
- Rate limit management
- Error handling and retry logic
- Response caching
- API key management
"""

from typing import Dict, Optional
from enum import Enum


class ExternalService(Enum):
    """Types of external services integrated with the system"""
    EXCHANGE = "exchange"
    DATA_PROVIDER = "data_provider"
    AUTH_SERVICE = "auth_service"
    PAYMENT = "payment"


class ServiceStatus(Enum):
    """Possible states of external services"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    DOWN = "down"
    MAINTENANCE = "maintenance"


class ExternalServiceManager:
    """
    مدیریت ارتباط با سرویس‌های خارجی.
    این کلاس تمام ارتباطات خارجی رو مدیریت و نظارت می‌کنه.
    """

    def __init__(self):
        self._services: Dict[str, ServiceStatus] = {}
        self._connections = {}

    def get_service_status(self, service: ExternalService) -> ServiceStatus:
        """وضعیت فعلی یک سرویس خارجی رو برمی‌گردونه"""
        return self._services.get(service.value, ServiceStatus.DOWN)
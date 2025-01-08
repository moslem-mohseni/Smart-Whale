
# infrastructure/kafka/domain/__init__.py
"""
این ماژول شامل مدل‌های دامنه برای کار با کافکا است.
ساختارهای داده و انتیتی‌های اصلی سیستم در اینجا تعریف می‌شوند.
"""

from .models import Message, TopicConfig

__all__ = ['Message', 'TopicConfig']

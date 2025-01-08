# infrastructure/kafka/adapters/__init__.py
"""
این ماژول شامل تطبیق‌دهنده‌های کافکا است که مسئول ارتباط مستقیم با سرور کافکا هستند.
Producer و Consumer اصلی در این بخش تعریف شده‌اند.
"""

from .consumer import MessageConsumer
from .producer import MessageProducer

__all__ = ['MessageConsumer', 'MessageProducer']
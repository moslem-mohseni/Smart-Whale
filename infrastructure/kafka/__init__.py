# infrastructure/kafka/__init__.py
"""
Kafka Message Broker
------------------
Handles message queuing and event streaming for the system.
Includes configuration and management scripts for Kafka clusters.
"""

"""
پکیج کافکا برای مدیریت عملیات پیام‌رسانی و ارتباطات ناهمگام در سیستم.

این پکیج شامل موارد زیر است:
- سرویس مدیریت پیام‌رسانی
- مدیریت موضوعات
- مدیریت تولیدکننده و مصرف‌کننده پیام‌ها
- ابزارهای نگهداری و مدیریت
"""

from .service import KafkaService, KafkaMaintenance, TopicManager
from .domain.models import Message, TopicConfig
from .config.settings import KafkaConfig
from .adapters import MessageConsumer, MessageProducer

__version__ = '1.0.0'

__all__ = [
    'KafkaService',
    'KafkaMaintenance',
    'TopicManager',
    'Message',
    'TopicConfig',
    'MessageConsumer',
    'MessageProducer',
    'KafkaConfig'
]

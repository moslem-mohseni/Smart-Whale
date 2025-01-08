
# infrastructure/kafka/migrations/__init__.py
"""
این ماژول شامل اسکریپت‌های مهاجرت برای مدیریت تغییرات در ساختار موضوعات کافکا است.
"""

from .v001_create_base_topics import upgrade, downgrade

__all__ = ['upgrade', 'downgrade']
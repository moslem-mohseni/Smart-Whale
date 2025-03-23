"""
federation Package
--------------------
این پکیج شامل ابزارها و سرویس‌های مربوط به آموزش توزیع‌شده و اشتراک دانش بین مدل‌های زبانی است.
موارد شامل:
  - KnowledgeSharing: به اشتراک‌گذاری دانش بین مدل‌ها.
  - DistributedLearning: هماهنگی آموزش توزیع‌شده بین مدل‌های مختلف.

تمامی کلاس‌ها به صورت نهایی و عملیاتی پیاده‌سازی شده‌اند.
"""

from .knowledge_sharing import KnowledgeSharing
from .distributed_learning import DistributedLearning

__all__ = [
    "KnowledgeSharing",
    "DistributedLearning"
]

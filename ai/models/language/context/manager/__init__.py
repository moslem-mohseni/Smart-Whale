"""
ماژول `manager/` مسئول مدیریت مکالمات، پیگیری نشست‌ها، وضعیت مکالمه، پاسخ‌های جایگزین و به‌روزرسانی داده‌ها است.

این ماژول شامل بخش‌های زیر است:
- `context_tracker`: رهگیری جریان مکالمه و ذخیره داده‌های تعاملات
- `session_handler`: مدیریت نشست‌های کاربران و تعاملات در طول زمان
- `state_manager`: تعیین وضعیت مکالمه و فازهای مختلف تعاملات
- `fallback_handler`: ارائه پاسخ‌های جایگزین در صورت عدم وجود داده کافی
- `update_policy`: تنظیم سیاست‌های حذف، نگهداری و انتقال داده‌های مکالمه‌ای
"""

from .context_tracker import ContextTracker
from .session_handler import SessionHandler
from .state_manager import StateManager
from .fallback_handler import FallbackHandler
from .update_policy import UpdatePolicy

# مقداردهی اولیه ماژول‌ها
context_tracker = ContextTracker()
session_handler = SessionHandler()
state_manager = StateManager()
fallback_handler = FallbackHandler()
update_policy = UpdatePolicy()

__all__ = [
    "ContextTracker",
    "SessionHandler",
    "StateManager",
    "FallbackHandler",
    "UpdatePolicy",
]

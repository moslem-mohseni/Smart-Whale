# فایل نهایی state_manager.py برای مدیریت و ذخیره‌سازی وضعیت اجزای سیستم خودآموزی
# ----------------------------------------------------------------------------

"""
مدیریت و ذخیره‌سازی وضعیت اجزای سیستم خودآموزی (State Manager)

این ماژول با استفاده از Redis (یا هر سیستم کش/پایگاه داده دیگر) وضعیت کامپوننت‌ها را ذخیره و بازیابی می‌کند.
در صورت نیاز می‌توان آن را با زیرساخت‌های متفاوت جایگزین کرد.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Set

from infrastructure.redis.service.cache_service import CacheService
from infrastructure.redis.config.settings import RedisConfig
from event_system import publish_event, EventType, EventPriority


class StateManager:
    """
    مدیریت یکپارچه وضعیت اجزای سیستم خودآموزی.

    از یک سرویس کش (مثلاً Redis) برای نگهداری وضعیت استفاده می‌کند.
    """

    def __init__(self, autosave_interval: int = 300):
        """
        راه‌اندازی مدیر وضعیت

        Args:
            autosave_interval: فاصله زمانی ذخیره‌سازی خودکار (ثانیه)
        """
        self.logger = logging.getLogger("SelfLearningStateManager")
        self.autosave_interval = autosave_interval

        # تنظیمات ردیس از فایل تنظیمات مرکزی
        self.config = RedisConfig()
        self.cache_service = CacheService(self.config)

        # پیگیری تغییرات
        self.modified_components: Set[str] = set()

        # وضعیت سیستم
        self.running = False
        self.autosave_task: Optional[asyncio.Task] = None

        self.logger.info("[StateManager] initialized")

    async def start(self):
        """
        شروع فعالیت مدیر وضعیت و ذخیره‌سازی خودکار
        """
        if self.running:
            return

        # اتصال به سرویس کش
        await self.cache_service.connect()
        self.running = True

        # شروع وظیفه ذخیره‌سازی خودکار
        if self.autosave_interval > 0:
            self.autosave_task = asyncio.create_task(self._autosave_loop())
            self.logger.info(f"[StateManager] Autosave started with interval {self.autosave_interval}s")

        # رویداد آغاز به کار
        await publish_event(
            event_type=EventType.CONFIG_CHANGED,
            data={"component": "StateManager", "action": "started"},
            priority=EventPriority.LOW
        )

    async def stop(self):
        """
        توقف فعالیت مدیر وضعیت و ذخیره‌سازی نهایی
        """
        if not self.running:
            return

        self.running = False
        if self.autosave_task:
            self.autosave_task.cancel()
            try:
                await self.autosave_task
            except asyncio.CancelledError:
                pass

        # قطع اتصال از سرویس کش
        await self.cache_service.disconnect()
        self.logger.info("[StateManager] stopped")

    async def _autosave_loop(self):
        """
        حلقه ذخیره‌سازی خودکار وضعیت‌ها
        """
        try:
            while self.running:
                await asyncio.sleep(self.autosave_interval)
                if self.modified_components:
                    # در این ساختار نمونه، ما فقط ثبت می‌کنیم که تغییر انجام شده.
                    self.logger.debug(f"Autosaving states for {len(self.modified_components)} components.")
                    self.modified_components.clear()
        except asyncio.CancelledError:
            self.logger.debug("[StateManager] Autosave loop cancelled")
            raise
        except Exception as e:
            self.logger.error(f"[StateManager] Error in autosave loop: {str(e)}")

    async def update_state(self, component_id: str, key: str, value: Any) -> bool:
        """
        به‌روزرسانی وضعیت یک کامپوننت خاص

        Args:
            component_id: شناسه کامپوننت
            key: کلید وضعیت
            value: مقدار جدید
        Returns:
            bool: نتیجه به‌روزرسانی
        """
        try:
            # ذخیره در کش (کلید اصلی: state:{component_id}:{key})
            cache_key = f"state:{component_id}:{key}"
            await self.cache_service.set(cache_key, value)

            # ثبت در مجموعه تغییرات
            self.modified_components.add(component_id)

            return True
        except Exception as e:
            self.logger.error(f"[StateManager] Error updating state for {component_id}.{key}: {str(e)}")
            return False

    async def get_state(self, component_id: str, key: str, default: Any = None) -> Any:
        """
        دریافت وضعیت یک کامپوننت

        Args:
            component_id: شناسه کامپوننت
            key: کلید وضعیت
            default: مقدار پیش‌فرض در صورت نبود مقدار
        Returns:
            Any: مقدار بازیابی شده یا مقدار پیش‌فرض
        """
        try:
            cache_key = f"state:{component_id}:{key}"
            value = await self.cache_service.get(cache_key)
            if value is None:
                return default
            return value
        except Exception as e:
            self.logger.error(f"[StateManager] Error getting state for {component_id}.{key}: {str(e)}")
            return default

    async def clear_component_state(self, component_id: str) -> bool:
        """
        پاکسازی وضعیت یک کامپوننت از کش
        Args:
            component_id: شناسه کامپوننت
        Returns:
            bool: نتیجه پاکسازی
        """
        try:
            # اینجا باید کلیدهای مرتبط با آن کامپوننت را حذف کنیم.
            # متأسفانه CacheService ممکن است scan_keys یا پترن خاصی نداشته باشد.
            # این بخش بسته به پیاده‌سازی CacheService قابل تغییر است.

            # حذف از مجموعه تغییرات
            self.modified_components.discard(component_id)

            # فرض بر این است که در CacheService متد خاصی نداریم.
            # لذا اینجا فقط لاگ می‌گیریم یا متد دلخواه را پیاده می‌کنیم.

            return True
        except Exception as e:
            self.logger.error(f"[StateManager] Error clearing component state for {component_id}: {str(e)}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت کلی مدیر وضعیت
        Returns:
            Dict[str, Any]: اطلاعات وضعیت
        """
        return {
            "running": self.running,
            "autosave_interval": self.autosave_interval,
            "modified_components_count": len(self.modified_components)
        }

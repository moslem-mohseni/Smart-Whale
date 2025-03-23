"""
LearningRateAdjuster Module
----------------------------
این فایل مسئول تنظیم خودکار نرخ یادگیری در فرآیند آموزش مدل‌های خودآموزی است.
این کلاس به صورت نهایی و عملیاتی پیاده‌سازی شده و با استفاده از الگوریتم ReduceLROnPlateau،
بر اساس مقادیر loss دریافتی از دوره‌های آموزشی، نرخ یادگیری را کاهش می‌دهد (در صورت عدم بهبود).
ویژگی‌های پیکربندی شامل:
  - initial_lr: نرخ یادگیری اولیه.
  - factor: ضریب کاهش نرخ یادگیری (مثلاً 0.5 برای نصف کردن).
  - patience: تعداد دوره‌های آموزشی که در صورت عدم بهبود loss، نرخ یادگیری کاهش می‌یابد.
  - min_lr: حداقل نرخ یادگیری مجاز.
تمامی امکانات با بهترین مکانیسم‌های هوشمند و کارایی بالا پیاده‌سازی شده‌اند.
"""

import logging
from typing import Dict, Any, Optional

from ..base.base_component import BaseComponent


class LearningRateAdjuster(BaseComponent):
    """
    LearningRateAdjuster مسئول تنظیم نرخ یادگیری به صورت تطبیقی بر اساس معیارهای عملکردی دوره‌های آموزشی است.

    الگوریتم:
      - اگر مقدار loss در دوره جاری بهبود یابد (کاهش یابد)، نرخ یادگیری ثابت می‌ماند و بهترین loss به‌روزرسانی می‌شود.
      - در غیر این صورت، پس از گذشت تعداد دوره‌های مشخص (patience)، نرخ یادگیری با ضریب factor کاهش می‌یابد.
      - نرخ یادگیری هرگز از مقدار min_lr کمتر نخواهد شد.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه LearningRateAdjuster.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختیاری شامل:
                - "initial_lr": نرخ یادگیری اولیه (پیش‌فرض: 0.01)
                - "factor": ضریب کاهش نرخ یادگیری (پیش‌فرض: 0.5)
                - "patience": تعداد دوره‌های آموزشی بدون بهبود (پیش‌فرض: 5)
                - "min_lr": حداقل نرخ یادگیری (پیش‌فرض: 1e-6)
        """
        super().__init__(component_type="learning_rate_adjuster", config=config)
        self.logger = logging.getLogger("LearningRateAdjuster")
        self.initial_lr = float(self.config.get("initial_lr", 0.01))
        self.factor = float(self.config.get("factor", 0.5))
        self.patience = int(self.config.get("patience", 5))
        self.min_lr = float(self.config.get("min_lr", 1e-6))
        self.best_loss: Optional[float] = None
        self.wait = 0
        self.current_lr = self.initial_lr
        self.logger.info(f"[LearningRateAdjuster] Initialized with initial_lr={self.initial_lr}, "
                         f"factor={self.factor}, patience={self.patience}, min_lr={self.min_lr}")

    def adjust_learning_rate(self, current_loss: float) -> float:
        """
        تنظیم نرخ یادگیری بر اساس مقدار loss فعلی.

        Args:
            current_loss (float): مقدار loss دوره جاری.

        Returns:
            float: نرخ یادگیری به‌روز شده.
        """
        # در اولین دوره، بهترین loss را تنظیم می‌کنیم.
        if self.best_loss is None or current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
            self.logger.debug(
                f"[LearningRateAdjuster] Loss improved to {current_loss:.4f}. LR remains {self.current_lr:.6f}")
        else:
            self.wait += 1
            self.logger.debug(
                f"[LearningRateAdjuster] No improvement (current_loss={current_loss:.4f}). Wait count: {self.wait}")
            if self.wait >= self.patience:
                new_lr = self.current_lr * self.factor
                if new_lr < self.min_lr:
                    new_lr = self.min_lr
                if new_lr < self.current_lr:
                    self.logger.info(
                        f"[LearningRateAdjuster] Reducing LR from {self.current_lr:.6f} to {new_lr:.6f} after {self.wait} epochs with no improvement.")
                    self.current_lr = new_lr
                    self.increment_metric("lr_reduction")
                self.wait = 0  # Reset wait after adjustment
        return self.current_lr

    def get_current_learning_rate(self) -> float:
        """
        دریافت نرخ یادگیری فعلی.

        Returns:
            float: نرخ یادگیری فعلی.
        """
        return self.current_lr

    def reset(self) -> None:
        """
        بازنشانی تنظیمات نرخ یادگیری به حالت اولیه.
        """
        self.best_loss = None
        self.wait = 0
        self.current_lr = self.initial_lr
        self.logger.info(f"[LearningRateAdjuster] Reset LR to initial value {self.initial_lr:.6f}.")


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # شبیه‌سازی یک دنباله از lossها
    simulated_losses = [0.5, 0.48, 0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.46, 0.46]

    lr_adjuster = LearningRateAdjuster(config={
        "initial_lr": 0.01,
        "factor": 0.5,
        "patience": 3,
        "min_lr": 1e-5
    })

    for epoch, loss in enumerate(simulated_losses, start=1):
        updated_lr = lr_adjuster.adjust_learning_rate(loss)
        print(f"Epoch {epoch}: Loss = {loss}, Learning Rate = {updated_lr:.6f}")

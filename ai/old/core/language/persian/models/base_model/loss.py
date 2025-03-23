import torch.nn as nn
from models.base_model.config import BaseModelConfig


class LossManager:
    """
    کلاس مدیریت توابع هزینه (Loss Functions) برای مدل ParsBERT.
    """

    def __init__(self):
        """
        مقداردهی اولیه و انتخاب تابع هزینه مناسب.
        """
        self.loss_function = self._get_loss_function()

    def _get_loss_function(self):
        """
        مقداردهی تابع هزینه بر اساس مقدار تنظیم‌شده در `BaseModelConfig`.
        """
        loss_name = BaseModelConfig.LOSS_FUNCTION.lower()

        if loss_name == "crossentropy":
            return nn.CrossEntropyLoss()
        elif loss_name == "mse":
            return nn.MSELoss()
        elif loss_name == "hinge":
            return nn.HingeEmbeddingLoss()
        else:
            raise ValueError(f"❌ تابع هزینه `{loss_name}` نامعتبر است. گزینه‌های معتبر: `crossentropy`, `mse`, `hinge`")

    def get_loss_function(self):
        """
        دریافت تابع هزینه برای استفاده در `trainer.py`.
        """
        return self.loss_function


# ==================== تست ====================
if __name__ == "__main__":
    loss_manager = LossManager()

    print("📌 تابع هزینه انتخاب‌شده:", loss_manager.get_loss_function())

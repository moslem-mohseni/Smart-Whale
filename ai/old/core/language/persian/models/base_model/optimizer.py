import torch.optim as optim
from transformers import get_scheduler
from models.base_model.config import BaseModelConfig


class OptimizerManager:
    """
    کلاس مدیریت بهینه‌سازها و تنظیمات مربوط به `learning rate` برای مدل ParsBERT.
    """

    def __init__(self, model):
        """
        مقداردهی اولیه و تنظیم بهینه‌ساز.

        :param model: مدل PyTorch که باید بهینه‌سازی شود.
        """
        self.model = model
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

    def _get_optimizer(self):
        """
        مقداردهی اولیه بهینه‌ساز.
        """
        optimizer_name = BaseModelConfig.OPTIMIZER.lower()

        if optimizer_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=BaseModelConfig.LEARNING_RATE, weight_decay=BaseModelConfig.WEIGHT_DECAY)
        elif optimizer_name == "sgd":
            return optim.SGD(self.model.parameters(), lr=BaseModelConfig.LEARNING_RATE, momentum=0.9)
        else:
            raise ValueError(f"❌ بهینه‌ساز `{optimizer_name}` نامعتبر است. گزینه‌های معتبر: `adamw`, `sgd`")

    def _get_scheduler(self):
        """
        مقداردهی اولیه `learning rate scheduler`.
        """
        return get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=BaseModelConfig.WARMUP_STEPS,
            num_training_steps=BaseModelConfig.EPOCHS
        )

    def step(self):
        """
        به‌روزرسانی بهینه‌ساز و `scheduler` در طول آموزش.
        """
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        """
        تنظیم مقدار `gradient` به صفر قبل از هر `backpropagation`.
        """
        self.optimizer.zero_grad()

    def get_optimizer(self):
        """
        دریافت بهینه‌ساز برای استفاده در `trainer.py`.
        """
        return self.optimizer

    def get_scheduler(self):
        """
        دریافت `scheduler` برای استفاده در `trainer.py`.
        """
        return self.scheduler


# ==================== تست ====================
if __name__ == "__main__":
    from models.base_model.model import PersianBERTModel

    model = PersianBERTModel(num_labels=3).model
    optimizer_manager = OptimizerManager(model)

    print("📌 بهینه‌ساز انتخاب‌شده:", optimizer_manager.get_optimizer())
    print("📌 `Scheduler` مقداردهی شد:", optimizer_manager.get_scheduler())

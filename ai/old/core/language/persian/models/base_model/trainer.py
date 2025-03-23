import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from models.base_model.config import BaseModelConfig
from models.base_model.dataset import get_dataloader
from models.base_model.model import PersianBERTModel
from tqdm import tqdm


class Trainer:
    """
    کلاس مدیریت فرآیند آموزش مدل ParsBERT.
    """

    def __init__(self, num_labels=2):
        """
        مقداردهی اولیه کلاس آموزش.

        :param num_labels: تعداد کلاس‌های خروجی
        """
        self.device = torch.device(BaseModelConfig.DEVICE)
        self.model = PersianBERTModel(num_labels=num_labels).model.to(self.device)

        # لود داده‌های آموزشی و اعتبارسنجی
        self.train_dataloader = get_dataloader(BaseModelConfig.TRAIN_DATA_PATH, BaseModelConfig.BATCH_SIZE)
        self.valid_dataloader = get_dataloader(BaseModelConfig.VALIDATION_DATA_PATH, BaseModelConfig.BATCH_SIZE)

        # تعریف تابع هزینه (Loss Function)
        self.criterion = nn.CrossEntropyLoss()

        # تعریف بهینه‌ساز (Optimizer)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=BaseModelConfig.LEARNING_RATE, weight_decay=BaseModelConfig.WEIGHT_DECAY)

        # تنظیم یادگیری تدریجی (Scheduler)
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=BaseModelConfig.WARMUP_STEPS,
            num_training_steps=len(self.train_dataloader) * BaseModelConfig.EPOCHS,
        )

    def train_epoch(self):
        """
        اجرای یک `epoch` از آموزش مدل.
        """
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_dataloader, desc="🚀 در حال آموزش", leave=True)
        for batch in loop:
            input_ids, labels = batch["input_ids"].to(self.device), batch["label"].to(self.device)

            # حذف گرادیان‌های قبلی
            self.optimizer.zero_grad()

            # اجرای پیش‌بینی
            outputs = self.model(input_ids)
            loss = self.criterion(outputs.logits, labels)

            # محاسبه گرادیان و به‌روزرسانی وزن‌ها
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # ذخیره مقدار loss
            total_loss += loss.item()

            # نمایش مقدار Loss در tqdm
            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_dataloader)

    def evaluate(self):
        """
        ارزیابی مدل روی داده‌های اعتبارسنجی.
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in self.valid_dataloader:
                input_ids, labels = batch["input_ids"].to(self.device), batch["label"].to(self.device)
                outputs = self.model(input_ids)
                loss = self.criterion(outputs.logits, labels)

                # محاسبه دقت مدل
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                total_loss += loss.item()

        accuracy = correct / total
        return total_loss / len(self.valid_dataloader), accuracy

    def train(self):
        """
        حلقه اصلی آموزش مدل.
        """
        best_accuracy = 0

        for epoch in range(BaseModelConfig.EPOCHS):
            print(f"\n🔹 **Epoch {epoch + 1}/{BaseModelConfig.EPOCHS}**")

            train_loss = self.train_epoch()
            valid_loss, valid_accuracy = self.evaluate()

            print(f"📌 `Train Loss`: {train_loss:.4f} | `Valid Loss`: {valid_loss:.4f} | `Accuracy`: {valid_accuracy:.4f}")

            # ذخیره بهترین مدل بر اساس دقت
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                self.save_model()

        print("✅ آموزش مدل تکمیل شد!")

    def save_model(self):
        """
        ذخیره مدل آموزش‌دیده.
        """
        torch.save(self.model.state_dict(), BaseModelConfig.BEST_MODEL_PATH)
        print(f"✅ مدل ذخیره شد در `{BaseModelConfig.BEST_MODEL_PATH}`")


# ==================== تست ====================
if __name__ == "__main__":
    trainer = Trainer(num_labels=3)
    trainer.train()

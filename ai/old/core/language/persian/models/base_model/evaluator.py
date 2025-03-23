import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification
from models.base_model.config import BaseModelConfig
from models.base_model.dataset import get_dataloader
from models.base_model.model import PersianBERTModel


class Evaluator:
    """
    کلاس مدیریت ارزیابی مدل ParsBERT.
    """

    def __init__(self, num_labels=2):
        """
        مقداردهی اولیه کلاس ارزیابی.

        :param num_labels: تعداد کلاس‌های خروجی
        """
        self.device = torch.device(BaseModelConfig.DEVICE)

        # بارگذاری مدل ذخیره‌شده
        self.model = PersianBERTModel(num_labels=num_labels).model
        self.model.load_state_dict(torch.load(BaseModelConfig.BEST_MODEL_PATH))
        self.model.to(self.device)
        self.model.eval()

        # بارگذاری داده‌های تست
        self.test_dataloader = get_dataloader(BaseModelConfig.TEST_DATA_PATH, BaseModelConfig.BATCH_SIZE, shuffle=False)

    def evaluate(self):
        """
        اجرای ارزیابی مدل روی داده‌های تست.
        """
        y_true = []
        y_pred = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                input_ids, labels = batch["input_ids"].to(self.device), batch["label"].to(self.device)

                # پیش‌بینی خروجی مدل
                outputs = self.model(input_ids)
                predictions = torch.argmax(outputs.logits, dim=1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())

        # محاسبه معیارهای ارزیابی
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

        print("\n📌 **نتایج ارزیابی مدل:**")
        print(f"🔹 `Accuracy`: {accuracy:.4f}")
        print(f"🔹 `Precision`: {precision:.4f}")
        print(f"🔹 `Recall`: {recall:.4f}")
        print(f"🔹 `F1-score`: {f1:.4f}")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


# ==================== تست ====================
if __name__ == "__main__":
    evaluator = Evaluator(num_labels=3)
    results = evaluator.evaluate()

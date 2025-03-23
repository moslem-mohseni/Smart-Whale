import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from models.base_model.config import BaseModelConfig


class PersianBERTModel:
    """
    کلاس مدیریت مدل ParsBERT برای فاین‌تیونینگ در پردازش زبان فارسی.
    """

    def __init__(self, num_labels=2):
        """
        مقداردهی اولیه مدل.

        :param num_labels: تعداد کلاس‌های خروجی (مثلاً برای طبقه‌بندی دودویی `2`)
        """
        self.device = torch.device(BaseModelConfig.DEVICE)

        # بارگذاری مدل ParsBERT
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BaseModelConfig.MODEL_NAME, num_labels=num_labels
        ).to(self.device)

        # بارگذاری tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BaseModelConfig.MODEL_NAME)

    def tokenize_text(self, text, max_length=512):
        """
        توکنیزه کردن متن با استفاده از tokenizer مدل ParsBERT.

        :param text: متن ورودی
        :param max_length: حداکثر طول توکن‌ها
        :return: دیکشنری شامل `input_ids`, `attention_mask`
        """
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def predict(self, text):
        """
        پیش‌بینی برچسب یک متن ورودی.

        :param text: متن ورودی
        :return: لیست نمرات خروجی مدل برای کلاس‌ها
        """
        self.model.eval()
        inputs = self.tokenize_text(text)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        return torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    def save_model(self, save_path=None):
        """
        ذخیره مدل فاین‌تیون‌شده.

        :param save_path: مسیر ذخیره مدل (در صورت عدم تعیین، مقدار پیش‌فرض `BaseModelConfig` استفاده می‌شود)
        """
        save_path = save_path or BaseModelConfig.MODEL_SAVE_PATH
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ مدل در `{save_path}` ذخیره شد.")

    def load_model(self, load_path=None):
        """
        بارگذاری مدل ذخیره‌شده.

        :param load_path: مسیر مدل ذخیره‌شده (در صورت عدم تعیین، مقدار پیش‌فرض `BaseModelConfig` استفاده می‌شود)
        """
        load_path = load_path or BaseModelConfig.MODEL_SAVE_PATH
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        print(f"✅ مدل از `{load_path}` بارگذاری شد.")


# ==================== تست ====================
if __name__ == "__main__":
    model = PersianBERTModel(num_labels=3)

    test_text = "هوش مصنوعی در حال تغییر جهان است."
    predictions = model.predict(test_text)

    print("📌 نمرات پیش‌بینی:", predictions)
    model.save_model()
    model.load_model()

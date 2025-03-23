import json
import torch
from torch.utils.data import Dataset, DataLoader
from models.base_model.config import BaseModelConfig
from preprocessing.tokenizer import PersianTokenizer
from preprocessing.normalizer import PersianNormalizer
from preprocessing.stopwords import PersianStopWords
from preprocessing.spellchecker import PersianSpellChecker


class PersianDataset(Dataset):
    """
    کلاس مدیریت داده‌های فارسی برای آموزش مدل ParsBERT.
    """

    def __init__(self, data_path, max_length=512):
        """
        :param data_path: مسیر فایل داده‌های آموزشی
        :param max_length: حداکثر طول توکن‌ها برای ورودی مدل
        """
        self.data_path = data_path
        self.max_length = max_length
        self.tokenizer = PersianTokenizer()
        self.normalizer = PersianNormalizer()
        self.stopwords_remover = PersianStopWords()
        self.spell_checker = PersianSpellChecker()

        # بارگذاری داده‌ها
        self.data = self._load_data()

    def _load_data(self):
        """
        بارگذاری داده‌های آموزشی از فایل JSON.
        """
        with open(self.data_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _preprocess_text(self, text):
        """
        پیش‌پردازش متن: نرمال‌سازی، حذف کلمات توقف، تصحیح املایی و توکن‌سازی.
        """
        text = self.normalizer.normalize(text)
        text = self.spell_checker.correct_text(text)
        text = self.stopwords_remover.remove_stopwords(text)
        tokens = self.tokenizer.tokenize(text)

        return tokens[:self.max_length]  # محدود کردن طول ورودی

    def __len__(self):
        """
        تعداد نمونه‌های داده.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        دریافت یک نمونه از داده‌ها.
        """
        sample = self.data[idx]
        text = sample["text"]
        label = sample.get("label", 0)  # مقدار پیش‌فرض برای برچسب

        # پردازش متن
        tokens = self._preprocess_text(text)

        return {
            "input_ids": tokens,
            "label": torch.tensor(label, dtype=torch.long)
        }


def get_dataloader(data_path, batch_size, shuffle=True):
    """
    ایجاد DataLoader برای مدیریت داده‌ها در `PyTorch`.
    """
    dataset = PersianDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)


# ==================== تست ====================
if __name__ == "__main__":
    train_loader = get_dataloader(BaseModelConfig.TRAIN_DATA_PATH, BaseModelConfig.BATCH_SIZE)

    for batch in train_loader:
        print("📌 نمونه داده:")
        print("توکن‌ها:", batch["input_ids"][:5])
        print("برچسب‌ها:", batch["label"][:5])
        break  # فقط یک نمونه نمایش داده شود

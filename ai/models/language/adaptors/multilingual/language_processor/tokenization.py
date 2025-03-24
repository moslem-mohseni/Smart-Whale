import nltk
import re
import importlib


class Tokenizer:
    def __init__(self, language="multilingual"):
        self.language = language
        self.smart_model = self._load_smart_model()
        self.teacher = self._load_teacher_model()

    def _load_smart_model(self):
        return importlib.import_module(f"ai.models.language.adaptors.{self.language}.smart_model").SmartModel()

    def _load_teacher_model(self):
        try:
            return importlib.import_module(f"ai.models.language.adaptors.{self.language}.teacher").TeacherModel()
        except ModuleNotFoundError:
            return None

    def basic_tokenize(self, text):
        """
        توکن‌سازی پایه‌ای با استفاده از `nltk`
        """
        return nltk.word_tokenize(text)

    def clean_text(self, text):
        """
        حذف نویزهای متنی و کاراکترهای اضافی
        """
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()

    def tokenize(self, text):
        """
        مدیریت فرآیند توکن‌سازی با یادگیری از مدل هوشمند و معلم
        """
        text = self.clean_text(text)

        if self.teacher:
            teacher_tokens = self.teacher.tokenize(text)
            self.smart_model.learn_from_teacher(text, teacher_tokens)
            return teacher_tokens

        return self.smart_model.tokenize(text)

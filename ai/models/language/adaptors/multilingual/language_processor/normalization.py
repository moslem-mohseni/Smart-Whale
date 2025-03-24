import re
import importlib


class Normalizer:
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

    def remove_punctuation(self, text):
        """
        حذف نشانه‌گذاری و کاراکترهای خاص
        """
        return re.sub(r'[!"#$%&()*+,./:;<=>?@[\\]^_`{|}~]', '', text)

    def normalize_text(self, text):
        """
        فرآیند نرمال‌سازی متن شامل حذف فاصله‌های اضافی، اعداد و نویسه‌های غیرضروری
        """
        text = text.lower()
        text = self.remove_punctuation(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self, text):
        """
        نرمال‌سازی متن با استفاده از مدل معلم در صورت موجود بودن
        """
        text = self.normalize_text(text)

        if self.teacher:
            teacher_output = self.teacher.normalize(text)
            self.smart_model.learn_from_teacher(text, teacher_output)
            return teacher_output

        return self.smart_model.normalize(text)

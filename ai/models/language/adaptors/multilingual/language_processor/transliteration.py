import importlib


class Transliterator:
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

    def transliterate(self, text):
        """
        تبدیل نویسه‌های متن بر اساس مدل معلم در صورت موجود بودن
        """
        if self.teacher:
            teacher_output = self.teacher.transliterate(text)
            self.smart_model.learn_from_teacher(text, teacher_output)
            return teacher_output

        return self.smart_model.transliterate(text)

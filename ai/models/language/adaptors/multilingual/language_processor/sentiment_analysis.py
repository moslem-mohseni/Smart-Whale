import importlib


class SentimentAnalyzer:
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

    def analyze_sentiment(self, text):
        """
        تحلیل احساسات متن با استفاده از مدل معلم در صورت موجود بودن
        """
        if self.teacher:
            teacher_output = self.teacher.analyze_sentiment(text)
            self.smart_model.learn_from_teacher(text, teacher_output)
            return teacher_output

        return self.smart_model.analyze_sentiment(text)

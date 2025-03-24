import importlib


class GrammarProcessor:
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

    def correct_grammar(self, text):
        """
        اصلاح گرامر متن با استفاده از مدل معلم در صورت موجود بودن
        """
        if self.teacher:
            teacher_output = self.teacher.correct_grammar(text)
            self.smart_model.learn_from_teacher(text, teacher_output)
            return teacher_output

        return self.smart_model.correct_grammar(text)

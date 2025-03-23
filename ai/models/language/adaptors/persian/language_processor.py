import importlib
from config import CONFIG
from .smart_model import SmartModel
from .teacher import TeacherModel


class PersianLanguageProcessor:
    """
    پردازش زبان فارسی و مدیریت تعامل با مدل معلم و مدل یادگیرنده Smart Whale.
    """

    def __init__(self):
        self.smart_model = SmartModel()
        self.teacher = TeacherModel() if CONFIG.get("use_hazm", True) else None

    def process_text(self, text):
        """
        پردازش ورودی متنی با مدل یادگیرنده و در صورت نیاز مدل معلم.
        """
        embedding = self.smart_model.forward(text)
        if self.teacher and self.smart_model.confidence_level(text) < CONFIG["confidence_threshold"]:
            teacher_output = self.teacher.forward(text)
            self.smart_model.train_model(text, teacher_output)
            return teacher_output
        return embedding

    def refine_model(self, training_data):
        """
        آموزش مدل بر اساس داده‌های جدید.
        """
        for input_text, target_text in training_data:
            self.smart_model.train_model(input_text, target_text)

    def save_models(self):
        """
        ذخیره مدل‌های پردازش زبان.
        """
        self.smart_model.save_model()
        if self.teacher:
            self.teacher.save_model()

    def load_models(self):
        """
        بارگذاری مدل‌های پردازش زبان.
        """
        self.smart_model.load_model()
        if self.teacher:
            self.teacher.load_model()
